import json
import os
import threading

import boto3
import optuna
import time

import pandas as pd

from HypertunerStorage import Storage
from Optimizer import Optimizer
from Shared import Utils
from Shared.SQSWrapper import Connector


LOGGER = Utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])
pd.set_option('display.width', 1000)


class Hypertuner:

    FULL_CLOSE = False
    DEBUG = True
    N_TRIALS = 2

    OPTIMIZATION_LOCK = False

    def __init__(self, ):

        self.storage = Storage()
        self.optimizer = Optimizer()
        self.__sqs_setup()

        # new optimal models
        self.new_opt_layer1 = None
        self.new_opt_layer2 = None

    def __sqs_setup(self):
        self.sqs_client = boto3.client('sqs')
        self.sqs_resource = boto3.resource('sqs')

        self.queue_urls = [
            'https://sqs.eu-west-3.amazonaws.com/818750160971/forward-objectives.fifo',
        ]
        self.queue_names = [
            'tuned-models.fifo'
        ]

        self.connector = Connector(
            sqs_client=self.sqs_client,
            sqs_resource=self.sqs_resource,
            queue_urls=self.queue_urls,
            queue_names=self.queue_names
        )

    def __objs_map(self, objectives: dict):

        self.OPTIMIZATION_LOCK = True

        self.__optimizer_setup()

        LOGGER.info("MAXIMIZE direction only, evaluating 1/FNR and 1/FPR for tuning.")
        function_mapping = {
            'accuracy': [self.optimizer.objective_accuracy_l1, self.optimizer.objective_accuracy_l2],
            'precision': [self.optimizer.objective_precision_l1, self.optimizer.objective_precision_l2],
            'fscore': [self.optimizer.objective_fscore_l1, self.optimizer.objective_fscore_l2],
            'tpr': [self.optimizer.objective_tpr_l1, self.optimizer.objective_tpr_l2],
            'tnr': [self.optimizer.objective_tnr_l1, self.optimizer.objective_tnr_l2],
            'fnr': [self.optimizer.objective_inv_fpr_l1, self.optimizer.objective_inv_fpr_l2],
            'fpr': [self.optimizer.objective_inv_fnr_l1, self.optimizer.objective_inv_fnr_l2],
        }

        # Obtain the functions to optimize and combine their outputs into a single objective
        objs_l1 = objectives['layer1']
        objs_l2 = objectives['layer2']

        fun_calls1 = []
        for obj in objs_l1:
            fun_calls1.append(function_mapping[obj][0])

        fun_calls2 = []
        for obj in objs_l2:
            fun_calls2.append(function_mapping[obj][1])

        if self.DEBUG:
            print(f'# funcs l1: {len(fun_calls1)}, # funcs l2: {len(fun_calls2)}')

        new_opt_layer1 = self.__layer1_tuning(fun_calls1, objs_l1)
        new_opt_layer2 = self.__layer2_tuning(fun_calls2, objs_l2)

        self.optimizer.unset()
        self.OPTIMIZATION_LOCK = False

        return new_opt_layer1, new_opt_layer2

    def __layer1_tuning(self, fun_calls1: list, objectives):

        study_l1 = optuna.create_study(
            study_name=f'Layer1 optimization',
            directions=['maximize' for _ in fun_calls1],
        )

        study_l1.optimize(lambda trial: self.optimizer.optimize_wrapper(fun_calls1, trial), n_trials=self.N_TRIALS)

        best_hps = self.__get_hps_from_trials(study_l1, objectives)

        LOGGER.info(f"Found new optimal hyperparameters for layer 1: {study_l1}")

        self.optimizer.train_new_hps('RandomForest', best_hps)

    def __layer2_tuning(self, fun_calls2: list, objectives):

        study_l2 = optuna.create_study(
            study_name=f'Layer2 optimization',
            directions=['maximize' for _ in fun_calls2],
        )

        study_l2.optimize(lambda trial: self.optimizer.optimize_wrapper(fun_calls2, trial), n_trials=self.N_TRIALS)
        LOGGER.info(f"Found new optimal hyperparameters for layer 2: {study_l2.best_params}")

        best_hps = self.__get_hps_from_trials(study_l2, objectives)

        return self.optimizer.train_new_hps('SVM', best_hps)

    def __get_pareto_front_hps(self, study: optuna.study.Study, objective_names):
        # Get all trials and their objective values as a DataFrame
        trials_df = study.trials_dataframe()

        if self.DEBUG:
            print(trials_df.columns)

        pareto_hps = {}

        for objective_name in objective_names:
            # Sort trials by the current objective in ascending order (for minimization)
            sorted_trials = trials_df.sort_values(by=f"user_attrs_{objective_name}")

            # Extract hyperparameters from the trial with the best objective value
            best_trial_params = sorted_trials["params"].iloc[0]
            pareto_hps[objective_name] = best_trial_params

        return pareto_hps

    def __get_hps_from_trials(self, study: optuna.study.Study, objective_names):

        best_trials = study.best_trials

        pareto_hps = self.__get_pareto_front_hps(study, objective_names)

        if self.DEBUG:
            for trial in best_trials:
                print(f'Trial: {trial.number}, Params: {trial.params}')
                print(f'Selected hyperparameters: {pareto_hps}')

        return pareto_hps

    def __optimizer_setup(self):

        query = {
            'select': "*"
        }

        LOGGER.info('Obtaining the datasets from SQLite3 in memory database.')
        datasets = []
        for dataset in ['x_train_l1', 'x_train_l2', 'x_validate_l1', 'x_validate_l2']:
            query['from'] = str(dataset)
            result = self.storage.perform_query(query)
            result_df = pd.DataFrame(result)
            datasets.append(result_df)

        self.optimizer.dataset_setter(
            x_train_1=datasets[0],
            x_train_2=datasets[1],
            x_val_1=datasets[2],
            x_val_2=datasets[3],
            y_train_1=datasets[0]['targets'],
            y_train_2=datasets[1]['targets'],
            y_val_1=datasets[2]['targets'],
            y_val_2=datasets[3]['targets'],
        )

        if self.DEBUG:
            print(
                "x_train_1: ", datasets[0].shape,
                "\nx_train_2: ", datasets[1].shape,
                "\nx_validate_1: ", datasets[2].shape,
                "\nx_validate_2: ", datasets[3].shape,
                "\ny_train_1", datasets[0]['targets'].shape,
                "\ny_train_2", datasets[1]['targets'].shape,
                "\ny_val_1", datasets[2]['targets'].shape,
                "\ny_val_2", datasets[3]['targets'].shape
            )
        pass


    def terminate(self):
        self.FULL_CLOSE = True
        self.connector.close()

    def poll_queues(self):
        while True:
            LOGGER.info('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()
            except Exception as e:
                LOGGER.error(f"Error in fetching messages from queue: {e}")
                raise KeyboardInterrupt

            if msg_body:
                if not self.OPTIMIZATION_LOCK:
                    LOGGER.info(f'Parsing message: {json.dumps(msg_body, indent=2)}')
                    objectives = Utils.parse_objs(msg_body)
                    self.__objs_map(objectives)
                else:
                    LOGGER.info('Process locked. Optimization in progress.')

            time.sleep(2)

    def run_tasks(self):
        queue_reading_thread = threading.Thread(target=self.poll_queues, daemon=True)

        queue_reading_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            Utils.save_current_timestamp()

        finally:
            LOGGER.info('Terminating Hypertuner instance.')
            raise KeyboardInterrupt


if __name__ == '__main__':
    hypertuner = Hypertuner()

    try:
        hypertuner.run_tasks()
    except KeyboardInterrupt:
        if hypertuner.FULL_CLOSE:
            hypertuner.terminate()
            LOGGER.info('Deleting queues..')
        else:
            LOGGER.info('Received keyboard interrupt. Preparing to terminate threads.')