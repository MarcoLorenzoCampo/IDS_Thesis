import json
import logging
import os
import threading

import boto3
import optuna
import time

import pandas as pd

from HypertunerStorage import Storage
from Optimizer import Optimizer
from Shared import LoggerConfig, Utils
from Shared.SQSWrapper import Connector

logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
filename = os.path.splitext(os.path.basename(__file__))[0]
LOGGER = logging.getLogger(filename)


class Hypertuner:
    FULL_CLOSE = False
    DEBUG = True
    OPTIMIZATION_LOCK = False

    def __init__(self, ):

        self.storage = Storage()
        self.optimizer = Optimizer(n_trials=7)
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

        study_l1 = optuna.create_study(study_name=f'Layer1 optimization', direction="maximize")
        study_l2 = optuna.create_study(study_name=f'Layer2 optimization', direction="maximize")

        study_l1.optimize(lambda trial: self.optimizer.optimize_wrapper(fun_calls1, trial))
        LOGGER.info(f"Found new optimal hyperparameters for layer 1: {study_l1.best_params}")

        study_l2.optimize(lambda trial: self.optimizer.optimize_wrapper(fun_calls2, trial))
        LOGGER.info(f"Found new optimal hyperparameters for layer 2: {study_l2.best_params}")

        self.new_opt_layer1 = self.optimizer.train_new_hps('RandomForest', study_l1.best_params)
        self.new_opt_layer2 = self.optimizer.train_new_hps('SVM', study_l2.best_params)

        self.optimizer.unset()
        self.OPTIMIZATION_LOCK = False

        return self.new_opt_layer1, self.new_opt_layer2

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