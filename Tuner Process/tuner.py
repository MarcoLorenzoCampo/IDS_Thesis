import pickle
from pprint import pprint

import optuna
import pandas as pd

from storage import Storage
from Shared.utils import LOGGER

from optimizer import Optimizer


class Tuner:
    DEBUG = False

    def __init__(self, n_trials: int, storage: Storage):
        self.storage = storage
        self.optimizer = Optimizer()
        self.N_TRIALS = n_trials

    def objs_map(self, objectives: dict):

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

        # Tuning for the first objectives set

        fun_calls1 = []
        for obj in objs_l1:
            fun_calls1.append(function_mapping[obj][0])

        if len(fun_calls1) > 0:
            if self.DEBUG:
                print(f'# funcs l1: {len(fun_calls1)}')

            new_layer1 = self.__layer1_tuning(fun_calls1)

            with open('AWS Downloads/Models/Tuned/NSL_l1_classifier.pkl', 'wb') as f:
                pickle.dump(new_layer1, f)
        else:
            LOGGER.info('No new objectives received for layer1.')

        # Tuning for the second objectives set

        fun_calls2 = []
        for obj in objs_l2:
            fun_calls2.append(function_mapping[obj][1])

        if len(fun_calls1) > 0:
            if self.DEBUG:
                print(f'# funcs l2: {len(fun_calls2)}')

            new_layer2 = self.__layer2_tuning(fun_calls2)

            with open('AWS Downloads/Models/Tuned/NSL_l2_classifier.pkl', 'wb') as f:
                pickle.dump(new_layer2, f)
        else:
            LOGGER.info('No new objectives received for layer2.')

        self.optimizer.unset()

    def __layer1_tuning(self, fun_calls1: list):

        study_l1 = optuna.create_study(
            study_name='Layer1 optimization',
            directions=['maximize' for _ in fun_calls1],
        )

        study_l1.optimize(
            lambda trial: self.optimizer.optimize_wrapper(fun_calls1, trial),
            n_trials=self.N_TRIALS
        )

        best_hps = self.__get_hps_from_trials(study_l1)

        LOGGER.info(f"Found new optimal hyperparameters for layer 1: {best_hps}")

        return self.optimizer.train_new_hps('RandomForest', best_hps)

    def __layer2_tuning(self, fun_calls2: list):

        study_l2 = optuna.create_study(
            study_name='Layer2 optimization',
            directions=['maximize' for _ in fun_calls2],
        )

        study_l2.optimize(
            lambda trial: self.optimizer.optimize_wrapper(fun_calls2, trial),
            n_trials=self.N_TRIALS,
        )

        best_hps = self.__get_hps_from_trials(study_l2)

        LOGGER.info(f"Found new optimal hyperparameters for layer 2: {best_hps}")

        return self.optimizer.train_new_hps('SVM', best_hps)

    def __get_hps_from_trials(self, study: optuna.study.Study):
        trials = sorted(study.best_trials, key=lambda t: t.values)

        if self.DEBUG:
            print(f'Number of best trials: {len(trials)}')
            for trial in trials:
                print("Trial#{}".format(trial.number))
                print("Values: Values={}".format(trial.values))
                print("Params: {}".format(trial.params))

        # for simplicity, we just take the first trial
        best_trial = trials[0]
        best_hps = best_trial.params

        if self.DEBUG:
            print(f'Best trial: {best_trial}')
            pprint(f'Best hyperparameters: {best_hps}')

        return best_hps

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
