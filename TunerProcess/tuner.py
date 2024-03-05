import os
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime

import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

from Shared import utils
from Shared.msg_enum import msg_type
from Shared.utils import LOGGER
from TunerProcess.optimizer import OptimizationManager, Optimizer
from TunerProcess.storage import Storage


class LayerTuner(ABC):

    @abstractmethod
    def tune_layer(self, fun_calls: list[callable]):
        pass


class TunerLayer1(LayerTuner):

    def __init__(self, n_trials: int, n_cores: int, optimizer: OptimizationManager):
        self.n_trials = n_trials
        self.n_cores = n_cores
        self.optimizer = optimizer

    def tune_layer(self, optimizers: list[Optimizer]) -> dict:

        study_l1 = optuna.create_study(
            study_name='Layer1 optimization',
            directions=['maximize' for _ in optimizers],
            #pruner=optuna.pruners.NopPruner()
        )

        study_l1.optimize(
            lambda trial: self.optimizer.optimization_wrapper(optimizers, trial),
            n_trials=self.n_trials,
            n_jobs=self.n_cores
        )

        return study_l1


class TunerLayer2(LayerTuner):

    def __init__(self, n_trials: int, n_cores: int, optimizer: OptimizationManager):
        self.n_trials = n_trials
        self.n_cores = n_cores
        self.optimizer = optimizer

    def tune_layer(self, fun_calls2: list):

        study_l2 = optuna.create_study(
            study_name='Layer2 optimization',
            directions=['maximize' for _ in fun_calls2]
        )

        study_l2.optimize(
            lambda trial: self.optimizer.optimization_wrapper(fun_calls2, trial),
            n_trials=self.n_trials,
            n_jobs=self.n_cores
        )

        return study_l2


class Tuner:

    def __init__(self, storage: Storage, optimization_manager: OptimizationManager, layer1_tuner: LayerTuner, layer2_tuner: LayerTuner):
        self.optimization_manager = optimization_manager
        self.storage = storage
        self.layer1_tuner = layer1_tuner
        self.layer2_tuner = layer2_tuner

    def tune(self, objectives: dict):

        objs_l1 = objectives['layer1']
        objs_l2 = objectives['layer2']

        self.optimization_manager.prepare_trainers_and_storage()

        if len(objs_l1) > 0:
            optimizers1 = self.optimization_manager.optimizers_mapper(objs_l1, 1)

            start = time.time()

            study_l1 = self.layer1_tuner.tune_layer(optimizers1)
            best_hps = self.__get_hps_from_trials(study_l1)
            new_layer1 = self.optimization_manager.rf_trainer.train(best_hps)

            LOGGER.debug(f"Found new optimal hyperparameters for layer 1: {best_hps}")
            tune_time = time.time() - start
            LOGGER.info(f"Optimization of layer1 took: {tune_time}")

            self.report_tuning_info(1, tune_time, best_hps)

            with open(f'TunedModels/l1_classifier.pkl', 'wb') as f:
                pickle.dump(new_layer1, f)
        else:
            LOGGER.warning('No new objectives received for layer1.')

        if len(objs_l2) > 0:
            optimizers2 = self.optimization_manager.optimizers_mapper(objs_l2, 2)

            start = time.time()

            study_l2 = self.layer2_tuner.tune_layer(optimizers2)
            best_hps = self.__get_hps_from_trials(study_l2)
            new_layer2 = self.optimization_manager.svm_trainer.train(best_hps)

            LOGGER.debug(f"Found new optimal hyperparameters for layer 2: {best_hps}")
            tune_time = time.time() - start
            LOGGER.info(f"Optimization of layer2 took: {tune_time}")

            self.report_tuning_info(2, tune_time, best_hps)

            with open('TunedModels/l2_classifier.pkl', 'wb') as f:
                pickle.dump(new_layer2, f)
        else:
            LOGGER.warning('No new objectives received for layer2.')

        self.optimization_manager.tear_down_storage()

    @staticmethod
    def __get_hps_from_trials(study: optuna.study.Study):
        trials = sorted(study.best_trials, key=lambda t: t.values)

        best_trial = trials[0]
        best_hps = best_trial.params

        return best_hps

    @staticmethod
    def report_tuning_info(layer, timing, hps):

        with open("optimal_tuning_hps.txt", "a") as f:
            f.write(f"Layer {layer}:\n")
            f.write(f"New optimal hyperparamters: [{hps}]\n")
            f.write(f"Training time: {timing}")


class TuningHandler:

    def __init__(self, storage: Storage, optimizer: OptimizationManager, tuner: Tuner):

        import hypertuner_main
        self.LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        self.storage = storage
        self.optimizer = optimizer
        self.tuner = tuner
        self.tuning_state = TuningStatusManager()

    def on_tuning_msg_received(self, json_dict: dict):

        self.tuning_state.set_start_tuning_state()

        if not self.tuning_state.is_optimization_locked():
            LOGGER.debug(f'Parsing message: {json_dict}')
            objectives = utils.parse_objs(json_dict)

            self.tuning_state.lock_optimization()
            self.tuner.tune(objectives)
            self.storage.s3_manager.publish_s3_models()

            LOGGER.debug('Models have been tuned and updated. Forwarding models update notification.')

            new_models_msg = {
                "MSG_TYPE": str(msg_type.MODEL_UPDATE_MSG),
                "SENDER": 'Hypertuner'
            }

            self.tuning_state.set_complete_tuning_state()

            return new_models_msg
        else:
            LOGGER.debug('Process locked. Optimization in progress.')

class TuningStatusManager:

    def __init__(self):
        self.OPTIMIZATION_LOCK = False
        self.COMPLETE_TUNING = True

    def set_complete_tuning_state(self):
        self.COMPLETE_TUNING = True
        self.OPTIMIZATION_LOCK = False

    def is_optimization_locked(self):
        return self.OPTIMIZATION_LOCK

    def is_tuning_complete(self):
        return self.COMPLETE_TUNING

    def set_start_tuning_state(self):
        self.COMPLETE_TUNING = False

    def lock_optimization(self):
        self.OPTIMIZATION_LOCK = True
