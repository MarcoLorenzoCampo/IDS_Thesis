import os
import threading
import time
from abc import ABC, abstractmethod

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.svm import SVC

from TunerProcess.storage import SQLiteManager


class TemporaryStorage:

    def __init__(self, sqlite_manager: SQLiteManager):
        import hypertuner_main
        self.LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        self.y_validate_l2 = None
        self.y_validate_l1 = None
        self.y_train_l2 = None
        self.x_train_l1 = None
        self.x_train_l2 = None
        self.x_validate_l1 = None
        self.x_validate_l2 = None
        self.y_train_l1 = None

        self.sqlite_manager = sqlite_manager
        self.prepare_temp_storage()

    def prepare_temp_storage(self):
        query = {
            'select': "*"
        }

        self.LOGGER.debug('Obtaining the datasets from SQLite3 in memory database.')
        datasets = []
        for dataset in ['x_train_l1', 'x_train_l2', 'x_validate_l1', 'x_validate_l2']:
            query['from'] = str(dataset)
            result = self.sqlite_manager.perform_query(query)
            result_df = pd.DataFrame(result)
            datasets.append(result_df)

        self.x_train_l1 = datasets[0]
        self.x_train_l2 = datasets[1]
        self.x_validate_l1 = datasets[2]
        self.x_validate_l2 = datasets[3]
        self.y_train_l1 = datasets[0]['targets']
        self.y_train_l2 = datasets[1]['targets']
        self.y_validate_l1 = datasets[2]['targets']
        self.y_validate_l2 = datasets[3]['targets']
        self.x_train_l1 = self.x_train_l1.drop(self.x_train_l1.columns[-1], axis=1)
        self.x_train_l2 = self.x_train_l2.drop(self.x_train_l2.columns[-1], axis=1)
        self.x_validate_l1 = self.x_validate_l1.drop(self.x_validate_l1.columns[-1], axis=1)
        self.x_validate_l2 = self.x_validate_l2.drop(self.x_validate_l2.columns[-1], axis=1)


class AbstractTrainer(ABC):

    @abstractmethod
    def train(self, parameters: dict):
        pass

    @abstractmethod
    def set_temp_storage(self, temp_storage: TemporaryStorage):
        pass


class RFTrainer(AbstractTrainer):

    def __init__(self):
        import hypertuner_main
        self.LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        self.temp_storage = None

    def set_temp_storage(self, temp_storage: TemporaryStorage):
        self.temp_storage = temp_storage

    def train(self, parameters: dict):
        classifier = RandomForestClassifier(
            n_estimators=parameters.get('n_estimators', 10),
            criterion=parameters.get('criterion', 'gini'),
            max_depth=parameters.get('max_depth', None),
            min_samples_split=parameters.get('min_samples_split', 2),
            min_samples_leaf=parameters.get('min_samples_leaf', 1),
            min_weight_fraction_leaf=parameters.get('min_weight_fraction_leaf', 0.0),
            max_features=parameters.get('max_features', 'sqrt'),
            max_leaf_nodes=parameters.get('max_leaf_nodes', None),
            min_impurity_decrease=parameters.get('min_impurity_decrease', 0.0),
            bootstrap=parameters.get('bootstrap', True),
            oob_score=parameters.get('oob_score', False),
            n_jobs=parameters.get('n_jobs', None),
            random_state=parameters.get('random_state', None),
            verbose=0,
            warm_start=parameters.get('warm_start', False),
            class_weight=parameters.get('class_weight', None),
            ccp_alpha=parameters.get('ccp_alpha', 0.0),
            max_samples=parameters.get('max_samples', None)
        )

        classifier.fit(self.temp_storage.x_train_l1, self.temp_storage.y_train_l1)
        self.LOGGER.debug('Trained a new RandomForest classifier.')
        return classifier


class SVMTrainer(AbstractTrainer):

    def __init__(self):
        import hypertuner_main
        self.LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        self.temp_storage = None

    def set_temp_storage(self, temp_storage: TemporaryStorage):
        self.temp_storage = temp_storage

    def train(self, parameters: dict):
        classifier = SVC(
            C=parameters.get('C', 10),
            kernel=parameters.get('kernel', 'rbf'),
            degree=parameters.get('degree', 3),
            gamma=parameters.get('gamma', 0.01),
            coef0=parameters.get('coef0', 0.0),
            shrinking=parameters.get('shrinking', True),
            probability=True,
            tol=parameters.get('tol', 1e-3),
            cache_size=parameters.get('cache_size', 200),
            class_weight=parameters.get('class_weight', None),
            verbose=False,
            max_iter=parameters.get('max_iter', -1),
            decision_function_shape=parameters.get('decision_function_shape', 'ovr')
        )

        self.LOGGER.debug('Trained a new SVM classifier.')
        classifier.fit(self.temp_storage.x_train_l2, self.temp_storage.y_train_l2)
        return classifier


class Optimizer(ABC):

    def __init__(self, trainer: AbstractTrainer, temp_storage: TemporaryStorage):
        self.trainer = trainer
        self.temp_storage = temp_storage

    @abstractmethod
    def optimize(self, trial: optuna.Trial):
        pass


class TPROptimizer_layer1(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("TPR optimizer for layer2")

        rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
        rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        min_samples_split = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)
        min_samples_leaf = trial.suggest_int(name='min_samples_leaf', low=1, high=10, step=1)
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])

        parameters = {
            'max_depth': rf_max_depth,
            'criterion': rf_criterion,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l1)

        tp = sum((self.temp_storage.y_validate_l1 == 1) & (predicted == 1))
        fn = sum((self.temp_storage.y_validate_l1 == 1) & (predicted == 0))
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

        return tpr


class TPROptimizer_layer2(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("TPR optimizer for layer2")

        svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)
        kernel = trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid'])

        # add to parameters all the hyperparameters that need tuning
        parameters = {
            'C': svc_c,
            'kernel': kernel
        }

        if kernel == 'rbf':
            gamma = trial.suggest_float(name='gamma', low=1e-10, high=1e10)
            parameters['gamma'] = gamma

        if kernel in ['sigmoid', 'poly']:
            degree = trial.suggest_int('degree', 2, 5)
            coef0 = trial.suggest_float('coef0', -1.0, 1.0)
            parameters['degree'] = degree
            parameters['coef0'] = coef0

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l2)

        tp = sum((self.temp_storage.y_validate_l2 == 1) & (predicted == 1))
        fn = sum((self.temp_storage.y_validate_l2 == 1) & (predicted == 0))
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

        return tpr


class FPROptimizer_layer1(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("FPR optimizer for layer1")

        min_samples_split = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)
        min_samples_leaf = trial.suggest_int(name='min_samples_leaf', low=1, high=10, step=1)
        max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])

        parameters = {
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l1)

        fp = sum((self.temp_storage.y_validate_l1 == 0) & (predicted == 1))
        tn = sum((self.temp_storage.y_validate_l1 == 0) & (predicted == 0))
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

        try:
            return 1 / fpr
        except ZeroDivisionError:
            return 0.0


class FPROptimizer_layer2(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("FPR optimizer for layer2")

        svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)
        kernel = trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid'])

        # add to parameters all the hyperparameters that need tuning
        parameters = {
            'C': svc_c,
            'kernel': kernel
        }

        if kernel == 'rbf':
            gamma = trial.suggest_float(name='gamma', low=1e-10, high=1e10)
            parameters['gamma'] = gamma

        if kernel in ['sigmoid', 'poly']:
            degree = trial.suggest_int('degree', 2, 5)
            coef0 = trial.suggest_float('coef0', -1.0, 1.0)
            parameters['degree'] = degree
            parameters['coef0'] = coef0

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l2)

        fp = sum((self.temp_storage.y_validate_l2 == 0) & (predicted == 1))
        tn = sum((self.temp_storage.y_validate_l2 == 0) & (predicted == 0))
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

        try:
            return 1 / fpr
        except ZeroDivisionError:
            return 0.0


class TNROptimizer_layer1(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("TNR optimizer for layer1")

        rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
        rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
        rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        min_samples_split = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)
        max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])

        parameters = {
            'n_estimators': rf_n_estimators,
            'max_depth': rf_max_depth,
            'criterion': rf_criterion,
            'max_features': max_features,
            'min_samples_split': min_samples_split
        }

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l1)

        tn = sum((self.temp_storage.y_validate_l1 == 0) & (predicted == 0))
        fp = sum((self.temp_storage.y_validate_l1 == 0) & (predicted == 1))
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0

        return tnr


class TNROptimizer_layer2(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("TNR optimizer for layer2")

        svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)
        kernel = trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid'])

        # add to parameters all the hyperparameters that need tuning
        parameters = {
            'C': svc_c,
            'kernel': kernel
        }

        if kernel == 'rbf':
            gamma = trial.suggest_float(name='gamma', low=1e-10, high=1e10)
            parameters['gamma'] = gamma

        if kernel in ['sigmoid', 'poly']:
            degree = trial.suggest_int('degree', 2, 5)
            coef0 = trial.suggest_float('coef0', -1.0, 1.0)
            parameters['degree'] = degree
            parameters['coef0'] = coef0

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l2)

        tn = sum((self.temp_storage.y_validate_l2 == 0) & (predicted == 0))
        fp = sum((self.temp_storage.y_validate_l2 == 0) & (predicted == 1))
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0

        return tnr


class FNROptimizer_layer1(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("FNR optimizer for layer1")

        rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
        rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        min_samples_split = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

        parameters = {
            'max_depth': rf_max_depth,
            'criterion': rf_criterion,
            'max_features': min_samples_split
        }

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l1)

        fn = sum((self.temp_storage.y_validate_l1 == 1) & (predicted == 0))
        tp = sum((self.temp_storage.y_validate_l1 == 1) & (predicted == 1))
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0

        try:
            return 1 / fnr
        except ZeroDivisionError:
            return 0.0


class FNROptimizer_layer2(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("FNR optimizer for layer2")

        svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)
        kernel = trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid'])

        # add to parameters all the hyperparameters that need tuning
        parameters = {
            'C': svc_c,
            'kernel': kernel
        }

        if kernel == 'rbf':
            gamma = trial.suggest_float(name='gamma', low=1e-10, high=1e10)
            parameters['gamma'] = gamma

        if kernel in ['sigmoid', 'poly']:
            degree = trial.suggest_int('degree', 2, 5)
            coef0 = trial.suggest_float('coef0', -1.0, 1.0)
            parameters['degree'] = degree
            parameters['coef0'] = coef0

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l2)

        fn = sum((self.temp_storage.y_validate_l2 == 1) & (predicted == 0))
        tp = sum((self.temp_storage.y_validate_l2 == 1) & (predicted == 1))
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0

        try:
            return 1 / fnr
        except ZeroDivisionError:
            return 0.0


class AccuracyOptimizer_layer1(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("Accuracy optimizer for layer1")

        rf_n_estimators = trial.suggest_int(name='n_estimators', low=10, high=100, step=5)
        max_features = trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])

        parameters = {
            'n_estimators': rf_n_estimators,
            'max_features': max_features
        }
        LOGGER.debug(f"Parameters: n_estimators={rf_n_estimators}, max_features={max_features}")

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l1)

        accuracy = accuracy_score(self.temp_storage.y_validate_l1, predicted)
        return accuracy


class AccuracyOptimizer_layer2(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("Accuracy optimizer for layer2")

        svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)
        kernel = trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid'])

        # add to parameters all the hyperparameters that need tuning
        parameters = {
            'C': svc_c,
            'kernel': kernel
        }

        if kernel == 'rbf':
            gamma = trial.suggest_float(name='gamma', low=1e-10, high=1e10)
            parameters['gamma'] = gamma

        if kernel in ['sigmoid', 'poly']:
            degree = trial.suggest_int('degree', 2, 5)
            coef0 = trial.suggest_float('coef0', -1.0, 1.0)
            parameters['degree'] = degree
            parameters['coef0'] = coef0

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l2)

        accuracy = accuracy_score(self.temp_storage.y_validate_l2, predicted)
        return accuracy


class PrecisionOptimizer_layer1(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("Precision optimizer for layer1")

        rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
        rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        min_samples_split = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)
        min_samples_leaf = trial.suggest_int(name='min_samples_leaf', low=2, high=10, step=1)
        rf_n_estimators = trial.suggest_int(name='n_estimators', low=10, high=100, step=5)

        parameters = {
            'n_estimators': rf_n_estimators,
            'max_depth': rf_max_depth,
            'criterion': rf_criterion,
            'max_features': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        }

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l1)

        precision = precision_score(self.temp_storage.y_validate_l1, predicted)

        return precision


class PrecisionOptimizer_layer2(Optimizer):

    def optimize(self, trial: optuna.Trial) -> float:
        import hypertuner_main
        LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        LOGGER.info("Precision optimizer for layer2")

        svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

        # add to parameters all the hyperparameters that need tuning
        parameters = {
            'C': svc_c
        }

        classifier = self.trainer.train(parameters)
        predicted = classifier.predict(self.temp_storage.x_validate_l2)

        precision = precision_score(self.temp_storage.y_validate_l2, predicted)

        return precision


class OptimizersFactory:

    @staticmethod
    def create_optimizer_object(objective: str, trainer: AbstractTrainer, temp_storage: TemporaryStorage,
                                layer: int) -> Optimizer:
        if objective == 'tpr':
            return TPROptimizer_layer1(trainer, temp_storage) if layer == 1 else TPROptimizer_layer2(trainer,
                                                                                                     temp_storage)
        if objective == 'tnr':
            return TNROptimizer_layer1(trainer, temp_storage) if layer == 1 else TNROptimizer_layer2(trainer,
                                                                                                     temp_storage)
        if objective == 'fnr':
            return FNROptimizer_layer1(trainer, temp_storage) if layer == 1 else FNROptimizer_layer2(trainer,
                                                                                                     temp_storage)
        if objective == 'fpr':
            return FPROptimizer_layer1(trainer, temp_storage) if layer == 1 else FPROptimizer_layer2(trainer,
                                                                                                     temp_storage)
        if objective == 'precision':
            return PrecisionOptimizer_layer1(trainer, temp_storage) if layer == 1 else PrecisionOptimizer_layer2(
                trainer, temp_storage)
        if objective == 'accuracy':
            return AccuracyOptimizer_layer1(trainer, temp_storage) if layer == 1 else AccuracyOptimizer_layer2(
                trainer, temp_storage)


class OptimizationManager:
    DEBUG = True

    def __init__(self, sqlite_manager: SQLiteManager, rf_trainer: AbstractTrainer, svm_trainer: AbstractTrainer):

        import hypertuner_main
        self.LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        self.sqlite_manager = sqlite_manager
        self.rf_trainer = rf_trainer
        self.svm_trainer = svm_trainer

        self.temp_storage = None

    def tear_down_storage(self):
        del self.temp_storage

    def prepare_trainers_and_storage(self):
        self.temp_storage = TemporaryStorage(self.sqlite_manager)

        self.svm_trainer.set_temp_storage(self.temp_storage)
        self.rf_trainer.set_temp_storage(self.temp_storage)

    def optimizers_mapper(self, objectives: list[str], layer: int):
        optimizers = [
            OptimizersFactory.create_optimizer_object(objective, self.rf_trainer, self.temp_storage, layer)
            for objective in objectives
        ]

        return optimizers

    def optimization_wrapper(self, optimizers: list[Optimizer], trial: optuna.Trial):
        """
        This function wraps the objective functions and optimizes them using Optuna.
        :return: A list of outputs from the objective functions.
        """
        self.LOGGER.debug("Entering optimization_wrapper")

        outputs = []
        for optimizer in optimizers:
            self.LOGGER.debug(f"ID: {threading.get_ident()} -> Processing optimizer: {optimizer}")

            objective_function = optimizer.optimize
            try:
                output_value = objective_function(trial)
                outputs.append(output_value)

            except Exception as e:
                self.LOGGER.debug(f"Error in function {objective_function.__name__}: {str(e)}")

        self.LOGGER.debug("Exiting optimization_wrapper")
        return [output for output in outputs]