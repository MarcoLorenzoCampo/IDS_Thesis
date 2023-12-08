import os
from typing import Callable, List

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score
from sklearn.svm import SVC

from Shared import Utils

LOGGER = Utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class Optimizer:

    DEBUG = True

    def __init__(self):
        pass

    def dataset_setter(self, x_train_1, x_train_2, y_train_1, y_train_2, x_val_1, y_val_1, x_val_2, y_val_2):
        self.x_train_l1 = x_train_1
        self.x_train_l2 = x_train_2
        self.y_train_l1 = y_train_1
        self.y_train_l2 = y_train_2
        self.x_validate_l1 = x_val_1
        self.y_validate_l1 = y_val_1
        self.x_validate_l2 = x_val_2
        self.y_validate_l2 = y_val_2

    def unset(self):
        LOGGER.info('Deleting training data and validation data.')
        del self.x_train_l1, self.x_train_l2
        del self.y_train_l1, self.y_train_l2
        del self.x_validate_l1, self.y_validate_l1
        del self.x_validate_l2, self.y_validate_l2

    def optimize_wrapper(self, fun_calls: List[Callable], trial: optuna.Trial):

        outputs = {}
        for fun_call in fun_calls:
            try:
                output_value = fun_call(trial)
                outputs[fun_call.__name__] = output_value

                LOGGER.info(f'Output: {output_value}')
            except Exception as e:
                LOGGER.info(f"Error in function {fun_call.__name__}: {str(e)}")

        LOGGER.info(f'Outputs: {outputs}')
        return [output for output in outputs.values()]

    def objective_tpr_l1(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        LOGGER.info(f'Calling optimization function: Objective TPR for layer 1')

        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            tp = sum((self.y_validate_l1 == 1) & (predicted == 1))
            fn = sum((self.y_validate_l1 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            return tpr

    def objective_tpr_l2(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective TPR for layer 2')

        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            tp = sum((self.y_validate_l2 == 1) & (predicted == 1))
            fn = sum((self.y_validate_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            return tpr

    def objective_inv_fpr_l1(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective FPR for layer 1')
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            fp = sum((self.y_validate_l1 == 0) & (predicted == 1))
            tn = sum((self.y_validate_l1 == 0) & (predicted == 0))
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

            try:
                return 1/fpr
            except ZeroDivisionError:
                return 0.0

    def objective_inv_fpr_l2(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective FPR for layer 2')

        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            fp = sum((self.y_validate_l2 == 0) & (predicted == 1))
            tn = sum((self.y_validate_l2 == 0) & (predicted == 0))
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

            try:
                return 1 / fpr
            except ZeroDivisionError:
                return 0.0

    def objective_tnr_l1(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        LOGGER.info(f'Calling optimization function: Objective TNR for layer 1')

        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            tn = sum((self.y_validate_l1 == 0) & (predicted == 0))
            fp = sum((self.y_validate_l1 == 0) & (predicted == 1))
            tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            return tnr

    def objective_tnr_l2(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective TNR for layer 2')

        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            tn = sum((self.y_validate_l2 == 0) & (predicted == 0))
            fp = sum((self.y_validate_l2 == 0) & (predicted == 1))
            tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            return tnr

    def objective_inv_fnr_l1(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        LOGGER.info(f'Calling optimization function: Objective FNR for layer 1')

        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            fn = sum((self.y_validate_l1 == 1) & (predicted == 0))
            tp = sum((self.y_validate_l1 == 1) & (predicted == 1))
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0

            try:
                return 1 / fnr
            except ZeroDivisionError:
                return 0.0

    def objective_inv_fnr_l2(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective FNR for layer 2')

        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            fn = sum((self.y_validate_l2 == 1) & (predicted == 0))
            tp = sum((self.y_validate_l2 == 1) & (predicted == 1))
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0

            try:
                return 1/fnr
            except ZeroDivisionError:
                return 0.0

    def objective_accuracy_l1(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective ACCURACY for layer 1')

        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            accuracy = accuracy_score(self.y_validate_l1, predicted)
            return accuracy

    def objective_accuracy_l2(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective ACCURACY for layer 2')

        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            accuracy = accuracy_score(self.y_validate_l2, predicted)
            return accuracy

    def objective_precision_l1(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective PRECISION for layer 1')

        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)

            return precision

    def objective_precision_l2(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective PRECISION for layer 2')

        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            precision = precision_score(self.y_validate_l2, predicted)

            return precision

    def objective_fscore_l1(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective FSCORE for layer 1')

        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)
            tp = sum((self.y_validate_l1 == 1) & (predicted == 1))
            fn = sum((self.y_validate_l1 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            fscore = 2 * precision * tpr / (precision + tpr)

            return fscore

    def objective_fscore_l2(self, trial: optuna.Trial) -> float:
        LOGGER.info(f'Calling optimization function: Objective FSCORE for layer 2')

        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            precision = precision_score(self.y_validate_l2, predicted)
            tp = sum((self.y_validate_l2 == 1) & (predicted == 1))
            fn = sum((self.y_validate_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            fscore = 2 * precision * tpr / (precision + tpr)

            return fscore

    def objective_quarantine_rate_l1(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)
            tp = sum((self.y_validate_l1 == 1) & (predicted == 1))
            fn = sum((self.y_validate_l1 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            fscore = 2 * precision * tpr / (precision + tpr)

            return fscore

    def objective_quarantine_rate_l2(self, trial: optuna.Trial) -> float:
        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            precision = precision_score(self.y_validate_l2, predicted)
            tp = sum((self.y_validate_l2 == 1) & (predicted == 1))
            fn = sum((self.y_validate_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            fscore = 2 * precision * tpr / (precision + tpr)

            return fscore

    def train_new_hps(self, classifier_name: str, parameters: dict):
        if classifier_name == 'RandomForest':
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

            LOGGER.info('Trained a new RandomForest classifier.')
            classifier.fit(self.x_train_l1, self.y_train_l1)
            return classifier

        else:
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

            LOGGER.info('Trained a new SVM classifier.')
            classifier.fit(self.x_train_l2, self.y_train_l2)
            return classifier
