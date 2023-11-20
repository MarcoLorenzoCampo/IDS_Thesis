import copy
import pickle
import optuna

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from sklearn.svm import SVC

from KnowledgeBase import KnowledgeBase
from DetectionSystem import DetectionSystem
import Utils


class Tuner:
    """
    This class is used to perform hyperparameter tuning on the two classifiers:
    Define objective functions for both single-objectives and multiple-objectives hyperparameter tuning.
    - Minimize false positives
    """

    def __init__(self, kb: KnowledgeBase, ids: DetectionSystem):

        # instance level logger
        self.logger = Utils.set_logger(__name__)

        # set a reference to the ids to tune
        self.ids = ids

        # do not modify the values, passed by reference
        # validation sets
        self.x_validate_l1 = kb.x_validate_l1
        self.x_validate_l2 = kb.x_validate_l2
        self.y_validate_l1 = kb.y_validate_l1
        self.y_validate_l2 = kb.y_validate_l2

        # train sets
        self.x_train_l1 = kb.x_train_l1
        self.x_train_l2 = kb.x_train_l2
        self.y_train_l1 = kb.y_train_l1
        self.y_train_l2 = kb.y_train_l2

        # classifiers
        # copied to avoid conflicts
        self.layer1 = copy.deepcopy(ids.layer1)
        self.layer2 = copy.deepcopy(ids.layer2)

        # new models to be stored as temporary variables in the class
        self.new_opt_layer1 = []
        self.new_opt_layer2 = []

        # number of trials for tuning
        self.n_trials = 7

        # accuracy for over fitting evaluations
        self.val_accuracy_l1 = []
        self.val_accuracy_l2 = []

        # accuracy of the best hyperparameters
        self.best_acc1 = 0
        self.best_acc2 = 0

    def tune(self, objs: list):
        study = optuna.create_study(study_name='RandomForest optimization', direction='minimize')
        study.optimize(self.__objective_fp_l1, n_trials=self.n_trials)

        # set the layers as the main for the ids
        self.ids.layer1 = self.new_opt_layer1

        self.best_acc1 = self.val_accuracy_l1[study.best_trial.number]

        study = optuna.create_study(study_name='SVM optimization', direction='minimize')
        study.optimize(self.__objective_fp_l2, n_trials=self.n_trials)

        # set the layer as the main for the ids
        self.ids.layer2 = self.new_opt_layer2

        self.best_acc2 = self.val_accuracy_l2[study.best_trial.number]

        # reset the storage variables
        self.reset()

        # return the newly trained models and hyperparameters
        return self.new_opt_layer1, self.new_opt_layer2

    def __objective_tpr_l1(self, trial: optuna.Trial) -> float:
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

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            return tpr

    def __objective_tpr_l2(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            return tpr

    def __objective_fpr_l1(self, trial: optuna.Trial) -> float:
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

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            fp = sum((self.y_train_l2 == 0) & (predicted == 1))
            tn = sum((self.y_train_l2 == 0) & (predicted == 0))
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

            return fpr

    def __objective_fpr_l2(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            fp = sum((self.y_train_l2 == 0) & (predicted == 1))
            tn = sum((self.y_train_l2 == 0) & (predicted == 0))
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

            return fpr

    def __objective_tnr_l1(self, trial: optuna.Trial) -> float:
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

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            tn = sum((self.y_train_l2 == 0) & (predicted == 0))
            fp = sum((self.y_train_l2 == 0) & (predicted == 1))
            tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            return tnr

    def __objective_tnr_l2(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            tn = sum((self.y_train_l2 == 0) & (predicted == 0))
            fp = sum((self.y_train_l2 == 0) & (predicted == 1))
            tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            return tnr

    def __objective_fnr_l1(self, trial: optuna.Trial) -> float:
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

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0

            return fnr

    def __objective_fnr_l2(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0

            return fnr

    def __objective_fp_l1(self, trial: optuna.Trial) -> int:
        """
        This function defines an objective function to be minimized.
        :param trial:
        :return: Number of false positives
        """
        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            # list now the hyperparameters that need tuning
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            # append validation accuracy
            self.val_accuracy_l1.append(accuracy_score(self.y_validate_l1, predicted))

            # store the new classifier
            self.new_opt_layer1 = classifier

            # confusion_matrix[1] is the false positives
            return confusion_matrix(self.y_validate_l1, predicted)[0][1]

    def __objective_fp_l2(self, trial: optuna.Trial) -> int:
        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            # store the new classifier
            self.new_opt_layer2 = classifier

            # append validation accuracy
            self.val_accuracy_l2.append(accuracy_score(self.y_validate_l2, predicted))

            # confusion_matrix[1] is the false positives
            return confusion_matrix(self.y_validate_l2, predicted)[0][1]

    def __objective_accuracy_l1(self, trial: optuna.Trial) -> float:
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

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            accuracy = accuracy_score(self.y_validate_l1, predicted)
            return accuracy

    def __objective_accuracy_l2(self, trial: optuna.Trial) -> float:
        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            accuracy = accuracy_score(self.y_validate_l1, predicted)
            return accuracy

    def __objective_precision_l1(self, trial: optuna.Trial) -> float:
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

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)

            return precision

    def __objective_precision_l2(self, trial: optuna.Trial) -> float:
        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)

            return precision

    def __objective_fscore_l1(self, trial: optuna.Trial) -> float:
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

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)
            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            fscore = 2*precision*tpr/(precision + tpr)

            return fscore

    def __objective_fscore_l2(self, trial: optuna.Trial) -> float:
        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)
            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            fscore = 2 * precision * tpr / (precision + tpr)

            return fscore

    def __train_new_hps(self, classifier_name: str, parameters: dict):

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

        classifier.fit(self.x_train_l2, self.y_train_l2)
        return classifier

    def reset(self):
        self.val_accuracy_l1 = []
        self.val_accuracy_l2 = []