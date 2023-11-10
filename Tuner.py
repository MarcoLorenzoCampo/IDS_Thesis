import copy
import pickle
import optuna

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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

        # version of the model
        self.model_version = 0

        # number of trials for tuning
        self.n_trials = 5

    def tune(self):
        study = optuna.create_study(direction='minimize')  # create a new study
        study.optimize(self.objective_fp_l1, n_trials=self.n_trials)

        # set the layers as the main for the ids
        self.ids.layer1 = self.new_opt_layer1

        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective_fp_l2, n_trials=self.n_trials)

        # set the layer as the main for the ids
        self.ids.layer2 = self.new_opt_layer2

        # return the newly trained models and hyperparameters
        return self.new_opt_layer1, self.new_opt_layer2

    def objective_fp_l1(self, trial: optuna.Trial):
        """
        This function defines an objective function to be minimized.
        :param trial:
        :return:
        """
        # providing a choice of classifiers to use in the 'choices' array
        regressor_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if regressor_name == 'RandomForest':
            # list now the hyperparameters that need tuning
            rf_n_estimators = 25
            # rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
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

            # train the model with the updated list of hyperparameters (if present)
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

            # fit the new classifier with the train set and predict on the validation set
            classifier.fit(self.x_train_l1, self.y_train_l1)
            predicted = classifier.predict(self.x_validate_l1)

            # store the new classifier
            self.new_opt_layer1 = classifier

            # confusion_matrix[1] is the false positives
            return confusion_matrix(self.y_validate_l1, predicted)[0][1]

    def objective_fp_l2(self, trial: optuna.Trial):
        # providing a choice of classifiers to use in the 'choices' array
        regressor_name = trial.suggest_categorical('classifier', ['SVC'])
        if regressor_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

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
            predicted = classifier.predict(self.x_validate_l2)

            # store the new classifier
            self.new_opt_layer2 = classifier

            # confusion_matrix[1] is the false positives
            return confusion_matrix(self.y_validate_l2, predicted)[0][1]

    def new_hp_models_train(self, parameters: dict, hp2: dict) -> (RandomForestClassifier, SVC):
        """
        Train models using the default hyperparameters set by researchers prior to hyperparameter tuning.
        For clarity, all the hyperparameters for random forest and svm are listed below.
        :return: Trained models for layer 1 and 2 respectively
        """

        # Now train classifier 2
        classifier2 = (SVC(
            C=hp2['C'],
            kernel=hp2['kernel'],
            degree=hp2['degree'],
            gamma=hp2['gamma'],
            coef0=hp2['coef0'],
            shrinking=hp2['shrinking'],
            probability=True,
            tol=hp2['tol'],
            cache_size=hp2['cache_size'],
            class_weight=hp2['class_weight'],
            verbose=False,
            max_iter=hp2['max_iter'],
            decision_function_shape=hp2['decision_function_shape']
        ).fit(self.x_train_l2, self.y_train_l2))

        # Save models to file

        with open(f'Models/Tuned/NSL_l2_classifier_{self.model_version}.pkl', 'wb') as model_file:
            pickle.dump(classifier2, model_file)
        self.model_version += 1

        return classifier2
