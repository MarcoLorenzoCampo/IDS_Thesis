import copy
import pickle
import optuna

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from KnowledgeBase import KnowledgeBase
from DetectionSystem import DetectionSystem
import Logger


class Tuner:
    """
    This class is used to perform hyperparameter tuning on the two classifiers:
    Define objective functions for both single-objectives and multiple-objectives hyperparameter tuning.
    - Minimize false positives
    """

    def __init__(self, kb: KnowledgeBase, ids: DetectionSystem):

        # instance level logger
        self.logger = Logger.set_logger(__name__)

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
        self.y_train_l2 = kb.y_train_l1

        # classifiers
        # copied to avoid conflicts
        self.layer1 = copy.deepcopy(ids.layer1)
        self.layer2 = copy.deepcopy(ids.layer2)

        # version of the model
        self.model_version = 0

    def tune(self):
        study = optuna.create_study()   # create a new study
        study.optimize(self.objective_fp, n_trials=100, layer=1)

    def objective_fp(self, trial: optuna.Trial, layer: int):
        """
        This function defines an objective function to be minimized.
        :param layer: Target layer
        :param trial:
        :return:
        """
        if layer == 1:
            # providing a choice of classifiers to use in the 'choices' array
            regressor_name = trial.suggest_categorical('classifier', ['RandomForest'])
            if regressor_name == 'RandomForest':
                # list now the hyperparameters that need tuning
                rf_max_depth = trial.suggest_int('max_depth', 2, 32)
                rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])

                # add to parameters all the hyperparameters that need tuning
                parameters = {
                    'max_depth': rf_max_depth,
                    'criterion': rf_criterion
                }

                # train the model with the updated list of hyperparameters
                classifier = RandomForestClassifier(
                    n_estimators=parameters.get('n_estimators', 100),
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

                # fit the new classifier with the train set
                classifier.fit(self.x_train_l1, self.y_train_l1)

                predicted = classifier.predict(self.x_validate_l1)

                # confusion_matrix[1] is the false positives
                return confusion_matrix(self.y_validate_l1, predicted)[1]



    def hp_iterator(self) -> RandomForestClassifier:
        from sklearn.model_selection import RandomizedSearchCV

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=10, stop=200, num=5)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in each tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_epth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf = RandomForestClassifier(random_state=42)

        # Random search of parameters, using 3-fold cross-validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                       n_iter=100, scoring='neg_mean_absolute_error',
                                       cv=3, verbose=2, random_state=42, n_jobs=-1,
                                       return_train_score=True)

        self.logger.debug('New hyperparameters selected:', rf_random.cv_results_)

        return rf_random.best_estimator_

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

        with open(f'Models/Tuned/NSL_l1_classifier_{self.model_version}.pkl', 'wb') as model_file:
            pickle.dump(classifier1, model_file)
        with open(f'Models/Tuned/NSL_l2_classifier_{self.model_version}.pkl', 'wb') as model_file:
            pickle.dump(classifier2, model_file)
        self.model_version += 1

        return classifier1, classifier2
