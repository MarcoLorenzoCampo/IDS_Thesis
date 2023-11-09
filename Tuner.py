import copy
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

from KnowledgeBase import KnowledgeBase
from DetectionSystem import DetectionSystem
import Logger


class Tuner:
    """
    This class is used to perform hyperparameter tuning on the two classifiers
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

    def tune(self, objs: list):
        """
        This function handles the hyperparameter tuning for both of the classifiers
        :param objs: objectives to reach
        :return:
        """

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

    def new_hp_models_train(self, hp1: dict, hp2: dict) -> (RandomForestClassifier, SVC):
        """
        Train models using the default hyperparameters set by researchers prior to hyperparameter tuning.
        For clarity, all the hyperparameters for random forest and svm are listed below.
        :return: Trained models for layer 1 and 2 respectively
        """

        # Start with training classifier 1
        classifier1 = (RandomForestClassifier(
            n_estimators=hp1['n_estimators'],
            criterion=hp1['criterion'],
            max_depth=hp1['max_depth'],
            min_samples_split=hp1['min_samples_split'],
            min_samples_leaf=hp1['min_samples_leaf'],
            min_weight_fraction_leaf=['min_weight_fraction_leaf'],
            max_features=hp1['max_features'],
            max_leaf_nodes=hp1['max_leaf_nodes'],
            min_impurity_decrease=hp1['min_impurity_decrease'],
            bootstrap=hp1['bootstrap'],
            oob_score=hp1['oob_score'],
            n_jobs=hp1['n_jobs'],
            random_state=hp1['random_state'],
            verbose=0,
            warm_start=hp1['warm_start'],
            class_weight=hp1['class_weight'],
            ccp_alpha=hp1['ccp_alpha'],
            max_samples=hp1['max_samples']
        ).fit(self.x_train_l1, self.y_train_l1))

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
