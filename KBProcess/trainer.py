import os
import pickle

from Shared import utils

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])

def train_models(self, model_name1: str, model_name2: str):
    # we reach this branch is there are no models to load or some specific model is required
    LOGGER.debug('Training new models.')
    classifier1, classifier2 = None, None

    # Start with training classifier 1
    if model_name1 == 'NBC':
        classifier1 = GaussianNB().fit(self.x_train_l1, self.y_train_l1)

    if model_name1 == 'SVM':
        classifier1 = (SVC(
            C=0.1,
            kernel='rbf',
            degree=3,
            gamma=0.01,
            coef0=0.0,
            shrinking=True,
            probability=True,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape='ovr'
        ).fit(self.x_train_l2, self.y_train_l2))

    # Now train classifier 2
    if model_name2 == 'NBC':
        classifier2 = GaussianNB().fit(self.x_train_l2, self.y_train_l2)

    if model_name2 == 'RandomForest':
        classifier2 = (RandomForestClassifier(
            n_estimators=25,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='sqrt',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None
        ).fit(self.x_train_l1, self.y_train_l1))

    # Default case, no classifier is specified
    if model_name1 is None:
        classifier1 = (RandomForestClassifier(
            n_estimators=25,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='sqrt',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None
        ).fit(self.x_train_l1, self.y_train_l1))

    if model_name2 is None:
        classifier2 = (SVC(
            C=0.1,
            kernel='rbf',
            degree=3,
            gamma=0.01,
            coef0=0.0,
            shrinking=True,
            probability=True,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape='ovr'
        ).fit(self.x_train_l2, self.y_train_l2))

    if classifier1 is None or classifier2 is None:
        LOGGER.critical('Error in training classifiers.')
    else:
        # Save models to file
        with open('Models/Original models/NSL_l1_classifier_og.pkl', 'wb') as model_file:
            pickle.dump(classifier1, model_file)
        with open('Models/Original models/NSL_l2_classifier_og.pkl', 'wb') as model_file:
            pickle.dump(classifier2, model_file)

        return classifier1, classifier2
