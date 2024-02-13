import pickle

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class DefaultTrainer:

    def __init__(self):
        self.x_train_l1 = joblib.load(f'AWS Downloads/Datasets/PCAEncoded/KDDTrain+_l1_pca.pkl')
        self.x_train_l2 = joblib.load(f'AWS Downloads/Datasets/PCAEncoded/KDDTrain+_l2_pca.pkl')
        self.y_train_l1 = np.load("AWS Downloads/Datasets/PCAEncoded/KDDTrain+_l1_targets.npy")
        self.y_train_l2 = np.load("AWS Downloads/Datasets/PCAEncoded/KDDTrain+_l2_targets.npy")

    def train_rf(self):
        classifier = RandomForestClassifier(
            n_estimators=50,
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
        )

        classifier.fit(self.x_train_l1, self.y_train_l1)

        with open('StartingModels/random_forest_model_default.pkl', 'wb') as f:
            pickle.dump(classifier, f)

    def train_svm(self):
        classifier = SVC(
            C=10,
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
        )

        classifier.fit(self.x_train_l2, self.y_train_l2)

        with open('StartingModels/support_vector_machine_model_default.pkl', 'wb') as f:
            pickle.dump(classifier, f)


def main():
    trainer = DefaultTrainer()
    trainer.train_svm()
    trainer.train_rf()


if __name__ == '__main__':
    main()
