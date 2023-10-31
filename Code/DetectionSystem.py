import copy
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
import joblib


class DetectionSystem:
    # thresholds
    ANOMALY_THRESHOLD1 = 0.9
    ANOMALY_THRESHOLD2 = 0.8
    BENIGN_THRESHOLD = 0.6

    # classed of traffic
    quarantine_samples = []
    anomaly_by_l1 = []
    anomaly_by_l2 = []
    normal_traffic = []

    # performance metrics


    def __init__(self):
        """
        This is the initialization function for the class responsible for setting up the classifiers and
        process data to make it ready for analysis.
        Data is loaded when the class is initiated, then updated when necessary, calling the function
        update_files(.)
        """

        # load the features obtained with ICFS for both layer 1 and layer 2
        with open('../NSL-KDD Files/NSL_features_l1.txt', 'r') as f:
            self.features_l1 = f.read().split(',')

        with open('../NSL-KDD Files/NSL_features_l2.txt', 'r') as f:
            self.features_l2 = f.read().split(',')

        # Load completely processed datasets for training
        self.x_train_l1 = joblib.load('../NSL-KDD Encoded Datasets/pca_transformed/pca_train1.pkl')
        self.x_train_l2 = joblib.load('../NSL-KDD Encoded Datasets/pca_transformed/pca_train2.pkl')
        self.y_train_l1 = np.load('../NSL-KDD Encoded Datasets/before_pca/KDDTrain+_l1_targets.npy',
                                  allow_pickle=True)
        self.y_train_l2 = np.load('../NSL-KDD Encoded Datasets/before_pca/KDDTrain+_l2_targets.npy',
                                  allow_pickle=True)

        # Load completely processed validations sets
        self.x_validate_l1 = joblib.load('../NSL-KDD Encoded Datasets/pca_transformed/pca_validate1.pkl')
        self.x_validate_l2 = joblib.load('../NSL-KDD Encoded Datasets/pca_transformed/pca_validate2.pkl')
        self.y_validate_l1 = np.load('../NSL-KDD Encoded Datasets/before_pca/KDDValidate+_l1_targets.npy',
                                     allow_pickle=True)
        self.y_validate_l2 = np.load('../NSL-KDD Encoded Datasets/before_pca/KDDValidate+_l2_targets.npy',
                                     allow_pickle=True)

        # Load completely processed test set
        self.x_test = pd.read_csv('../NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
        self.y_test = np.load('../NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)

        # set the categorical features
        self.cat_features = ['protocol_type', 'service', 'flag']

        # load the minmax scalers used in training
        self.scaler1 = joblib.load('../NSL-KDD Files/scalers/scaler1.pkl')
        self.scaler2 = joblib.load('../NSL-KDD Files/scalers/scaler2.pkl')

        # load one hot encoder for processing according to layer
        self.ohe1 = joblib.load('../NSL-KDD Files/one_hot_encoders/ohe1.pkl')
        self.ohe2 = joblib.load('../NSL-KDD Files/one_hot_encoders/ohe2.pkl')

        # load pca transformers to transform features according to layer
        self.pca1 = joblib.load('../NSL-KDD Encoded Datasets/pca_transformed/layer1_transformer.pkl')
        self.pca2 = joblib.load('../NSL-KDD Encoded Datasets/pca_transformed/layer2_transformer.pkl')

        # set up the dataframes containing the analyzed data
        self.quarantine_samples = pd.DataFrame(columns=self.x_test.columns)
        self.anomaly_by_l1 = pd.DataFrame(columns=self.x_test.columns)
        self.anomaly_by_l2 = pd.DataFrame(columns=self.x_test.columns)
        self.normal_traffic = pd.DataFrame(columns=self.x_test.columns)

    def update_files(self, to_update):
        # reload the datasets/transformers/encoders from memory if they have been changed
        if to_update == 'train':
            self.x_train_l1 = joblib.load('../NSL-KDD Encoded Datasets/pca_transformed/pca_train1.pkl')
            self.x_train_l2 = joblib.load('../NSL-KDD Encoded Datasets/pca_transformed/pca_train2.pkl')
            # target variables should not change ideally, but the number of samples itself may change over time
            self.y_train_l1 = np.load('../NSL-KDD Encoded Datasets/before_pca/KDDTrain+_l1_targets.npy',
                                      allow_pickle=True)
            self.y_train_l2 = np.load('../NSL-KDD Encoded Datasets/before_pca/KDDTrain+_l2_targets.npy',
                                      allow_pickle=True)

        # load pca transformers to transform features according to layer
        if to_update == 'pca':
            self.pca1 = joblib.load('../NSL-KDD Encoded Datasets/pca_transformed/layer1_transformer.pkl')
            self.pca2 = joblib.load('../NSL-KDD Encoded Datasets/pca_transformed/layer2_transformer.pkl')

        # load one hot encoder for processing according to layer
        if to_update == 'ohe':
            self.ohe1 = joblib.load('../NSL-KDD Files/one_hot_encoders/ohe1.pkl')
            self.ohe2 = joblib.load('../NSL-KDD Files/one_hot_encoders/ohe2.pkl')

    def train_models(self):
        """
        :return: trained models for layer 1 and 2 respectively
        """

        # Start with training classifier 1
        classifier1 = (RandomForestClassifier(n_estimators=25, criterion='gini')
                       .fit(self.x_train_l1, self.y_train_l1))

        # Now train classifier 2
        classifier2 = (SVC(C=0.1, gamma=0.01, kernel='rbf')
                       .fit(self.x_train_l2, self.y_train_l2))

        # Save models to file
        with open('../Models/NSL_l1_classifier.pkl', 'wb') as model_file:
            pickle.dump(classifier1, model_file)
        with open('../Models/NSL_l2_classifier.pkl', 'wb') as model_file:
            pickle.dump(classifier2, model_file)

        return classifier1, classifier2

    def train_accuracy(self, layer1, layer2):
        """
        Function to see how the IDS performs on training data, useful to see if overfitting happens
        :param layer1: classifier 1
        :param layer2: classifier 2
        """

        l1_prediction = layer1.predict(self.x_train_l1, self.y_train_l1)
        l2_prediction = layer2.predict(self.x_train_l2, self.y_train_l2)

        # Calculate the accuracy score for layer 1.
        l1_accuracy = accuracy_score(self.y_train_l1, l1_prediction)

        # Calculate the accuracy score for layer 2.
        l2_accuracy = accuracy_score(self.y_train_l2, l2_prediction)

        # Print the accuracy scores.
        print("Layer 1 accuracy:", l1_accuracy)
        print("Layer 2 accuracy:", l2_accuracy)
