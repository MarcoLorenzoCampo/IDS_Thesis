import copy
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
import joblib


class ModelMaker:
    ANOMALY_THRESHOLD1 = 0.9
    ANOMALY_THRESHOLD2 = 0.8
    BENIGN_THRESHOLD = 0.6

    quarantine_samples = []
    anomaly_by_l1 = []
    anomaly_by_l2 = []
    normal_traffic = []

    def __init__(self):
        """
        This is the initialization function for the class responsible for setting up the classifiers and
        process data to make it ready for analysis.
        Data is loaded when the class is initiated, then updated when necessary, calling the function
        update_files(.)
        """

        # load the features obtained with ICFS for both layer 1 and layer 2
        with open('NSL-KDD Files/NSL_features_l1.txt', 'r') as f:
            self.features_l1 = f.read().split(',')

        with open('NSL-KDD Files/NSL_features_l2.txt', 'r') as f:
            self.features_l2 = f.read().split(',')

        # Load completely processed datasets for training
        self.x_train_l1 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/pca_train1.pkl')
        self.x_train_l2 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/pca_train2.pkl')
        self.y_train_l1 = np.load('NSL-KDD Encoded Datasets/before_pca/KDDTrain+_l1_targets.npy',
                                  allow_pickle=True)
        self.y_train_l2 = np.load('NSL-KDD Encoded Datasets/before_pca/KDDTrain+_l2_targets.npy',
                                  allow_pickle=True)

        # Load completely processed validations sets
        self.x_validate_l1 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/pca_validate1.pkl')
        self.x_validate_l2 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/pca_validate2.pkl')
        self.y_validate_l1 = np.load('NSL-KDD Encoded Datasets/before_pca/KDDValidate+_l1_targets.npy',
                                     allow_pickle=True)
        self.y_validate_l2 = np.load('NSL-KDD Encoded Datasets/before_pca/KDDValidate+_l2_targets.npy',
                                     allow_pickle=True)

        # Load completely processed test set
        self.x_test = pd.read_csv('NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
        self.y_test = np.load('NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)

        # set the categorical features
        self.cat_features = ['protocol_type', 'service', 'flag']

        # load the minmax scalers used in training
        self.scaler1 = joblib.load('NSL-KDD Files/scalers/scaler1.pkl')
        self.scaler2 = joblib.load('NSL-KDD Files/scalers/scaler2.pkl')

        # load one hot encoder for processing according to layer
        self.ohe1 = joblib.load('NSL-KDD Files/one_hot_encoders/ohe1.pkl')
        self.ohe2 = joblib.load('NSL-KDD Files/one_hot_encoders/ohe2.pkl')

        # load pca transformers to transform features according to layer
        self.pca1 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/layer1_transformer.pkl')
        self.pca2 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/layer2_transformer.pkl')

        # set up the dataframes containing the analyzed data
        self.quarantine_samples = pd.DataFrame(columns=self.x_test.columns)
        self.anomaly_by_l1 = pd.DataFrame(columns=self.x_test.columns)
        self.anomaly_by_l2 = pd.DataFrame(columns=self.x_test.columns)
        self.normal_traffic = pd.DataFrame(columns=self.x_test.columns)

    def update_files(self, to_update):
        # reload the datasets/transformers/encoders from memory if they have been changed
        if to_update == 'train':
            self.x_train_l1 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/pca_train1.pkl')
            self.x_train_l2 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/pca_train2.pkl')
            # target variables should not change ideally, but the number of samples itself may change over time
            self.y_train_l1 = np.load('NSL-KDD Encoded Datasets/before_pca/KDDTrain+_l1_targets.npy',
                                      allow_pickle=True)
            self.y_train_l2 = np.load('NSL-KDD Encoded Datasets/before_pca/KDDTrain+_l2_targets.npy',
                                      allow_pickle=True)

        # load pca transformers to transform features according to layer
        if to_update == 'pca':
            self.pca1 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/layer1_transformer.pkl')
            self.pca2 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/layer2_transformer.pkl')

        # load one hot encoder for processing according to layer
        if to_update == 'ohe':
            self.ohe1 = joblib.load('NSL-KDD Files/one_hot_encoders/ohe1.pkl')
            self.ohe2 = joblib.load('NSL-KDD Files/one_hot_encoders/ohe2.pkl')

    def train_models(self, hp1, hp2):
        """
        :param hp1: n-set of hyperparameters to train layer 1
        :param hp2: n-set of hyperparameters to train layer 2
        :return: trained models for layer 1 and 2 respectively
        """

        # Start with training classifier 1
        classifier1 = (RandomForestClassifier(n_estimators=25, criterion='gini')
                       .fit(self.x_train_l1, self.y_train_l1))

        # Now train classifier 2
        classifier2 = (SVC(C=0.1, gamma=0.01, kernel='rbf')
                       .fit(self.x_train_l2, self.y_train_l2))

        # Save models to file
        with open('Models/NSL_l1_classifier.pkl', 'wb') as model_file:
            pickle.dump(classifier1, model_file)
        with open('Models/NSL_l2_classifier.pkl', 'wb') as model_file:
            pickle.dump(classifier2, model_file)

        return classifier1, classifier2

    def pipeline_data_process(self, incoming_data, target_layer):
        """
        This function is used to process the incoming data:
        - The features are selected starting f

        :param target_layer: Indicates if the data is processed to be fed to layer 1 or layer 2
        :param incoming_data: A single or multiple data samples to process, always in the format of a DataFrame
        :return: The processed data received as an input
        """

        data = copy.deepcopy(incoming_data)

        if target_layer == 1:
            to_scale = data[self.features_l1]
            scaler = self.scaler1
            ohe = self.ohe1
            pca = self.pca1
        else:
            to_scale = data[self.features_l2]
            scaler = self.scaler2
            ohe = self.ohe2
            pca = self.pca2

        scaled = scaler.transform(to_scale)
        scaled_data = pd.DataFrame(scaled, columns=to_scale.columns)
        label_enc = ohe.transform(data[self.cat_features])
        label_enc.toarray()
        new_labels = ohe.get_feature_names_out(self.cat_features)
        new_encoded = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
        processed = pd.concat([scaled_data, new_encoded], axis=1)
        pca_transformed = pca.transform(processed)

        return pca_transformed
