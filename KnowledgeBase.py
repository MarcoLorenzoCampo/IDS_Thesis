import copy
import functools
import os.path
import pickle
import sqlite3
import sys
import threading
import time

import numpy as np
import pandas as pd
import joblib
import pika

from pika.adapters.asyncio_connection import AsyncioConnection
from pika.exchange_type import ExchangeType
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import KBConnectionHandler
import Utils

# set an instance-level logger
LOGGER = Utils.set_logger(__name__)
LOGGER.info('Creating an instance of KnowledgeBase.')

class KnowledgeBase:
    """
    This class contains all the datasets and files needed for the classification process. It
    also contains the metrics that are constantly updated after each classification attempt.
    This is 'unprotected' data that can be accessed from the outside classes.
    """

    EXCHANGE = 'message'
    EXCHANGE_TYPE = ExchangeType.topic
    QUEUE = 'text'
    ROUTING_KEY = 'example.text'

    def __init__(self, ampq_url, model_name1: str = None, model_name2: str = None):

        # manually set the detection thresholds
        self.ANOMALY_THRESHOLD1, self.ANOMALY_THRESHOLD2, self.BENIGN_THRESHOLD = 0.9, 0.8, 0.6

        # load the features obtained with ICFS for both layer 1 and layer 2
        with open('Required Files/NSL_features_l1.txt', 'r') as f:
            self.features_l1 = f.read().split(',')

        with open('Required Files/NSL_features_l2.txt', 'r') as f:
            self.features_l2 = f.read().split(',')

        # Load completely processed datasets for training
        LOGGER.info('Loading the train sets.')
        self.x_train_l1 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/pca_train1.pkl')
        self.x_train_l2 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/pca_train2.pkl')
        self.y_train_l1 = np.load('NSL-KDD Encoded Datasets/before_pca/KDDTrain+_l1_targets.npy',
                                  allow_pickle=True)
        self.y_train_l2 = np.load('NSL-KDD Encoded Datasets/before_pca/KDDTrain+_l2_targets.npy',
                                  allow_pickle=True)

        # Load completely processed validations sets
        LOGGER.info('Loading the validation sets.')
        self.x_validate_l1 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/pca_validate1.pkl')
        self.x_validate_l2 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/pca_validate2.pkl')
        self.y_validate_l1 = np.load('NSL-KDD Encoded Datasets/before_pca/KDDValidate+_l1_targets.npy',
                                     allow_pickle=True)
        self.y_validate_l2 = np.load('NSL-KDD Encoded Datasets/before_pca/KDDValidate+_l2_targets.npy',
                                     allow_pickle=True)

        # Load completely processed test set
        LOGGER.info('Loading test sets.')
        self.x_test = pd.read_csv('NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
        self.y_test = np.load('NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)

        # set the categorical features
        self.cat_features = ['protocol_type', 'service', 'flag']

        # load the minmax scalers used in training
        LOGGER.info('Loading scalers.')
        self.scaler1 = joblib.load('Required Files/scalers/scaler1.pkl')
        self.scaler2 = joblib.load('Required Files/scalers/scaler2.pkl')

        # load one hot encoder for processing according to layer
        LOGGER.info('Loading one hot encoders.')
        self.ohe1 = joblib.load('Required Files/one_hot_encoders/ohe1.pkl')
        self.ohe2 = joblib.load('Required Files/one_hot_encoders/ohe2.pkl')

        # load pca transformers to transform features according to layer
        LOGGER.info('Loading test pca encoders.')
        self.pca1 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/layer1_transformer.pkl')
        self.pca2 = joblib.load('NSL-KDD Encoded Datasets/pca_transformed/layer2_transformer.pkl')

        # load or train the classifiers
        if (os.path.exists('Models/Original models/NSL_l1_classifier_og.pkl') and
                os.path.exists('Models/Original models/NSL_l2_classifier_og.pkl')):

            LOGGER.info('Loading existing models.')
            with open('Models/Original models/NSL_l1_classifier_og.pkl', 'rb') as file:
                self.layer1 = pickle.load(file)
            with open('Models/Original models/NSL_l2_classifier_og.pkl', 'rb') as file:
                self.layer2 = pickle.load(file)
        else:
            LOGGER.error('First program execution has no models, training them..')
            self.layer1, self.layer2 = self.__default_training(model_name1=model_name1, model_name2=model_name2)

        # Initialize an in-memory SQLite3 database
        self.sql_connection = sqlite3.connect(':memory:')
        self.cursor = self.sql_connection.cursor()

        # Create tables in the in-memory database for your datasets
        self.__fill_tables()
        # Remove all the variables containing the dataframes
        self.__clean()

        # Component that handles connections
        self.connection_handler = KBConnectionHandler.Connector(ampq_url)

    def __fill_tables(self):
        # create a table for each train set
        self.x_train_l1.to_sql('x_train_l1', self.sql_connection, index=False, if_exists='replace')
        self.x_train_l2.to_sql('x_train_l2', self.sql_connection, index=False, if_exists='replace')

        # create a table for each validation set
        self.x_validate_l1.to_sql('x_validate_l1', self.sql_connection, index=False, if_exists='replace')
        self.x_validate_l2.to_sql('x_validate_l2', self.sql_connection, index=False, if_exists='replace')

        # also for the test set since we may need it
        self.x_test.to_sql('x_test', self.sql_connection, index=False, if_exists='replace')

        # now append target variables as the last column of each table
        self.__append_to_table('x_train_l1', 'target', self.y_train_l1)
        self.__append_to_table('x_train_l2', 'target', self.y_train_l2)
        self.__append_to_table('x_validate_l1', 'target', self.y_validate_l1)
        self.__append_to_table('x_validate_l2', 'target', self.y_validate_l2)
        self.__append_to_table('x_test', 'target', self.y_test)

    def __append_to_table(self, table_name, column_name, target_values):
        # Fetch the existing table from the in-memory database
        existing_data = pd.read_sql_query(f'SELECT * FROM {table_name}', self.sql_connection)
        # Append the target column to the existing table
        existing_data[column_name] = target_values
        # Update the table in the in-memory database
        existing_data.to_sql(table_name, self.sql_connection, if_exists='replace', index=False)

    def __clean(self):
        # Remove instances of datasets to free up memory
        del self.x_train_l1, self.x_train_l2, self.y_train_l1, self.y_train_l2
        del self.x_validate_l1, self.x_validate_l2, self.y_validate_l1, self.y_validate_l2
        del self.x_test, self.y_test

    def __perform_query(self, sql_query):
        result_df = pd.read_sql_query(sql_query, self.sql_connection)
        return result_df

    def __update_files(self, to_update):
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
            self.ohe1 = joblib.load('Required Files/one_hot_encoders/ohe1.pkl')
            self.ohe2 = joblib.load('Required Files/one_hot_encoders/ohe2.pkl')

        # Create tables in the in-memory database for your datasets
        self.__fill_tables()
        # Remove all the variables containing the dataframes
        self.__clean()

    def __default_training(self, model_name1: str = None, model_name2: str = None):
        """
        Train models using the default hyperparameters set by researchers prior to hyperparameter tuning.
        For clarity, all the hyperparameters for random forest and svm are listed below.
        :return: Trained models for layer 1 and 2 respectively
        """

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

    def __show_info(self):
        LOGGER.info('Shapes and sized of the sets:')
        LOGGER.info(f'TRAIN:\n'
                         f'x_train_l1 = {self.x_train_l1.shape}\n'
                         f'x_train_l2 = {self.x_train_l2.shape}\n'
                         f'y_train_l1 = {len(self.y_train_l1)}\n'
                         f'y_train_l2 = {len(self.y_train_l2)}')
        LOGGER.info(f'VALIDATE:\n'
                         f'x_validate_l1 = {self.x_validate_l1.shape}\n'
                         f'x_validate_l2 = {self.x_validate_l2.shape}\n'
                         f'y_validate_l1 = {len(self.y_validate_l1)}\n'
                         f'y_validate_l2 = {len(self.y_validate_l2)}')

class ReconnectingConsumer:
    """
    Declares an instance of a knowledge base, and handles the setup of its component
    connection_handler.
    """
    def __init__(self, amqp_url, model_name1: str = None, model_name2: str = None):
        self._reconnect_delay = 0
        self._amqp_url = amqp_url
        self._consumer = KnowledgeBase(self._amqp_url, model_name1=model_name1, model_name2=model_name2)

    def run(self):
        while True:
            try:
                self._consumer.connection_handler.run()
            except KeyboardInterrupt:
                self._consumer.connection_handler.stop()
                break
            self._maybe_reconnect()

    def _maybe_reconnect(self):
        if self._consumer.connection_handler.should_reconnect:
            self._consumer.connection_handler.stop()
            reconnect_delay = self._get_reconnect_delay()
            LOGGER.info('Reconnecting after %d seconds', reconnect_delay)
            time.sleep(reconnect_delay)
            self._consumer = KnowledgeBase(self._amqp_url)

    def _get_reconnect_delay(self):
        if self._consumer.connection_handler.was_consuming:
            self._reconnect_delay = 0
        else:
            self._reconnect_delay += 1
        if self._reconnect_delay > 30:
            self._reconnect_delay = 30
        return self._reconnect_delay


def main():
    ampq_url = sys.argv[1] if len(sys.argv) > 1 else 'localhost'
    model1 = sys.argv[2] if len(sys.argv) > 2 else None
    model2 = sys.argv[3] if len(sys.argv) > 3 else None

    consumer = ReconnectingConsumer(amqp_url=ampq_url, model_name1=model1, model_name2=model2)
    consumer.run()


if __name__ == '__main__':
    main()