import logging
import os.path
import pickle
import sqlite3
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import KBConnectionHandler

# set an instance-level logger
LOGGER = logging.getLogger(__name__)
LOG_FORMAT = '%(levelname) -10s %(name) -45s %(funcName) -35s %(lineno) -5d: %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class KnowledgeBase:
    """
    This class contains all the datasets and files needed for the classification process. It
    also contains the metrics that are constantly updated after each classification attempt.
    This is 'unprotected' data that can be accessed from the outside classes.
    """

    def __init__(self, ampq_url, model_name1: str = None, model_name2: str = None):

        LOGGER.info('Creating an instance of KnowledgeBase.')
        self.ANOMALY_THRESHOLD1, self.ANOMALY_THRESHOLD2, self.BENIGN_THRESHOLD = 0.9, 0.8, 0.6

        LOGGER.info('Loading minimal features.')
        self.features_l1 = self.__load_features('Required Files/NSL_features_l1.txt')
        self.features_l2 = self.__load_features('Required Files/NSL_features_l2.txt')

        LOGGER.info('Loading train sets.')
        self.x_train_l1, self.y_train_l1 = self.__load_dataset('pca_train1.pkl', 'KDDTrain+_l1_targets.npy')
        self.x_train_l2, self.y_train_l2 = self.__load_dataset('pca_train2.pkl', 'KDDTrain+_l2_targets.npy')

        LOGGER.info('Loading validation sets.')
        self.x_validate_l1, self.y_validate_l1 = self.__load_dataset('pca_validate1.pkl', 'KDDValidate+_l1_targets.npy')
        self.x_validate_l2, self.y_validate_l2 = self.__load_dataset('pca_validate2.pkl', 'KDDValidate+_l2_targets.npy')

        LOGGER.info('Loading test sets.')
        self.x_test, self.y_test = self.__load_test_set()

        self.cat_features = ['protocol_type', 'service', 'flag']

        LOGGER.info('Loading scalers.')
        self.scaler1, self.scaler2 = self.__load_scalers('scaler1.pkl', 'scaler2.pkl')

        LOGGER.info('Loading one hot encoders.')
        self.ohe1, self.ohe2 = self.__load_encoders('ohe1.pkl', 'ohe2.pkl')

        LOGGER.info('Loading pca transformers.')
        self.pca1, self.pca2 = self.__load_pca_transformers('layer1_transformer.pkl', 'layer2_transformer.pkl')

        LOGGER.info('Loading models.')
        self.layer1, self.layer2 = self.__load_or_train(model_name1=model_name1, model_name2=model_name2)

        LOGGER.info('Connecting to sqlite3 in memory database.')
        self.sql_connection = sqlite3.connect(':memory:')
        self.cursor = self.sql_connection.cursor()

        self.__fill_tables()
        self.__clean()
        LOGGER.info('Completed sqlite3 in memory databases setup.')

        self.connection_handler = KBConnectionHandler.Connector(self, ampq_url)

    def __load_features(self, file_path):
        with open(file_path, 'r') as f:
            return f.read().split(',')

    def __load_dataset(self, pca_file, targets_file):
        x = joblib.load(f'NSL-KDD Encoded Datasets/pca_transformed/{pca_file}')
        x_df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
        y = np.load(f'NSL-KDD Encoded Datasets/before_pca/{targets_file}', allow_pickle=True)
        return x_df, y

    def __load_test_set(self):
        x_test = pd.read_csv('NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
        y_test = np.load('NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)
        return x_test, y_test

    def __load_scalers(self, scaler1_file, scaler2_file):
        scaler1 = joblib.load(f'Required Files/scalers/{scaler1_file}')
        scaler2 = joblib.load(f'Required Files/scalers/{scaler2_file}')
        return scaler1, scaler2

    def __load_encoders(self, ohe1_file, ohe2_file):
        ohe1 = joblib.load(f'Required Files/one_hot_encoders/{ohe1_file}')
        ohe2 = joblib.load(f'Required Files/one_hot_encoders/{ohe2_file}')
        return ohe1, ohe2

    def __load_pca_transformers(self, pca1_file, pca2_file):
        pca1 = joblib.load(f'NSL-KDD Encoded Datasets/pca_transformed/{pca1_file}')
        pca2 = joblib.load(f'NSL-KDD Encoded Datasets/pca_transformed/{pca2_file}')
        return pca1, pca2

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

    def perform_query(self, sql_query):
        LOGGER.info(f'Performing query: {sql_query}')

        try:
            result_df = pd.read_sql_query(sql_query, self.sql_connection)
        except pd.errors.DatabaseError:
            LOGGER.error('Could not fulfill the requests.')
            result_df = None

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

    def __load_or_train(self, model_name1: str = None, model_name2: str = None):
        """
        Train models using the default hyperparameters set by researchers prior to hyperparameter tuning.
        For clarity, all the hyperparameters for random forest and svm are listed below.
        :return: Trained models for layer 1 and 2 respectively
        """
        can_load = (os.path.exists('Models/Original models/NSL_l1_classifier_og.pkl') and
                    os.path.exists('Models/Original models/NSL_l2_classifier_og.pkl'))

        new_requests = model_name1 is not None or model_name2 is not None

        # if the models already exist and no specific model is required, load the sets
        if can_load or not new_requests:

            LOGGER.info('Loading existing models.')
            with open('Models/Original models/NSL_l1_classifier_og.pkl', 'rb') as file:
                layer1 = pickle.load(file)
            with open('Models/Original models/NSL_l2_classifier_og.pkl', 'rb') as file:
                layer2 = pickle.load(file)

            return layer1, layer2

        # we reach this branch is there are no models to load or some specific model is required
        LOGGER.info('Training new models.')
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
    model1 = sys.argv[1] if len(sys.argv) > 1 else None
    model2 = sys.argv[2] if len(sys.argv) > 2 else None
    ampq_url = sys.argv[3] if len(sys.argv) > 3 else "amqp://guest:guest@host:5672/"

    consumer = ReconnectingConsumer(amqp_url=ampq_url, model_name1=model1, model_name2=model2)
    consumer.run()


if __name__ == '__main__':
    main()