import copy
import json
import logging
import sqlite3
import sys
import time
import boto3

import pandas as pd

from Loader import Loader

import KBConnectionHandler

# set an instance-level logger
LOGGER = logging.getLogger('KnowledgeBase')
LOG_FORMAT = '%(levelname) -10s %(name) -45s %(funcName) -35s %(lineno) -5d: %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class KnowledgeBase:
    def __init__(self, ampq_url):

        LOGGER.info('Creating an instance of KnowledgeBase.')
        self.ANOMALY_THRESHOLD1, self.ANOMALY_THRESHOLD2, self.BENIGN_THRESHOLD = 0.9, 0.8, 0.6
        self.cat_features = ['flag', 'protocol_type', 'service']

        self.__s3_setup_and_load()
        self.__load_data_instances()
        self.__sqlite3_setup()
        self.connection_handler = KBConnectionHandler.Connector(self, ampq_url)

    def __sqlite3_setup(self):
        LOGGER.info('Connecting to sqlite3 in memory database.')
        self.sql_connection = sqlite3.connect(':memory:')
        self.cursor = self.sql_connection.cursor()

        LOGGER.info('Instantiating the needed SQL in memory tables.')
        self.__fill_tables()

        LOGGER.info('Removing local instances.')
        self.__clean()

        LOGGER.info('Completed sqlite3 in memory databases setup.')

    def __s3_setup_and_load(self):
        self.s3_resource = boto3.client('s3')
        self.loader = Loader(s3_resource=self.s3_resource)
        self.loader.s3_load()

        LOGGER.info('Loading from S3 bucket complete.')

    def __load_data_instances(self):
        LOGGER.info('Loading original train set.')
        self.x_train = self.loader.load_og_dataset('KDDTrain+_with_labels.txt')
        LOGGER.info('Loading original reduced train set.')
        self.x_train_20p = self.loader.load_og_dataset('KDDTrain+20_percent_with_labels.txt')

        LOGGER.info('Loading train sets.')
        self.x_train_l1, self.y_train_l1 = self.loader.load_dataset(
            'KDDTrain+_l1_pca.pkl',
            'KDDTrain+_l1_targets.npy'
        )
        self.x_train_l2, self.y_train_l2 = self.loader.load_dataset(
            'KDDTrain+_l2_pca.pkl',
            'KDDTrain+_l2_targets.npy'
        )

        LOGGER.info('Loading validation sets.')
        self.x_validate_l1, self.y_validate_l1 = self.loader.load_dataset(
            'KDDValidate+_l1_pca.pkl',
            'KDDValidate+_l1_targets.npy'
        )
        self.x_validate_l2, self.y_validate_l2 = self.loader.load_dataset(
            'KDDValidate+_l2_pca.pkl',
            'KDDValidate+_l2_targets.npy'
        )

        LOGGER.info('Loading scalers.')
        self.scaler1, self.scaler2 = self.loader.load_scalers('Scaler_l1.pkl', 'Scaler_l2.pkl')

        LOGGER.info('Loading one hot encoders.')
        self.ohe1, self.ohe2 = self.loader.load_encoders('OneHotEncoder_l1.pkl', 'OneHotEncoder_l2.pkl')

        LOGGER.info('Loading pca transformers.')
        self.pca1, self.pca2 = self.loader.load_pca_transformers('layer1_pca_transformer.pkl',
                                                                 'layer2_pca_transformer.pkl')

        LOGGER.info('Loading models.')
        self.pca1, self.pca2 = self.loader.load_models('NSL_l1_classifier.pkl',
                                                       'NSL_l2_classifier.pkl')

        LOGGER.info('Loading minimal features.')
        self.features_l1 = self.loader.load_features('AWS Downloads/MinimalFeatures/NSL_features_l1.txt')
        self.features_l2 = self.loader.load_features('AWS Downloads/MinimalFeatures/NSL_features_l2.txt')

    def __fill_tables(self):
        # create a table for each train set
        self.x_train_l1.to_sql('x_train_l1', self.sql_connection, index=False, if_exists='replace')
        self.x_train_l2.to_sql('x_train_l2', self.sql_connection, index=False, if_exists='replace')

        # create a table for each train set
        self.x_train.to_sql('x_train', self.sql_connection, index=False, if_exists='replace')
        self.x_train_20p.to_sql('x_train_20p', self.sql_connection, index=False, if_exists='replace')

        # create a table for each validation set
        self.x_validate_l1.to_sql('x_validate_l1', self.sql_connection, index=False, if_exists='replace')
        self.x_validate_l2.to_sql('x_validate_l2', self.sql_connection, index=False, if_exists='replace')

        # now append target variables as the last column of each table
        self.__append_to_table('x_train_l1', 'target', self.y_train_l1)
        self.__append_to_table('x_train_l2', 'target', self.y_train_l2)
        self.__append_to_table('x_validate_l1', 'target', self.y_validate_l1)
        self.__append_to_table('x_validate_l2', 'target', self.y_validate_l2)

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
        # del self.x_test, self.y_test
        del self.x_train, self.x_train_20p

    def perform_query(self, received):
        LOGGER.info(f'Received a query.')

        message = copy.deepcopy(received)

        try:
            # Parse the content using json and dictionaries
            message_data = json.loads(message)

            select_clause = message_data.get("select")
            from_clause = message_data.get("from")
            where_clause = message_data.get("where")

            if where_clause == "":
                sql_query = f'SELECT {select_clause} FROM {from_clause}'
            else:
                sql_query = f'SELECT {select_clause} FROM {from_clause} WHERE {where_clause}'

            LOGGER.info(f'Executing the query: {sql_query}')

            result_df = pd.read_sql_query(sql_query, self.sql_connection)

        except pd.errors.DatabaseError:
            LOGGER.error('Could not fulfill the requests.')
            return None

        LOGGER.info('Query was executed correctly.')
        return result_df

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

    def __init__(self, amqp_url):
        self._reconnect_delay = 0
        self._amqp_url = amqp_url
        self._consumer = KnowledgeBase(self._amqp_url)

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

    consumer = ReconnectingConsumer(amqp_url=ampq_url)
    consumer.run()


if __name__ == '__main__':
    main()
