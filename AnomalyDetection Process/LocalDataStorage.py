import logging
import sqlite3

import boto3
import pandas as pd

from LoaderDetectionSystem import Loader
import LoggerConfig

LOGGER = logging.getLogger('LocalDataStorage')
logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
LOGGER.info('Creating an instance of DetectionSystem.')

class Data:

    def __init__(self):
        self.__s3_setup_and_load()
        self.__load_data_instances()
        self.__sqlite3_setup()

    def __s3_setup_and_load(self):
        self.s3_resource = boto3.client('s3')
        self.loader = Loader(s3_resource=self.s3_resource)

        LOGGER.info('Loading models.')
        self.loader.s3_load()

        LOGGER.info('Loading from S3 bucket complete.')

    def __load_data_instances(self):
        LOGGER.info('Loading test set.')
        self.x_test, self.y_test = self.loader.load_testset('KDDTest+.txt', 'KDDTest+_targets.npy')

        LOGGER.info('Loading one hot encoders.')
        self.ohe1, self.ohe2 = self.loader.load_encoders('OneHotEncoder_l1.pkl', 'OneHotEncoder_l2.pkl')

        LOGGER.info('Loading scalers.')
        self.scaler1, self.scaler2 = self.loader.load_scalers('Scaler_l1.pkl', 'Scaler_l2.pkl')

        LOGGER.info('Loading pca transformers.')
        self.pca1, self.pca2 = self.loader.load_pca_transformers('layer1_pca_transformer.pkl',
                                                                 'layer2_pca_transformer.pkl')

        LOGGER.info('Loading models.')
        self.layer1, self.layer2 = self.loader.load_models('NSL_l1_classifier.pkl',
                                                           'NSL_l2_classifier.pkl')

        LOGGER.info('Loading minimal features.')
        self.features_l1 = self.loader.load_features('NSL_features_l1.txt')
        self.features_l2 = self.loader.load_features('NSL_features_l2.txt')

    def __set_local_storage(self):
        self.quarantine_samples = pd.DataFrame(columns=self.x_test.columns)
        self.anomaly_by_l1 = pd.DataFrame(columns=self.x_test.columns)
        self.anomaly_by_l2 = pd.DataFrame(columns=self.x_test.columns)
        self.normal_traffic = pd.DataFrame(columns=self.x_test.columns)

    def __sqlite3_setup(self):
        LOGGER.info('Connecting to sqlite3 in-memory database.')
        self.sql_connection = sqlite3.connect(':memory:')
        self.cursor = self.sql_connection.cursor()

        LOGGER.info('Instantiating the needed SQL in-memory tables.')
        self.__fill_tables()

        LOGGER.info('Removing local instances.')
        self.__clean()

        LOGGER.info('Completed sqlite3 in-memory databases setup.')

    def __fill_tables(self):
        self.x_test.to_sql('x_test', self.sql_connection, index=False, if_exists='replace')
        self.__append_to_table('x_test', 'target', self.y_test)

    def __append_to_table(self, table_name, column_name, target_values):
        existing_data = pd.read_sql_query(f'SELECT * FROM {table_name}', self.sql_connection)
        existing_data[column_name] = target_values
        existing_data.to_sql(table_name, self.sql_connection, if_exists='replace', index=False)

    def __clean(self):
        del self.x_test
