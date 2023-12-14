import json
import os
import sqlite3
from typing import Tuple

import boto3
import pandas as pd

from Shared import s3_wrapper, utils

LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class Storage:

    def __init__(self):
        self.bucket_name = 'nsl-kdd-datasets'
        self.__s3_setup()
        self.__load_data_instances()
        self.__set_local_storage()
        self.__sqlite3_setup()
        self.__parse_detection_parameters()

    def __parse_detection_parameters(self):
        json_data = json.load(open('config.json', 'r'))

        self.ANOMALY_THRESHOLD1 = json_data.get("ANOMALY_THRESHOLD1", None)
        self.ANOMALY_THRESHOLD2 = json_data.get("ANOMALY_THRESHOLD2", None)
        self.BENIGN_THRESHOLD = json_data.get("BENIGN_THRESHOLD", None)
        self.cat_features = json_data.get("cat_features", None)

    def __s3_setup(self):

        self.s3_resource = boto3.client('s3')
        self.loader = s3_wrapper.Loader(bucket_name=self.bucket_name, s3_resource=self.s3_resource)

        if self.__s3_ok() and not utils.need_s3_update():
            LOGGER.debug('S3 is already setup and loaded.')
            return

        self.__s3_load()

    def __s3_load(self):
        LOGGER.debug(f'Loading data from S3 bucket {self.bucket_name}.')

        self.loader.s3_min_features()
        self.loader.s3_one_hot_encoders()
        self.loader.s3_pca_encoders()
        self.loader.s3_scalers()
        self.loader.s3_models()
        self.loader.s3_original_test_set()

        LOGGER.debug('Loading from S3 bucket complete.')

    def __s3_ok(self):
        l = self.loader
        return (
            l.check_test_original() and
            l.check_features() and
            l.check_models() and
            l.check_encoders() and
            l.check_pca_encoders() and
            l.check_scalers()
        )

    def __load_data_instances(self):
        LOGGER.debug('Loading test set.')
        self.x_test, self.y_test = s3_wrapper.Loader.load_test_set()

        LOGGER.debug('Loading one hot encoders.')
        self.ohe1, self.ohe2 = s3_wrapper.Loader.load_encoders('OneHotEncoder_l1.pkl', 'OneHotEncoder_l2.pkl')

        LOGGER.debug('Loading scalers.')
        self.scaler1, self.scaler2 = s3_wrapper.Loader.load_scalers('Scaler_l1.pkl', 'Scaler_l2.pkl')

        LOGGER.debug('Loading pca transformers.')
        self.pca1, self.pca2 = s3_wrapper.Loader.load_pca_transformers('layer1_pca_transformer.pkl',
                                                                         'layer2_pca_transformer.pkl')

        LOGGER.debug('Loading models.')
        self.layer1, self.layer2 = s3_wrapper.Loader.load_models('NSL_l1_classifier.pkl',
                                                                   'NSL_l2_classifier.pkl')

        LOGGER.debug('Loading minimal features.')
        self.features_l1 = s3_wrapper.Loader.load_features('NSL_features_l1.txt')
        self.features_l2 = s3_wrapper.Loader.load_features('NSL_features_l2.txt')

    def __set_local_storage(self):
        self.quarantine_samples = pd.DataFrame(columns=self.x_test.columns)
        self.anomaly_by_l1 = pd.DataFrame(columns=self.x_test.columns)
        self.anomaly_by_l2 = pd.DataFrame(columns=self.x_test.columns)
        self.normal_traffic = pd.DataFrame(columns=self.x_test.columns)

    def __sqlite3_setup(self):
        LOGGER.debug('Connecting to sqlite3 in-memory database.')
        self.sql_connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.cursor = self.sql_connection.cursor()

        LOGGER.debug('Instantiating the needed SQL in-memory tables.')
        self.__fill_tables()

        LOGGER.debug('Removing local instances.')
        self.__clean()

        LOGGER.debug('Completed sqlite3 in-memory databases setup.')

    def __fill_tables(self):
        self.x_test.to_sql('x_test', self.sql_connection, index=False, if_exists='replace')
        self.__append_to_table('x_test', 'targets', self.y_test)

    def __append_to_table(self, table_name, column_name, target_values):
        existing_data = pd.read_sql_query(f'SELECT * FROM {table_name}', self.sql_connection)
        existing_data[column_name] = target_values
        existing_data.to_sql(table_name, self.sql_connection, if_exists='replace', index=False)

    def __clean(self):
        del self.x_test

    def add_to_quarantine(self, sample: pd.DataFrame) -> Tuple[str, int]:
        self.quarantine_samples = pd.concat([self.quarantine_samples, sample], axis=0)
        return 'quarantine', 1

    def add_to_anomaly1(self, sample: pd.DataFrame) -> Tuple[str, int]:
        self.anomaly_by_l1 = pd.concat([self.anomaly_by_l1, sample], axis=0)
        return 'l1_anomaly', 1

    def add_to_anomaly2(self, sample: pd.DataFrame) -> Tuple[str, int]:
        self.anomaly_by_l2 = pd.concat([self.anomaly_by_l2, sample], axis=0)
        return 'l2_anomaly', 1

    def add_to_normal1(self, sample: pd.DataFrame) -> Tuple[str, int]:
        self.normal_traffic = pd.concat([self.normal_traffic, sample], axis=0)
        return 'normal_traffic', 1

    def add_to_normal2(self, sample: pd.DataFrame) -> Tuple[str, int]:
        self.normal_traffic = pd.concat([self.normal_traffic, sample], axis=0)
        return 'normal_traffic', 1
