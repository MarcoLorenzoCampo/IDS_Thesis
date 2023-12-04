import copy
import logging
import sqlite3
import time
from typing import Union

import boto3
import pandas as pd

from DSConnectionHandler import Connector
from LoaderDetectionSystem import Loader
from Metrics import Metrics
import DataProcessor
import LoggerConfig

LOGGER = logging.getLogger('DetectionSystem')
logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
LOGGER.info('Creating an instance of DetectionSystem.')


class DetectionSystem:

    def __init__(self):
        self.ANOMALY_THRESHOLD1, self.ANOMALY_THRESHOLD2, self.BENIGN_THRESHOLD = 0.9, 0.8, 0.6
        self.cat_features = ['flag', 'protocol_type', 'service']

        self.__s3_setup_and_load()
        self.__load_data_instances()
        self.__set_local_storage()

        self.__sqlite3_setup()
        self.__set_switchers()
        self.metrics = Metrics()

        self.__sqs_setup()

    def __sqs_setup(self):
        queue_url = 'https://sqs.eu-west-3.amazonaws.com/818750160971/detection-system-update.fifo'
        self.sqs_resource = boto3.resource('sqs')
        self.connector = Connector(sqs=self.sqs_resource, queue_url=queue_url)

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

    def __set_switchers(self):
        LOGGER.info('Loading the switch cases.')
        self.clf_switcher = {
            'QUARANTINE': self.__add_to_quarantine,
            'L1_ANOMALY': self.__add_to_anomaly1,
            'L2_ANOMALY': self.__add_to_anomaly2,
            'NOT_ANOMALY1': self.__add_to_normal1,
            'NOT_ANOMALY2': self.__add_to_normal2
        }

        self.metrics_switcher = {
            ('NOT_ANOMALY1', 1): lambda: self.metrics.update_count('fn', 1, 1),
            ('NOT_ANOMALY1', 0): lambda: self.metrics.update_count('tn', 1, 1),
            ('NOT_ANOMALY2', 1): lambda: self.metrics.update_count('fn', 1, 2),
            ('NOT_ANOMALY2', 0): lambda: self.metrics.update_count('tn', 1, 2),
            ('L1_ANOMALY', 0): lambda: self.metrics.update_count('fp', 1, 1),
            ('L1_ANOMALY', 1): lambda: self.metrics.update_count('tp', 1, 1),
            ('L2_ANOMALY', 0): lambda: self.metrics.update_count('fp', 1, 2),
            ('L2_ANOMALY', 1): lambda: self.metrics.update_count('tp', 1, 2),
        }

    def classify(self, incoming_data, actual: int = None):
        unprocessed_sample = copy.deepcopy(incoming_data)
        prediction1, computation_time, cpu_usage = self.__clf_layer1(unprocessed_sample)

        self.metrics.add_cpu_usage(cpu_usage)
        self.metrics.add_classification_time(computation_time)

        if prediction1:
            self.__finalize_clf(incoming_data, [1, 'L1_ANOMALY'], actual)
            return
        else:
            self.__finalize_clf(incoming_data, [0, 'NOT_ANOMALY1'], actual)
            anomaly_confidence, computation_time, cpu_usage = self.__clf_layer2(unprocessed_sample)

            self.metrics.add_cpu_usage(cpu_usage)
            self.metrics.add_classification_time(computation_time)

            benign_confidence_2 = 1 - anomaly_confidence[0, 1]

            if anomaly_confidence[0, 1] >= self.ANOMALY_THRESHOLD2:
                self.__finalize_clf(incoming_data, [anomaly_confidence, 'L2_ANOMALY'], actual)
                return

            if benign_confidence_2 >= self.BENIGN_THRESHOLD:
                self.__finalize_clf(incoming_data, [benign_confidence_2, 'NOT_ANOMALY2'], actual)
                return

        self.__finalize_clf(incoming_data, [0, 'QUARANTINE'], actual)

    def __clf_layer1(self, unprocessed_sample):
        sample = DataProcessor.data_process(unprocessed_sample, self.scaler1, self.ohe1,
                                            self.pca1, self.features_l1, self.cat_features)

        start = time.time()
        prediction1 = self.layer1.predict(sample)
        cpu_usage = 0
        computation_time = time.time() - start

        return prediction1, computation_time, cpu_usage

    def __clf_layer2(self, unprocessed_sample):
        sample = DataProcessor.data_process(unprocessed_sample, self.scaler2, self.ohe2,
                                            self.pca2, self.features_l2, self.cat_features)

        start = time.time()
        anomaly_confidence = self.layer2.predict_proba(sample)
        cpu_usage = 0
        computation_time = time.time() - start

        return anomaly_confidence, computation_time, cpu_usage

    def __finalize_clf(self, sample: pd.DataFrame, output: list[Union[int, str]], actual: int = None):
        if actual is None:
            switch_function = self.clf_switcher.get(output[1], lambda: "Invalid value")
            switch_function(sample)

        if actual is not None:
            switch_function = self.clf_switcher.get(output[1], lambda: "Invalid value")
            switch_function(sample)

        switch_function = self.metrics_switcher.get((output[1], actual), lambda: "Invalid value")
        switch_function()

    def __add_to_quarantine(self, sample: pd.DataFrame) -> None:
        self.quarantine_samples = pd.concat([self.quarantine_samples, sample], axis=0)
        self.metrics.update_classifications('quarantine', 1)

    def __add_to_anomaly1(self, sample: pd.DataFrame) -> None:
        self.anomaly_by_l1 = pd.concat([self.anomaly_by_l1, sample], axis=0)
        self.metrics.update_classifications(tag='l1_anomaly', value=1)

    def __add_to_anomaly2(self, sample: pd.DataFrame) -> None:
        self.anomaly_by_l2 = pd.concat([self.anomaly_by_l2, sample], axis=0)
        self.metrics.update_classifications('l2_anomaly', 1)

    def __add_to_normal1(self, sample: pd.DataFrame) -> None:
        self.normal_traffic = pd.concat([self.normal_traffic, sample], axis=0)
        self.metrics.update_classifications('normal_traffic', 1)

    def __add_to_normal2(self, sample: pd.DataFrame) -> None:
        self.normal_traffic = pd.concat([self.normal_traffic, sample], axis=0)
        self.metrics.update_classifications('normal_traffic', 1)

    def __parse_messages(self, messages):
        pass

    def poll_queues(self):
        while True:
            messages = self.connector.receive_messages(100, 0)

            if messages:
                for message in messages:
                    LOGGER.info(f'Parsing message: {message}'.format(message=message))
                    self.__parse_messages(message)


def main():
    ds = DetectionSystem()

    while True:
        ds.poll_queues()


if __name__ == '__main__':
    main()
