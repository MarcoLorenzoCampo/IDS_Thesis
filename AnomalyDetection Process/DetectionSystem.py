import copy
import logging
import re
import time
from typing import Union
import threading

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from LocalDataStorage import Data
from DSConnectionHandler import Connector
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

        self.storage = Data()
        self.metrics = Metrics()
        self.__sqs_setup()

    def __sqs_setup(self):
        queue_url = 'https://sqs.eu-west-3.amazonaws.com/818750160971/detection-system-update.fifo'
        self.sqs_client = boto3.client('sqs')
        self.connector = Connector(sqs_client=self.sqs_client, queue_url=queue_url)

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
        sample = DataProcessor.data_process(unprocessed_sample, self.storage.scaler1, self.storage.ohe1,
                                            self.storage.pca1, self.storage.features_l1, self.cat_features)

        start = time.time()
        prediction1 = self.storage.layer1.predict(sample)
        cpu_usage = 0
        computation_time = time.time() - start

        return prediction1, computation_time, cpu_usage

    def __clf_layer2(self, unprocessed_sample):
        sample = DataProcessor.data_process(unprocessed_sample, self.storage.scaler2, self.storage.ohe2,
                                            self.storage.pca2, self.storage.features_l2, self.cat_features)

        start = time.time()
        anomaly_confidence = self.storage.layer2.predict_proba(sample)
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
        self.quarantine_samples = pd.concat([self.storage.quarantine_samples, sample], axis=0)
        self.metrics.update_classifications('quarantine', 1)

    def __add_to_anomaly1(self, sample: pd.DataFrame) -> None:
        self.anomaly_by_l1 = pd.concat([self.storage.anomaly_by_l1, sample], axis=0)
        self.metrics.update_classifications(tag='l1_anomaly', value=1)

    def __add_to_anomaly2(self, sample: pd.DataFrame) -> None:
        self.anomaly_by_l2 = pd.concat([self.storage.anomaly_by_l2, sample], axis=0)
        self.metrics.update_classifications('l2_anomaly', 1)

    def __add_to_normal1(self, sample: pd.DataFrame) -> None:
        self.normal_traffic = pd.concat([self.normal_traffic, sample], axis=0)
        self.metrics.update_classifications('normal_traffic', 1)

    def __add_to_normal2(self, sample: pd.DataFrame) -> None:
        self.normal_traffic = pd.concat([self.normal_traffic, sample], axis=0)
        self.metrics.update_classifications('normal_traffic', 1)

    def poll_queues(self):
        while True:
            LOGGER.info('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()
            except ClientError:
                LOGGER.error("Couldn't fetch messages from queue. Restarting the program.")
                raise KeyboardInterrupt

            if msg_body:
                LOGGER.info(f'Parsing message: {msg_body}')
                parsed = DataProcessor.parse_message_body(msg_body)

            time.sleep(1.5)

    def run_classification(self):
        while True:
            try:
                LOGGER.info('Classifying data..')
            except KeyboardInterrupt:
                LOGGER.info('Closing the instance.')
                raise KeyboardInterrupt
            except Exception as e:
                LOGGER.error(e)
                LOGGER.info('Closing the instance.')
                raise KeyboardInterrupt

            time.sleep(1.5)

def main():
    ds = DetectionSystem()

    queue_reading_thread = threading.Thread(target=ds.poll_queues)
    classification_thread = threading.Thread(target=ds.run_classification)

    try:
        queue_reading_thread.start()
    except KeyboardInterrupt:
        LOGGER.info('Closing the instance.')

    try:
        classification_thread.start()
    except KeyboardInterrupt:
        LOGGER.info('Closing the instance.')

    classification_thread.join()


if __name__ == '__main__':
    main()
