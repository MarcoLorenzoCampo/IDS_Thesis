import sys
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/AnomalyDetectionProcess')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/KBProcess')

import copy
import json
import logging
import os
import time
import datetime
from typing import Union
import threading

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from Runner import Runner
from LocalDataStorage import Data
from SQSWrapper import Connector
from Metrics import Metrics
import DataProcessor

from KBProcess import LoggerConfig


logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
filename = os.path.splitext(os.path.basename(__file__))[0]
LOGGER = logging.getLogger(filename)

class DetectionSystem:

    FULL_CLOSE = False
    SYSTEM_CLOSE = False

    def __init__(self, metrics_snapshot_timer: float, polling_timer: float, classification_delay: float):

        self.exit_signal = threading.Event()

        self.metrics_snapshot_timer = metrics_snapshot_timer
        self.polling_timer = polling_timer
        self.classification_delay = classification_delay

        self.metrics_switcher = {}
        self.clf_switcher = {}

        self.storage = Data()
        self.metrics = Metrics()
        self.__set_switchers()
        self.__sqs_setup()

        # only for testing purposes
        self.runner = Runner()

    def __sqs_setup(self):
        queue_url = 'https://sqs.eu-west-3.amazonaws.com/818750160971/detection-system-update.fifo'
        self.sqs_client = boto3.client('sqs')
        self.sqs_resource = boto3.resource('sqs')

        self.connector = Connector(
            sqs_client=self.sqs_client,
            sqs_resource=self.sqs_resource,
            queue_urls=[queue_url],
            queue_names=['forward-metrics.fifo']
        )

    def classify(self, incoming_data, actual: int = None):
        unprocessed_sample = copy.deepcopy(incoming_data)
        prediction1 = self.__clf_layer1(unprocessed_sample)

        if prediction1:
            label1, tag1 = 1, 'L1_ANOMALY'
        else:
            label1, tag1 = 0, 'NOT_ANOMALY1'

        self.__finalize_clf(incoming_data, [label1, tag1], actual)

        if not prediction1:
            anomaly_confidence = self.__clf_layer2(unprocessed_sample)

            benign_confidence_2 = 1 - anomaly_confidence[0, 1]

            if anomaly_confidence[0, 1] >= self.storage.ANOMALY_THRESHOLD2:
                self.__finalize_clf(incoming_data, [anomaly_confidence, 'L2_ANOMALY'], actual)
            elif benign_confidence_2 >= self.storage.BENIGN_THRESHOLD:
                self.__finalize_clf(incoming_data, [benign_confidence_2, 'NOT_ANOMALY2'], actual)
            else:
                self.__finalize_clf(incoming_data, [0, 'QUARANTINE'], actual)

    def __clf_layer1(self, unprocessed_sample):
        sample = DataProcessor.data_process(unprocessed_sample, self.storage.scaler1, self.storage.ohe1,
                                            self.storage.pca1, self.storage.features_l1, self.storage.cat_features)
        return self.storage.layer1.predict(sample)

    def __clf_layer2(self, unprocessed_sample):
        sample = DataProcessor.data_process(unprocessed_sample, self.storage.scaler2, self.storage.ohe2,
                                            self.storage.pca2, self.storage.features_l2, self.storage.cat_features)
        return self.storage.layer2.predict_proba(sample)

    def __finalize_clf(self, sample: pd.DataFrame, output: list[Union[int, str]], actual: int = None):
        switch_function = self.clf_switcher.get(output[1], lambda _: ("Invalid value", None))
        tag, value = switch_function(sample)
        self.metrics.update_classifications(tag, value)

        metrics_switch_key = (output[1], actual) if actual is not None else ("Invalid value", None)
        switch_function = self.metrics_switcher.get(metrics_switch_key, lambda: None)
        switch_function()

    def __set_switchers(self):
        LOGGER.info('Loading the switch cases.')
        self.clf_switcher = {
            'QUARANTINE': self.storage.add_to_quarantine,
            'L1_ANOMALY': self.storage.add_to_anomaly1,
            'L2_ANOMALY': self.storage.add_to_anomaly2,
            'NOT_ANOMALY1': self.storage.add_to_normal1,
            'NOT_ANOMALY2': self.storage.add_to_normal2
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

    def terminate(self):
        self.FULL_CLOSE = True
        self.connector.close()

    def poll_queues(self):
        try:
            while not self.exit_signal.is_set():
                LOGGER.info('Fetching messages..')

                try:
                    msg_body = self.connector.receive_messages()
                except ClientError:
                    LOGGER.error("Couldn't fetch messages from queue. Restarting the program.")
                    raise KeyboardInterrupt

                if msg_body:
                    LOGGER.info(f'Parsing message: {msg_body}')
                    parsed = DataProcessor.parse_update_msg(msg_body)
                    LOGGER.info(f'Parsed message: {parsed}')

                time.sleep(self.polling_timer)
        except KeyboardInterrupt:
            LOGGER.error("Received keyboard interrupt. Terminating queue_reading.")
            self.exit_signal.set()

    def run_classification(self):
        try:
            while not self.exit_signal.is_set():
                try:
                    with self.metrics.get_lock():
                        LOGGER.info('Classifying data..')
                        sample, actual = self.runner.get_packet()
                        self.classify(sample, actual)
                except Exception as e:
                    LOGGER.error(f"Error in run_classification: {e}")
                    self.exit_signal.set()

                time.sleep(self.classification_delay)
        except KeyboardInterrupt:
            LOGGER.info("Received keyboard interrupt. Terminating run_classification.")
            self.exit_signal.set()

    def snapshot_metrics(self):
        try:
            while not self.exit_signal.is_set():
                with self.metrics.get_lock():
                    LOGGER.info('Snapshotting metrics..')
                    json_output = self.metrics.snapshot_metrics()
                    msg_body = json.dumps(json_output) if json_output is not None else "ERROR"

                try:
                    self.connector.send_message_to_queues(msg_body)
                except ClientError as e:
                    LOGGER.error(f"Error in snapshot_metrics: {e}")
                    self.exit_signal.set()

                time.sleep(self.metrics_snapshot_timer)
        except KeyboardInterrupt:
            LOGGER.info("Received keyboard interrupt. Terminating snapshot_metrics.")
            self.exit_signal.set()

    def run_tasks(self):
        queue_reading_thread = threading.Thread(target=self.poll_queues, daemon=True)
        classification_thread = threading.Thread(target=self.run_classification, daemon=True)
        metrics_snapshot_thread = threading.Thread(target=self.snapshot_metrics, daemon=True)

        try:
            queue_reading_thread.start()
            #classification_thread.start()
            #metrics_snapshot_thread.start()
            queue_reading_thread.join()
            #classification_thread.join()
            #metrics_snapshot_thread.join()
        except KeyboardInterrupt:
            LOGGER.info("Received keyboard interrupt. Preparing to terminate threads.")

        LOGGER.info('Saving last online timestamp.')
        current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('last_online.txt', 'w') as file:
            file.write(current_timestamp)

        LOGGER.info('Terminating DetectionSystem instance.')


if __name__ == '__main__':
    arg1, arg2, arg3 = DataProcessor.process_command_line_args()
    ds = DetectionSystem(arg1, arg2, arg3)

    try:
        ds.run_tasks()
    except KeyboardInterrupt:
        if ds.SYSTEM_CLOSE:
            ds.terminate()
            LOGGER.info('Deleting queues..')
        else:
            LOGGER.info('Received keyboard interrupt. Preparing to terminate threads.')

        # Optionally, you may add a time.sleep here to give threads time to terminate gracefully.
        time.sleep(5)

