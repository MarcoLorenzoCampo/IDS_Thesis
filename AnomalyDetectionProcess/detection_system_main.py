import argparse
import json
import sys

from Shared.msg_enum import msg_type

sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/AnomalyDetectionProcess')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/KBProcess')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/Other')

import copy
import os
import time
from typing import Union
import threading

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from AnomalyDetectionProcess.artificial_traffic import Runner
from storage import Storage
from Shared.sqs_wrapper import Connector
from metrics import Metrics

from Shared import utils

LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class DetectionSystem:
    FULL_CLOSE = False
    DEBUG = True

    def __init__(self, metrics_snapshot_timer: float, polling_timer: float, classification_delay: float):

        self.metrics_snapshot_timer = metrics_snapshot_timer
        self.polling_timer = polling_timer
        self.classification_delay = classification_delay

        self.metrics_switcher = {}
        self.clf_switcher = {}

        self.storage = Storage()
        self.metrics = Metrics()
        self.__set_switchers()
        self.__sqs_setup()

        # only for testing purposes
        self.runner = Runner()

    def __sqs_setup(self):
        self.sqs_client = boto3.client('sqs')
        self.sqs_resource = boto3.resource('sqs')

        self.queue_urls = [
            'https://sqs.eu-west-3.amazonaws.com/818750160971/detection-system-update.fifo',
        ]

        self.queue_names = [
            'forward-metrics.fifo',
        ]

        self.connector = Connector(
            sqs_client=self.sqs_client,
            sqs_resource=self.sqs_resource,
            queue_urls=self.queue_urls,
            queue_names=self.queue_names
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
        sample = utils.data_process(unprocessed_sample, self.storage.scaler1, self.storage.ohe1,
                                    self.storage.pca1, self.storage.features_l1, self.storage.cat_features)
        return self.storage.layer1.predict(sample)

    def __clf_layer2(self, unprocessed_sample):
        sample = utils.data_process(unprocessed_sample, self.storage.scaler2, self.storage.ohe2,
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
        LOGGER.debug('Loading the switch cases.')
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
        while True:
            # LOGGER.debug('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()
            except Exception as e:
                LOGGER.error(f"Error in fetching messages from queue: {e}")
                raise KeyboardInterrupt

            if msg_body:

                json_dict = json.loads(msg_body)

                if json_dict['MSG_TYPE'] == str(msg_type.MODEL_UPDATE_MSG):
                    LOGGER.debug('Parsed an UPDATE MODELS message, updating from S3.')
                    self.storage.loader.s3_models()

                    self.storage.layer1, self.storage.layer2 = (
                        self.storage.loader.load_models('NSL_l1_classifier.pkl', 'NSL_l2_classifier.pkl')
                    )
                    LOGGER.debug('Replaced current models with models from S3.')

                elif json_dict['MSG_TYPE'] == str(msg_type.MULTIPLE_UPDATE_MSG):
                    to_update = json_dict['UPDATE']

                    LOGGER.debug(f'Received multiple update notification: {to_update}')

                    update_calls = {
                        'FEATURES': self.storage.loader.s3_features,
                        'TRAIN': self.storage.loader.s3_train,
                        'VALIDATE': self.storage.loader.s3_validate,
                    }

                    for update in to_update:
                        update_calls[update]()

                else:
                    LOGGER.debug(f'Received unexpected message of type {json_dict["MSG_TYPE"]}')

            time.sleep(self.polling_timer)

    def run_classification(self):
        while True:
            try:
                with self.metrics.get_lock():
                    # LOGGER.debug('Classifying data..')
                    sample, actual = self.runner.get_packet()
                    self.classify(sample, actual)
            except Exception as e:
                LOGGER.error(f"Error in classification: {e}")
                raise KeyboardInterrupt

            time.sleep(self.classification_delay)

    def snapshot_metrics(self):
        while True:
            if self.metrics.ALLOW_SNAPSHOT:
                with self.metrics.get_lock():
                    # LOGGER.debug('Snapshotting metrics..')
                    json_output = self.metrics.snapshot_metrics()
                    msg_body = json_output if json_output is not None else "ERROR"

                try:
                    self.connector.send_message_to_queues(msg_body)
                except ClientError as e:
                    LOGGER.error(f"Error in snapshot metrics: {e}")
                    raise KeyboardInterrupt

                time.sleep(self.metrics_snapshot_timer)

    def run_tasks(self):
        queue_reading_thread = threading.Thread(target=self.poll_queues, daemon=True)
        classification_thread = threading.Thread(target=self.run_classification, daemon=True)
        metrics_snapshot_thread = threading.Thread(target=self.snapshot_metrics, daemon=True)

        queue_reading_thread.start()
        classification_thread.start()
        metrics_snapshot_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            LOGGER.debug("Received keyboard interrupt. Preparing to terminate threads.")
            utils.save_current_timestamp()
        finally:
            LOGGER.debug('Terminating DetectionSystem instance.')
            raise KeyboardInterrupt


def process_command_line_args():
    parser = argparse.ArgumentParser(description='Process command line arguments for a Python script.')

    parser.add_argument('-metrics_snapshot_timer',
                        type=float,
                        default=10,
                        help='Specify the metrics snapshot timer (float)'
                        )
    parser.add_argument('-polling_timer',
                        type=float,
                        default=5,
                        help='Specify the polling timer (float)'
                        )
    parser.add_argument('-classification_delay',
                        type=float,
                        default=1,
                        help='Specify the classification delay (float)'
                        )

    args = parser.parse_args()

    # Access the arguments using dot notation
    metrics_snapshot_timer = args.metrics_snapshot_timer
    polling_timer = args.polling_timer
    classification_delay = args.classification_delay

    # You can check if the arguments are provided and then use them in your script
    if metrics_snapshot_timer is not None:
        LOGGER.debug(f'Metrics Snapshot Timer: {metrics_snapshot_timer}')

    if polling_timer is not None:
        LOGGER.debug(f'Polling Timer: {polling_timer}')

    if classification_delay is not None:
        LOGGER.debug(f'Classification Delay: {classification_delay}')

    return metrics_snapshot_timer, polling_timer, classification_delay


if __name__ == '__main__':
    arg1, arg2, arg3 = process_command_line_args()
    ds = DetectionSystem(arg1, arg2, arg3)

    try:
        ds.run_tasks()
    except KeyboardInterrupt:
        if ds.FULL_CLOSE:
            ds.terminate()
            LOGGER.debug('Deleting queues..')
        else:
            LOGGER.debug('Received keyboard interrupt. Preparing to terminate threads.')
