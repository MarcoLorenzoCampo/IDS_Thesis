import argparse
import sys

sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/AnomalyDetectionProcess')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/KBProcess')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/Other')

import json
import os
import threading

import boto3
import time

from Shared.msg_enum import msg_type
from tuner import Tuner
from storage import Storage
from Shared import utils
from Shared.sqs_wrapper import Connector

LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class Hypertuner:
    FULL_CLOSE = False
    DEBUG = True

    OPTIMIZATION_LOCK = False

    def __init__(self, n_trials: int):

        self.storage = Storage()
        self.tuner = Tuner(n_trials=n_trials, storage=self.storage)
        self.__sqs_setup()

        # new optimal models
        self.new_opt_layer1 = None
        self.new_opt_layer2 = None

    def __sqs_setup(self):
        """
        Set up AWS SQS client, SQS resource, and Connector for handling SQS queues.
        This method initializes the SQS client and resource using the boto3 library.
        It also defines the URLs of the queues to fetch messages from
        and names of the queue names to be created (if not already present)
        """
        self.sqs_client = boto3.client('sqs')
        self.sqs_resource = boto3.resource('sqs')

        queue_urls = [
            'https://sqs.eu-west-3.amazonaws.com/818750160971/forward-objectives.fifo',
            'https://sqs.eu-west-3.amazonaws.com/818750160971/tuner-update.fifo',
        ]

        queue_names = [
            'tuned-models-kb.fifo',  # Send new models notification to knowledge base
            'tuned-models-ds.fifo',  # Send new models notification to the detection system
        ]

        self.connector = Connector(
            sqs_client=self.sqs_client,
            sqs_resource=self.sqs_resource,
            queue_urls=queue_urls,
            queue_names=queue_names
        )

    def terminate(self):
        self.FULL_CLOSE = True
        self.connector.close()

    def poll_queues(self):
        while True:
            LOGGER.debug('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()

            except Exception as e:
                LOGGER.error(f"Error in fetching messages from queue: {e}")
                raise KeyboardInterrupt

            if msg_body:

                json_dict = json.loads(msg_body)

                if json_dict['MSG_TYPE'] == str(msg_type.MODEL_UPDATE_MSG):
                    LOGGER.debug(f'Received update notification: {json_dict}')
                    self.storage.loader.s3_models()

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

                elif json_dict['MSG_TYPE'] == str(msg_type.OBJECTIVES_MSG):
                    LOGGER.debug(f'Received objectives notification: {json_dict}')

                    if not self.OPTIMIZATION_LOCK:
                        LOGGER.debug(f'Parsing message: {json_dict}')
                        objectives = utils.parse_objs(json_dict)

                        self.OPTIMIZATION_LOCK = True

                        self.tuner.objs_map(objectives)

                        self.storage.publish_s3_models()

                        LOGGER.debug('Models have been tuned and updated. Forwarding models update notification.')

                        msg_body = {
                            "MSG_TYPE": str(msg_type.MODEL_UPDATE_MSG)
                        }

                        self.connector.send_message_to_queues(msg_body)

                        self.OPTIMIZATION_LOCK = False
                    else:
                        LOGGER.debug('Process locked. Optimization in progress.')

            time.sleep(2)

    def run_test(self):

        LOGGER.warning('TEST: Testing the tuning engine with a fake objectives set.')
        test_objs = {
            "layer1": ['accuracy', 'precision'],
            "layer2": ['tpr', 'fpr']
        }

        self.tuner.objs_map(test_objs)

    def run_tasks(self):
        queue_reading_thread = threading.Thread(target=self.poll_queues, daemon=True)

        if not self.DEBUG:
            queue_reading_thread.start()
        else:
            self.run_test()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            utils.save_current_timestamp()
            LOGGER.error('Terminating Hypertuner instance.')
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
    hypertuner = Hypertuner(n_trials=100)

    try:
        hypertuner.run_tasks()
    except KeyboardInterrupt:
        if hypertuner.FULL_CLOSE:
            hypertuner.terminate()
            LOGGER.error('Deleting queues..')
        else:
            LOGGER.error('Received keyboard interrupt. Preparing to terminate threads.')
