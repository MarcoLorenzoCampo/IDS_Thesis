import argparse
import json
import logging
import sys
import os
import time
import threading
import boto3

from botocore.exceptions import ClientError

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from Shared.message_handler import FullMsgHandler
from artificial_traffic import Runner
from storage import Storage
from Shared.sqs_wrapper import Connector
from metrics import Metrics
from Shared.msg_enum import msg_type
from Shared import utils
from classification_pipeline import ClassificationProcess


LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])

class DetectionSystemMain(FullMsgHandler):
    FULL_CLOSE = False
    stop_forward_metrics = threading.Event()

    def __init__(self, metrics_snapshot_timer: float, polling_timer: float, classification_delay: float,
                 storage: Storage, classification_pipeline: ClassificationProcess):

        self.snapshot_event = threading.Event()
        self.metrics_snapshot_timer = metrics_snapshot_timer
        self.polling_timer = polling_timer
        self.classification_delay = classification_delay
        self.classification_pipeline = classification_pipeline
        self.storage = storage
        self.__sqs_setup()

        # only for testing purposes
        self.runner = Runner()

    def __sqs_setup(self):
        self.sqs_client = boto3.client('sqs')
        self.sqs_resource = boto3.resource('sqs')

        self.queue_urls = [
            'https://sqs.eu-west-3.amazonaws.com/818750160971/detection-system-update.fifo',
            'https://sqs.eu-west-3.amazonaws.com/818750160971/tuned-models-ds.fifo'
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

    def terminate(self):
        self.FULL_CLOSE = True
        self.connector.close()

    def poll_queues(self):
        while True:
            LOGGER.info('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()
            except Exception as e:
                LOGGER.error(f"Error in fetching messages from queue: {e}")
                raise KeyboardInterrupt

            if msg_body:

                json_dict = json.loads(msg_body)

                # Case 1: Update models sent from Hypertuner or KnowledgeBase
                if json_dict['MSG_TYPE'] == str(msg_type.MODEL_UPDATE_MSG):
                    self.handle_models_update_msg(json_dict)

                # Case 2: Multiple objects update from KnowledgeBase
                elif json_dict['MSG_TYPE'] == str(msg_type.MULTIPLE_UPDATE_MSG):
                    self.handle_multiple_updates_msg(json_dict)

                else:
                    LOGGER.debug(f'Received unexpected message of type {json_dict["MSG_TYPE"]}')

            time.sleep(self.polling_timer)

    def handle_multiple_updates_msg(self, json_dict: dict):
        to_update = json_dict['UPDATE']

        LOGGER.debug(f'Received multiple update notification: {to_update}')

        update_calls = {
            'FEATURES': self.storage.loader.s3_features,
            'TRAIN': self.storage.loader.s3_train,
            'VALIDATE': self.storage.loader.s3_validate,
        }

        for update in to_update:
            update_calls[update]()

    def handle_objs_msg(self, json_dict: dict):
        pass

    def handle_models_update_msg(self, json_dict: dict):
        LOGGER.debug('Parsed an UPDATE MODELS message, updating from S3.')

        self.storage.loader.s3_models()

        self.storage.layer1, self.storage.layer2 = (
            self.storage.loader.load_models('NSL_l1_classifier.pkl', 'NSL_l2_classifier.pkl')
        )

        # Activate snapshots only after the models have been updated
        if json_dict['SENDER'] == 'Hypertuner':
            LOGGER.debug('Update message from the tuner, starting snapshots back.')
            self.snapshot_event.set()

        LOGGER.debug('Replaced current models with models from S3.')

    def run_classification(self):
        while True:
            try:
                with self.classification_pipeline.metrics.get_lock():
                    LOGGER.info('Classifying data..')
                    sample, actual = self.runner.get_packet()
                    self.classification_pipeline.classify(sample, actual)
            except Exception as e:
                LOGGER.error(f"Error in classification: {e}")
                raise KeyboardInterrupt

            time.sleep(self.classification_delay)

    def snapshot_metrics(self):

        while True:
            if self.classification_pipeline.metrics.BEGIN_SNAPSHOTS:

                # Reset the wait event before forwarding metrics
                self.snapshot_event.clear()

                with self.classification_pipeline.metrics.get_lock():
                    LOGGER.info('Snapshotting metrics..')
                    metrics_json = self.classification_pipeline.metrics.snapshot_metrics()

                    msg_body = metrics_json if metrics_json is not None else "ERROR"

                try:
                    self.connector.send_message_to_queues(msg_body)
                except ClientError as e:
                    LOGGER.error(f"Error in snapshot metrics: {e}")
                    raise KeyboardInterrupt

                # After sending the snapshot, suspend the snapshot process until an answer is received
                self.snapshot_event.wait()

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
            utils.save_current_timestamp("")
        finally:
            LOGGER.debug('Terminating DetectionSystem instance.')
            raise KeyboardInterrupt


class CommandLineParser:

    @staticmethod
    def process_command_line_args():
        parser = argparse.ArgumentParser(description='Process command line arguments for a Python script.')

        parser.add_argument('-metrics_snapshot_timer',
                            type=float,
                            default=120,
                            help='Specify the metrics snapshot timer (float)'
                            )
        parser.add_argument('-polling_timer',
                            type=float,
                            default=5,
                            help='Specify the polling timer (float)'
                            )
        parser.add_argument('-classification_delay',
                            type=float,
                            default=0.5,
                            help='Specify the classification delay (float)'
                            )
        parser.add_argument('-verbose',
                            action='store_true',
                            help='Set the logging default to "DEBUG"'
                            )

        args = parser.parse_args()

        verbose = args.verbose
        if verbose:
            LOGGER.setLevel(logging.DEBUG)

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


def main():
    snapshot_timer, poll_timer, clf_delay = CommandLineParser.process_command_line_args()

    metrics = Metrics()
    storage = Storage()

    classification_pipeline = ClassificationProcess(
        metrics=metrics,
        storage=storage
    )

    ds_main = DetectionSystemMain(
        metrics_snapshot_timer=snapshot_timer,
        polling_timer=poll_timer,
        classification_delay=clf_delay,
        classification_pipeline=classification_pipeline,
        storage=storage
    )

    try:
        ds_main.run_tasks()
    except KeyboardInterrupt:
        if ds_main.FULL_CLOSE:
            ds_main.terminate()
            LOGGER.debug('Deleting queues..')
        else:
            LOGGER.debug('Received keyboard interrupt. Preparing to terminate threads.')


if __name__ == '__main__':
    main()
