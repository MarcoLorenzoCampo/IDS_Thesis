import argparse
import logging
import sys
import os
import json
import threading
import time
import boto3

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from optimizer import OptimizationManager, RFTrainer, SVMTrainer
from Shared.msg_enum import msg_type
from TunerProcess.tuner import TuningHandler
from TunerProcess.storage import Storage, S3Manager, SQLiteManager
from Shared import utils
from Shared.sqs_wrapper import Connector
from Shared.message_handler import FullMsgHandler
from tuner import Tuner, TunerLayer1, TunerLayer2

LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class MsgManager(FullMsgHandler):

    def __init__(self, storage: Storage, polling_timer: float, tuner: TuningHandler):
        self.polling_timer = polling_timer

        self.storage = storage
        self.tuner = tuner
        self.__sqs_setup()

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

    def process_message(self, msg_body: str):
        json_dict = json.loads(msg_body)

        if json_dict['MSG_TYPE'] == str(msg_type.MODEL_UPDATE_MSG):
            self.handle_models_update_msg(json_dict)

        elif json_dict['MSG_TYPE'] == str(msg_type.MULTIPLE_UPDATE_MSG):
            self.handle_models_update_msg(json_dict)

        elif json_dict['MSG_TYPE'] == str(msg_type.OBJECTIVES_MSG):
            self.handle_objs_msg(json_dict)

    def handle_models_update_msg(self, json_dict: dict):
        LOGGER.debug(f'Received update notification: {json_dict}')
        self.storage.loader.s3_models()

    def handle_objs_msg(self, json_dict: dict):
        LOGGER.debug(f'Received objectives notification: {json_dict}')

        models_update_msg = self.tuner.on_tuning_msg_received(json_dict)
        self.connector.send_message_to_queues(models_update_msg)

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

    def poll_queues(self):
        while True:
            LOGGER.debug('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()
                self.process_message(msg_body)

            except Exception as e:
                LOGGER.error(f"Error in fetching messages from queue: {e}")
                raise KeyboardInterrupt

            time.sleep(self.polling_timer)


class HypertunerMain:
    FULL_CLOSE = False
    DEBUG = True

    def __init__(self, reading_timer: float, message_manager: MsgManager, storage: Storage, tuner: TuningHandler):

        self.polling_timer = reading_timer

        self.message_manager = message_manager
        self.storage = storage

        self.tuner = tuner

    def terminate(self):
        self.__set_full_close()
        self.message_manager.connector.close()

    def run_test(self):

        LOGGER.warning('TEST: Testing the tuning engine with a fake objectives set.')
        test_objs = {
            "objs_layer1": ['accuracy', 'precision'],
            "objs_layer2": ['tpr', 'fpr']
        }
        self.tuner.on_tuning_msg_received(test_objs)

    def run_tasks(self):
        queue_reading_thread = threading.Thread(target=self.message_manager.poll_queues, daemon=True)

        if not self.DEBUG:
            queue_reading_thread.start()
        else:
            self.run_test()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            utils.save_current_timestamp("")
        finally:
            LOGGER.debug('Terminating DetectionSystem instance.')
            raise KeyboardInterrupt

    def __set_full_close(self):
        self.FULL_CLOSE = True


class CommandLineParser:

    @staticmethod
    def process_command_line_args():
        # Args: -n_cores, -polling_timer, -verbose
        parser = argparse.ArgumentParser(description='Process command line arguments for a Python script.')

        parser.add_argument('-n_cores',
                            type=int,
                            default=-1,
                            help='Specify the number of cores (int)'
                            )
        parser.add_argument('-polling_timer',
                            type=float,
                            default=5,
                            help='Specify the polling timer (float)'
                            )
        parser.add_argument('-n_trials',
                            type=int,
                            default=100,
                            help='Specify the number of trials for tuning (default 100) (int)'
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
        cores = args.n_cores
        timer = args.polling_timer
        trials = args.n_trials

        # You can check if the arguments are provided and then use them in your script
        if cores is not None:
            LOGGER.debug(f'Number of Cores: {cores}')

        if timer is not None:
            LOGGER.debug(f'Polling Timer: {timer}')

        if trials is not None:
            LOGGER.debug(f'Number of Trials: {trials}')

        return cores, timer, trials


def main():
    cores, polling_timer, trials = CommandLineParser.process_command_line_args()

    s3_manager = S3Manager(
        bucket_name='nsl-kdd-datasets'
    )

    storage = Storage(
        s3_manager=s3_manager
    )

    rf_trainer = RFTrainer()
    svm_trainer = SVMTrainer()

    sqlite_manager = SQLiteManager(
        storage=storage
    )

    optimizer = OptimizationManager(
        sqlite_manager=sqlite_manager,
        rf_trainer=rf_trainer,
        svm_trainer=svm_trainer
    )

    layer1_tuner = TunerLayer1(
        n_cores=cores,
        n_trials=trials,
        optimizer=optimizer
    )

    layer2_tuner = TunerLayer2(
        n_cores=cores,
        n_trials=trials,
        optimizer=optimizer
    )

    tuner = Tuner(
        storage=storage,
        optimization_manager=optimizer,
        layer1_tuner=layer1_tuner,
        layer2_tuner=layer2_tuner
    )

    tuning_handler = TuningHandler(
        storage=storage,
        optimizer=optimizer,
        tuner=tuner
    )

    message_manager = MsgManager(
        storage=storage,
        polling_timer=polling_timer,
        tuner=tuning_handler
    )

    hypertuner = HypertunerMain(
        reading_timer=polling_timer,
        message_manager=message_manager,
        storage=storage,
        tuner=tuning_handler
    )

    try:
        hypertuner.run_tasks()
    except KeyboardInterrupt:
        if hypertuner.FULL_CLOSE:
            hypertuner.terminate()
            LOGGER.error('Deleting queues..')
        else:
            if tuning_handler.tuning_state.is_tuning_complete():
                msg_body = {
                    "MSG_TYPE": str(msg_type.MODEL_UPDATE_MSG),
                    "SENDER": 'Hypertuner'
                }

                hypertuner.message_manager.connector.send_message_to_queues(msg_body)

            LOGGER.error('Received keyboard interrupt. Preparing to terminate threads.')


if __name__ == '__main__':
    main()
