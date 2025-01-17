import argparse
import logging
import sys
import json
import os
import threading
import time
import boto3

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from KBProcess.storage import Storage
from KBProcess import features_selector
from Shared.msg_enum import msg_type
from Shared import sqs_wrapper, utils

LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class KnowledgeBase:
    FULL_CLOSE = False
    DEBUG = True

    def __init__(self, polling_timer: int):

        self.polling_timer = polling_timer

        LOGGER.debug('Creating an instance of KnowledgeBase.')
        self.storage = Storage()
        self.features_selector = features_selector.DataManager(self.storage)

        self.__sqs_setup()

    def __sqs_setup(self):
        self.sqs_resource = boto3.resource('sqs')
        self.sqs_client = boto3.client('sqs')

        queue_urls = [
            'https://sqs.eu-west-3.amazonaws.com/818750160971/tuned-models-kb.fifo',
        ]

        queue_names = [
            'tuner-update.fifo',
            'detection-system-update.fifo'
        ]

        self.connector = sqs_wrapper.Connector(
            sqs_resource=self.sqs_resource,
            sqs_client=self.sqs_client,
            queue_names=queue_names,
            queue_urls=queue_urls
        )

    def terminate(self):
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
                else:
                    LOGGER.debug(f'Unexpected message type received: {json_dict["MSG_TYPE"]}')

            time.sleep(self.polling_timer)

    def __select_features_procedure(self, feature_selection_func):

        query_dict = {
            "select": "*",
            "from": "x_train"
        }
        if feature_selection_func(self.storage.perform_query(query_dict)):

            update_msg = {
                "MSG_TYPE": str(msg_type.MULTIPLE_UPDATE_MSG),
                "UPDATE": ['FEATURES', 'TRAIN', 'VALIDATE'],
                "SENDER": 'KnowledgeBase'
            }

            self.connector.send_message_to_queues(update_msg)

        else:
            LOGGER.error('Feature selection function failed. Retry.')

    def input_reading(self):

        action_mapping = {
            1: self.features_selector.analyze_datasets
        }

        while True:

            print("\nSelect the number of the action to perform:"
                  "\n1. Analyze the dataset for changes"
                  "\n'exit' to quit to program.")

            choice = input('>> ')

            if choice.lower() == 'exit':
                raise KeyboardInterrupt

            try:
                action_number = int(choice)
                selected_function = action_mapping.get(action_number)
                if selected_function:
                    self.__select_features_procedure(selected_function)
                else:
                    print("Invalid action number.")
                    continue

            except ValueError:
                print("Invalid input. Please enter a valid number.")
                continue

            time.sleep(1.5)

    def run_tasks(self):

        queue_reading_thread = threading.Thread(target=self.poll_queues, daemon=True)
        input_reading_thread = threading.Thread(target=self.input_reading, daemon=True)

        queue_reading_thread.start()
        input_reading_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            utils.save_current_timestamp("/")

        finally:
            LOGGER.debug('Terminating KnowledgeBase instance.')
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
    parser.add_argument('-DEBUG',
                        type=bool,
                        default=False,
                        help='Specify the if additional prints are needed'
                        )
    parser.add_argument('-verbose',
                        action='store_true',
                        help='Set the logging default to "DEBUG"'
                        )

    args = parser.parse_args()

    verbose = args.verbose
    if verbose:
        LOGGER.setLevel(logging.DEBUG)

    polling_timer = args.polling_timer

    if polling_timer is not None:
        LOGGER.debug(f'Polling Timer: {polling_timer}')

    return polling_timer


def main():
    poll_timer = process_command_line_args()
    kb = KnowledgeBase(poll_timer)

    try:
        kb.run_tasks()
    except KeyboardInterrupt:
        if kb.FULL_CLOSE:
            kb.terminate()
            LOGGER.debug('Deleting queues..')

        LOGGER.debug('Received keyboard interrupt. Terminating knowledge base instance.')


if __name__ == '__main__':
    main()
