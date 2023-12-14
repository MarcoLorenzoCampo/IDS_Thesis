import json
import os
import threading
import time
import boto3

from storage import Storage
import features_selector

from Shared.msg_enum import msg_type
from Shared import sqs_wrapper, utils

LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class KnowledgeBase:
    FULL_CLOSE = False
    DEBUG = True

    def __init__(self):

        LOGGER.info('Creating an instance of KnowledgeBase.')
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
            LOGGER.info('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()

            except Exception as e:
                LOGGER.error(f"Error in fetching messages from queue: {e}")
                raise KeyboardInterrupt

            if msg_body:

                json_dict = json.loads(msg_body)

                if json_dict['MSG_TYPE'] == str(msg_type.MODEL_UPDATE_MSG):
                    LOGGER.info(f'Received update notification: {json_dict}')
                    self.storage.loader.s3_models()
                else:
                    LOGGER.info(f'Unexpected message type received: {json_dict["MSG_TYPE"]}')

            time.sleep(2)

    def __select_features_procedure(self, feature_selection_func):

        query_dict = {
            "select": "*",
            "from": "x_train"
        }
        if feature_selection_func(self.storage.perform_query(query_dict)):

            update_msg = {
                "MSG_TYPE": str(msg_type.MULTIPLE_UPDATE_MSG),
                "UPDATE": ['FEATURES', 'TRAIN', 'VALIDATE']
            }

            self.connector.send_message_to_queues(update_msg)

        else:
            LOGGER.error('Feature selection function failed. Retry.')

    def input_reading(self):

        action_mapping = {
            1: self.features_selector.perform_icfs,
            2: self.features_selector.perform_sfs,
            3: self.features_selector.perform_bfs,
            4: self.features_selector.perform_fisher,
            5: self.features_selector.analyze_datasets
        }

        while True:

            print("\nSelect the number of the action to perform:"
                  "\n1. ICFS feature selection"
                  "\n2. SFS feature selection"
                  "\n3. BFS feature selection"
                  "\n4. Fisher feature selection"
                  "\n5. Analyze the dataset for changes"
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
                    print("Invalid action number. Please enter a number between 1 and 4.")
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
            utils.save_current_timestamp()

        finally:
            LOGGER.info('Terminating KnowledgeBase instance.')
            raise KeyboardInterrupt


if __name__ == '__main__':
    kb = KnowledgeBase()

    try:
        kb.run_tasks()
    except KeyboardInterrupt:
        if kb.FULL_CLOSE:
            kb.terminate()
            LOGGER.info('Deleting queues..')

        utils.save_current_timestamp()
        LOGGER.info('Received keyboard interrupt. Terminating knowledge base instance.')
