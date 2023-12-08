import json
import os
import time
import boto3

from KBStorage import Storage
import FeaturesSelector

from Shared import SQSWrapper, Utils

LOGGER = Utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class KnowledgeBase:
    FULL_CLOSE = False
    DEBUG = True

    def __init__(self):

        LOGGER.info('Creating an instance of KnowledgeBase.')
        self.storage = Storage()

        self.__sqs_setup()

    def __sqs_setup(self):
        self.sqs_resource = boto3.resource('sqs')
        self.connector = SQSWrapper.Connector(
            sqs_resource=self.sqs_resource,
            queue_names=['tuner-update.fifo', 'detection-system-update.fifo']
        )

    def terminate(self):
        self.connector.close()

    def __select_features_procedure(self, feature_selection_func):

        query_dict = {
            "select": "*",
            "from": "x_train"
        }
        if feature_selection_func(self.storage.perform_query(query_dict)):

            update_msg = 'UPDATED FEATURES'
            self.connector.send_message_to_queues(update_msg, None)

        else:
            LOGGER.error('Feature selection function failed. Retry.')

    def __test(self):
        test_dict = {
            "UPDATE": ["FEATURES", "TRAIN", "VALIDATE"]
        }
        msg_body = json.dumps(test_dict)
        self.connector.send_message_to_queues(msg_body, None)

    def run_tasks(self):

        action_mapping = {
            1: FeaturesSelector.perform_icfs,
            2: FeaturesSelector.perform_sfs,
            3: FeaturesSelector.perform_bfs,
            4: FeaturesSelector.perform_fisher
        }

        LOGGER.info('Running background tasks..')
        time.sleep(2)

        try:
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

        except KeyboardInterrupt:
            raise KeyboardInterrupt('Keyboard interrupt. Terminating knowledge base instance.')


if __name__ == '__main__':
    kb = KnowledgeBase()

    try:
        kb.run_tasks()
    except KeyboardInterrupt:
        if kb.FULL_CLOSE:
            kb.terminate()
            LOGGER.info('Deleting queues..')

        Utils.save_current_timestamp()
        LOGGER.info('Received keyboard interrupt. Terminating knowledge base instance.')
