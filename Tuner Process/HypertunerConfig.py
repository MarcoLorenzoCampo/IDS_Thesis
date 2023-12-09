import json
import os
import threading

import boto3
import time

import pandas as pd

from Tuner import Tuner
from HypertunerStorage import Storage
from Shared import Utils
from Shared.SQSWrapper import Connector

LOGGER = Utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])
pd.set_option('display.width', 1000)


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
            LOGGER.info('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()
            except Exception as e:
                LOGGER.error(f"Error in fetching messages from queue: {e}")
                raise KeyboardInterrupt

            if msg_body:
                if not self.OPTIMIZATION_LOCK:
                    LOGGER.info(f'Parsing message: {json.dumps(msg_body, indent=2)}')
                    objectives = Utils.parse_objs(msg_body)

                    self.OPTIMIZATION_LOCK = True

                    self.tuner.objs_map(objectives)
                    self.storage.update_s3_models()

                    msg_body = {
                        'UPDATE': 'MODELS'
                    }

                    self.connector.send_message_to_queues(msg_body)

                    self.OPTIMIZATION_LOCK = False
                else:
                    LOGGER.info('Process locked. Optimization in progress.')

            time.sleep(2)

    def run_tasks(self):
        queue_reading_thread = threading.Thread(target=self.poll_queues, daemon=True)

        queue_reading_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            Utils.save_current_timestamp()

        finally:
            LOGGER.info('Terminating Hypertuner instance.')
            raise KeyboardInterrupt


if __name__ == '__main__':
    hypertuner = Hypertuner(n_trials=1)

    try:
        hypertuner.run_tasks()
    except KeyboardInterrupt:
        if hypertuner.FULL_CLOSE:
            hypertuner.terminate()
            LOGGER.info('Deleting queues..')
        else:
            LOGGER.info('Received keyboard interrupt. Preparing to terminate threads.')
