import json
import logging
import os
import threading
import time

import boto3
from botocore.exceptions import ClientError

from AnomalyDetectionProcess import DataProcessor
from KBProcess import LoggerConfig
from AnomalyDetectionProcess.SQSWrapper import Connector


logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
filename = os.path.splitext(os.path.basename(__file__))[0]
LOGGER = logging.getLogger(filename)

class Analyzer:

    def __init__(self, polling_timer: float):

        self.polling_timer = polling_timer
        path = 'thresholds.json'
        with open(path, 'r') as f:
            json_file = json.load(f)

        self._metrics_thresholds_1 = json_file['_metrics_thresh_1']
        self._metrics_thresholds_2 = json_file['_metrics_thresh_2']

        self.__sqs_setup()

    def __sqs_setup(self):
        queue_url = 'https://sqs.eu-west-3.amazonaws.com/818750160971/forward-metrics.fifo'
        self.sqs_client = boto3.client('sqs')
        self.sqs_resource = boto3.resource('sqs')

        self.connector = Connector(
            sqs_client=self.sqs_client,
            sqs_resource=self.sqs_resource,
            queue_urls=[queue_url],
            queue_names=['forward-objs.fifo']
        )

    def analyze(self, metrics1: dict, metrics2: dict, classification_metrics: dict):
        pass

    def poll_queues(self):
        while True:
            LOGGER.info('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()
            except ClientError:
                LOGGER.error("Couldn't fetch messages from queue. Restarting the program.")
                raise KeyboardInterrupt

            if msg_body:
                LOGGER.info(f'Parsing message: {json.dumps(msg_body, indent=2)}')
                metrics1, metrics2, classification_metrics = DataProcessor.parse_metrics_msg(msg_body)
                LOGGER.info(f'Parsed message: {metrics1, metrics2, classification_metrics}')

            time.sleep(self.polling_timer)


def main():
    args = DataProcessor.process_command_line_args()
    analyzer = Analyzer(polling_timer=1)

    queue_reading_thread = threading.Thread(target=analyzer.poll_queues)

    try:
        queue_reading_thread.start()
    except KeyboardInterrupt:
        LOGGER.info('Closing the instance.')
    pass


if __name__ == '__main__':
    main()
