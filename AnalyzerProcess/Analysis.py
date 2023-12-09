import json
import logging
import os
import threading
import time

import boto3

from Shared import Utils
from Shared.SQSWrapper import Connector

LOGGER = Utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])


class Analyzer:
    FULL_CLOSE = False

    def __init__(self, polling_timer: float):

        self.polling_timer = polling_timer
        path = 'thresholds.json'
        with open(path, 'r') as f:
            json_file = json.load(f)

        self._metrics_thresholds_1 = json_file['_metrics_thresh_1']
        self._metrics_thresholds_2 = json_file['_metrics_thresh_2']

        self.__sqs_setup()

    def __sqs_setup(self):
        self.sqs_client = boto3.client('sqs')
        self.sqs_resource = boto3.resource('sqs')

        self.queue_urls = [
            'https://sqs.eu-west-3.amazonaws.com/818750160971/forward-metrics.fifo',
        ]
        self.queue_names = [
            'forward-objectives.fifo'
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

    def analyze(self, metrics1: dict, metrics2: dict, classification_metrics: dict):
        objectives = {
            "layer1": [],
            "layer2": []
        }
        for metric, value in metrics1.items():
            if metrics1[metric] < self._metrics_thresholds_1[metric]:
                objectives['layer1'].append(metric)

        for metric, value in metrics2.items():
            if metrics2[metric] < self._metrics_thresholds_2[metric]:
                objectives['layer2'].append(metric)

        LOGGER.info(f'Identified {len(objectives["layer1"])} objective(s) for layer1: [{objectives["layer1"]}]')
        LOGGER.info(f'Identified {len(objectives["layer2"])} objective(s) for layer2: [{objectives["layer2"]}]')

        return objectives

    def poll_queues(self):
        while True:
            LOGGER.info('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()
            except Exception as e:
                LOGGER.error(f"Error in fetching messages from queue: {e}")
                raise KeyboardInterrupt

            if msg_body:
                LOGGER.info(f'Parsing message: {msg_body}')
                metrics1, metrics2, classification_metrics = Utils.parse_metrics_msg(msg_body)
                objectives = self.analyze(metrics1, metrics2, classification_metrics)

                self.connector.send_message_to_queues(objectives)

            time.sleep(self.polling_timer)

    def run_tasks(self):
        queue_reading_thread = threading.Thread(target=self.poll_queues, daemon=True)

        queue_reading_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            LOGGER.info("Received keyboard interrupt. Preparing to terminate threads.")
            Utils.save_current_timestamp()

        finally:
            LOGGER.info('Terminating DetectionSystem instance.')
            raise KeyboardInterrupt


if __name__ == '__main__':
    arg1, arg2, arg3 = Utils.process_command_line_args()
    analyzer = Analyzer(polling_timer=arg2)

    try:
        analyzer.run_tasks()
    except KeyboardInterrupt:
        if analyzer.FULL_CLOSE:
            analyzer.terminate()
            LOGGER.info('Deleting queues..')
        else:
            LOGGER.info('Received keyboard interrupt. Preparing to terminate threads.')
