import argparse
import sys

sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/AnomalyDetectionProcess')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/KBProcess')
sys.path.append('C:/Users/marco/PycharmProjects/IDS_Thesis/Other')

import json
import os
import threading
import time

import boto3

from Shared import utils
from Shared.sqs_wrapper import Connector
from Shared.msg_enum import msg_type


LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])


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
            "MSG_TYPE": str(msg_type.OBJECTIVES_MSG),
            "objs_layer1": [],
            "objs_layer2": []
        }

        for metric, value in metrics1.items():
            if metrics1[metric] < self._metrics_thresholds_1[metric]:
                objectives['objs_layer1'].append(metric)

        for metric, value in metrics2.items():
            if metrics2[metric] < self._metrics_thresholds_2[metric]:
                objectives['objs_layer2'].append(metric)

        LOGGER.debug(f'Identified {len(objectives["objs_layer1"])} objective(s) for layer1: [{objectives["objs_layer1"]}]')
        LOGGER.debug(f'Identified {len(objectives["objs_layer2"])} objective(s) for layer2: [{objectives["objs_layer2"]}]')

        return objectives

    def poll_queues(self):
        while True:
            LOGGER.debug('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()
            except Exception as e:
                LOGGER.error(f"Error in fetching messages from queue: {e}")
                raise KeyboardInterrupt

            if msg_body:
                LOGGER.debug(f'Parsing message: {msg_body}')
                json_dict = json.loads(msg_body)

                if json_dict['MSG_TYPE'] == str(msg_type.METRICS_SNAPSHOT_MSG):
                    metrics1, metrics2, classification_metrics = utils.parse_metrics_msg(json_dict)
                    objectives = self.analyze(metrics1, metrics2, classification_metrics)

                    self.connector.send_message_to_queues(objectives)
                else:
                    LOGGER.debug(f'Received message of type {json_dict["MSG_TYPE"]}')

            time.sleep(self.polling_timer)

    def run_tasks(self):
        queue_reading_thread = threading.Thread(target=self.poll_queues, daemon=True)

        queue_reading_thread.start()

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

    parser.add_argument('-polling_timer',
                        type=float,
                        default=5,
                        help='Specify the polling timer (float)'
                        )

    args = parser.parse_args()

    polling_timer = args.polling_timer

    if polling_timer is not None:
        LOGGER.debug(f'Polling Timer: {polling_timer}')

    return polling_timer


if __name__ == '__main__':
    arg = process_command_line_args()
    analyzer = Analyzer(polling_timer=arg)

    try:
        analyzer.run_tasks()
    except KeyboardInterrupt:
        if analyzer.FULL_CLOSE:
            analyzer.terminate()
            LOGGER.debug('Deleting queues..')
        else:
            LOGGER.debug('Received keyboard interrupt. Preparing to terminate threads.')
