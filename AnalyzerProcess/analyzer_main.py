import argparse
import sys
import json
import os
import threading
import time

import boto3
import logging

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from Shared import utils
from Shared.sqs_wrapper import Connector
from Shared.msg_enum import msg_type
from Shared.message_handler import MetricsMsgHandler
from analyzer import Analyzer


LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])

class MsgHandler(MetricsMsgHandler):

    def __init__(self, polling_timer: float, analyzer: Analyzer):

        self._polling_timer = polling_timer
        self.analyzer = analyzer

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

    def handle_metrics_msg(self, msg_body: str):

        if msg_body:
            LOGGER.debug(f'Parsing message: {msg_body}')
            json_dict = json.loads(msg_body)

            if json_dict['MSG_TYPE'] == str(msg_type.METRICS_SNAPSHOT_MSG):
                metrics1, metrics2, classification_metrics = utils.parse_metrics_msg(json_dict)
                objectives = self.analyzer.analyze_incoming_metrics(metrics1, metrics2, classification_metrics)

                self.connector.send_message_to_queues(objectives)
            else:
                LOGGER.error(f'Received message of type {json_dict["MSG_TYPE"]}')

    def poll_queues(self):
        while True:
            LOGGER.debug('Fetching messages..')

            try:
                msg_body = self.connector.receive_messages()
                if msg_body:
                    self.handle_metrics_msg(msg_body)
            except Exception as e:
                LOGGER.error(f"Error in fetching messages from queue: {e}")
                raise KeyboardInterrupt

            time.sleep(self._polling_timer)


class AnalyzerMain:

    def __init__(self, sqs_manager: MsgHandler, analyzer: Analyzer):

        self.FULL_CLOSE = False
        self.analyzer = analyzer
        self.sqs_manager = sqs_manager

    def terminate_instance(self):
        self.FULL_CLOSE = True
        self.sqs_manager.connector.close()

    def run_tasks(self):
        queue_reading_thread = threading.Thread(target=self.sqs_manager.poll_queues, daemon=True)

        queue_reading_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            LOGGER.debug("Received keyboard interrupt. Preparing to terminate threads.")
            utils.save_current_timestamp("")

        finally:
            raise KeyboardInterrupt

class CommandLineParser:

    @staticmethod
    def process_command_line_args():
        # Args: -polling_timer: time to wait each cycle to read messages, -verbose
        parser = argparse.ArgumentParser(description='Process command line arguments for a Python script.')

        parser.add_argument('-polling_timer',
                            type=float,
                            default=5,
                            help='Specify the polling timer (float)'
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
    timer = CommandLineParser.process_command_line_args()

    analyzer = Analyzer("thresholds.json")
    sqs_manager = MsgHandler(polling_timer=timer, analyzer=analyzer)

    analyzer_main = AnalyzerMain(analyzer=analyzer, sqs_manager=sqs_manager)

    try:
        analyzer_main.run_tasks()
    except KeyboardInterrupt:
        if analyzer_main.FULL_CLOSE:
            analyzer_main.terminate_instance()
            LOGGER.debug('Deleting queues..')
        else:
            LOGGER.debug('Received keyboard interrupt. Preparing to terminate threads.')


if __name__ == '__main__':
    main()
