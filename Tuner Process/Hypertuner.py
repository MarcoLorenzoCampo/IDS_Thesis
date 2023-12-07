import json
import logging
import os
import threading
from typing import List

import boto3
import optuna
import time

from AnomalyDetectionProcess import Utils
from HypertunerStorage import Storage
from Optimizer import Optimizer
from KBProcess import LoggerConfig
from AnomalyDetectionProcess.SQSWrapper import Connector

logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
filename = os.path.splitext(os.path.basename(__file__))[0]
LOGGER = logging.getLogger(filename)


class Hypertuner:
    FULL_CLOSE = False
    DEBUG = True

    def __init__(self, ):

        self.storage = Storage()
        self.optimizer = Optimizer(n_trials=7)
        self.__sqs_setup()

        # new optimal models
        self.new_opt_layer1 = None
        self.new_opt_layer2 = None

    def __sqs_setup(self):
        self.sqs_client = boto3.client('sqs')
        self.sqs_resource = boto3.resource('sqs')

        self.queue_urls = [
            'https://sqs.eu-west-3.amazonaws.com/818750160971/forward-objectives.fifo',
        ]
        self.queue_names = [
            'tuned-models.fifo'
        ]

        self.connector = Connector(
            sqs_client=self.sqs_client,
            sqs_resource=self.sqs_resource,
            queue_urls=self.queue_urls,
            queue_names=self.queue_names
        )

    def __objs_map(self, objectives: dict):
        pass

    def tune(self, to_opt: str, direction: str):

        self.__optimizer_setup()

        study_l1 = optuna.create_study(study_name=f'RandomForest optimization: {to_opt}', direction=direction)
        study_l2 = optuna.create_study(study_name=f'SupportVectorMachine optimization: {to_opt}', direction=direction)

        if to_opt == 'accuracy':
            LOGGER.info('Optimizing L1 for accuracy.')
            study_l1.optimize(self.optimizer.objective_accuracy_l1)
            study_l2.optimize(self.optimizer.objective_accuracy_l2)

        if to_opt == 'precision':
            LOGGER.info('Optimizing L1 for precision.')
            study_l1.optimize(self.optimizer.objective_precision_l1)
            study_l2.optimize(self.optimizer.objective_precision_l2)

        if to_opt == 'fscore':
            LOGGER.info('Optimizing L1 for fscore.')
            study_l1.optimize(self.optimizer.objective_fscore_l1)
            study_l2.optimize(self.optimizer.objective_fscore_l2)

        if to_opt == 'tpr':
            LOGGER.info('Optimizing L1 for tpr.')
            study_l1.optimize(self.optimizer.objective_tpr_l1)
            study_l2.optimize(self.optimizer.objective_tpr_l2)

        if to_opt == 'tnr':
            LOGGER.info('Optimizing L1 for tnr.')
            study_l1.optimize(self.optimizer.objective_tnr_l1)
            study_l2.optimize(self.optimizer.objective_tnr_l2)

        if to_opt == 'fpr':
            LOGGER.info('Optimizing L1 for fpr.')
            study_l1.optimize(self.optimizer.objective_fpr_l1)
            study_l2.optimize(self.optimizer.objective_fpr_l2)

        if to_opt == 'fnr':
            LOGGER.info('Optimizing L1 for fnr.')
            study_l1.optimize(self.optimizer.objective_fnr_l1)
            study_l2.optimize(self.optimizer.objective_fnr_l2)

        if to_opt == 'quarantine_ratio':
            LOGGER.info('Optimizing L1 for tpr.')
            study_l1.optimize(self.optimizer.objective_quarantine_rate_l1)
            study_l2.optimize(self.optimizer.objective_quarantine_rate_l2)

        # obtain the optimal classifiers from the studies
        self.new_opt_layer1 = self.optimizer.train_new_hps('RandomForest', study_l1.best_params)
        self.new_opt_layer2 = self.optimizer.train_new_hps('SVM', study_l2.best_params)

        self.optimizer.unset()

        return self.new_opt_layer1, self.new_opt_layer2

    def __optimizer_setup(self):

        partial = 'SELECT * FROM '

        LOGGER.info('Obtaining the datasets from SQLite3 in memory database.')
        datasets = []
        for dataset in ['x_train_l1', 'x_train_l2', 'x_validate_l1', 'x_validate_l2']:
            query = partial + dataset
            datasets.append(self.storage.perform_query(query))

        self.optimizer.optimization_wrapper(
            x_train_1=datasets[0],
            x_train_2=datasets[1],
            x_val_1=datasets[2],
            x_val_2=datasets[3],
            y_train_1=datasets[0]['targets'],
            y_train_2=datasets[5]['targets'],
            y_val_1=datasets[2]['targets'],
            y_val_2=datasets[3]['targets'],
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
                LOGGER.info(f'Parsing message: {json.dumps(msg_body, indent=2)}')
                objectives = Utils.parse_objs(msg_body)
                self.tune(objectives)

            time.sleep(2)

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
    hypertuner = Hypertuner()

    try:
        hypertuner.run_tasks()
    except KeyboardInterrupt:
        if hypertuner.FULL_CLOSE:
            hypertuner.terminate()
            LOGGER.info('Deleting queues..')
        else:
            LOGGER.info('Received keyboard interrupt. Preparing to terminate threads.')