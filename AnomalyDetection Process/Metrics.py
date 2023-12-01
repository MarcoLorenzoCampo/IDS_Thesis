import json
import logging
import threading
import time

import numpy as np
from CustomExceptions import accuracyException, precisionException, fException, tprException, fprException, \
    tnrException, fnrException

LOGGER = logging.getLogger('Metrics')
LOG_FORMAT = '%(asctime)-10s %(levelname)-10s %(name)-45s %(funcName)-35s %(lineno)-5d: %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class Metrics:
    def __init__(self):

        file_path = 'metrics_thresholds.json'
        try:
            with open(file_path, 'r') as file:
                metrics_thresholds = json.load(file)
                self._metrics_thresh_1 = metrics_thresholds['_metrics_thresh_1']
                self._metrics_thresh_2 = metrics_thresholds['_metrics_thresh_2']
                self._time_interval = metrics_thresholds['time_interval']
                self._max_usage = metrics_thresholds['max_usage']
                self._max_clf_time = metrics_thresholds['max_clf_time']
                LOGGER.info("Metrics thresholds loaded from file.")
        except FileNotFoundError:
            LOGGER.error(f"Metrics thresholds file not found: {file_path}")

        # count of the outputs for layer1
        self._count_1 = {
            'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0,
            'all': 0
        }

        # count of the outputs for layer2
        self._count_2 = {
            'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0,
            'all': 0
        }

        # metrics computed from count_1
        self._metrics_1 = {
            'accuracy': 0.0,
            'precision': 0.0,
            'fscore': 0.0,
            'tpr': 0.0,
            'fpr': 0.0,
            'tnr': 0.0,
            'fnr': 0.0}

        # metrics computed from count_2
        self._metrics_2 = {
            'accuracy': 0.0,
            'precision': 0.0,
            'fscore': 0.0,
            'tpr': 0.0,
            'fpr': 0.0,
            'tnr': 0.0,
            'fnr': 0.0}

        # destination of classifications
        self._overall = {
            'total': 0,
            'quarantine': 0,
            'l1_anomaly': 0,
            'l2_anomaly': 0,
            'normal_traffic': 0
        }

        self._classification_metrics = {
            'normal_ratio': 0.0,
            'quarantine_ratio': 0.0,
            'l1_anomaly_ratio': 0.0,
            'l2_anomaly_ratio': 0.0
        }

        # additional metrics
        self.classification_times = []
        self.cpu_usages = []
        self._tprs_1 = []
        self._fprs_1 = []
        self._tprs_2 = []
        self._fprs_2 = []

        # Event to signal when there's enough data for analysis
        self.enough_data_event = threading.Event()
        # unset the event because it starts with no data
        self.enough_data_event.clear()

    def __compute_performance_metrics(self, target: int):

        if target == 1:
            counts = self._count_1
            metrics = self._metrics_1
            tprs = self._tprs_1
            fprs = self._fprs_1
        else:
            counts = self._count_2
            metrics = self._metrics_2
            tprs = self._tprs_2
            fprs = self._fprs_2

        # Calculate true positive rate (recall)
        tpr = counts['tp'] / (counts['tp'] + counts['fn'])
        metrics['tpr'] = tpr
        tprs.append(tpr)

        # Calculate false positive rate
        fpr = counts['fp'] / (counts['fp'] + counts['tn'])
        metrics['fpr'] = fpr
        fprs.append(fpr)

        # Calculate true negative rate
        metrics['tnr'] = counts['tn'] / (counts['tn'] + counts['fn'])

        # Calculate false negative rate
        metrics['fnr'] = counts['fn'] / (counts['tn'] + counts['fn'])

        # Calculate accuracy
        metrics['accuracy'] = (counts['tp'] + counts['tn']) / (
                counts['tp'] + counts['tn'] + counts['fp'] + counts['fn'])

        # Calculate precision
        metrics['precision'] = counts['tp'] / (counts['tp'] + counts['fp'])

        # Calculate F1 score
        metrics['fscore'] = 2 * (metrics['precision'] * metrics['tpr']) / (
                metrics['precision'] + metrics['tpr'])

        # set the event, enough data has been collected
        self.enough_data_event.set()

    def update_count(self, tag, value, layer: int):
        # increase the count of encountered traffic samples
        self._overall['total'] += 1

        # Update metrics based on the specified layer
        count_dict = self._count_1 if layer == 1 else self._count_2

        count_dict['all'] += value
        count_dict[tag] += value

        # Compute metrics only if enough samples have been collected
        if all(val != 0 for val in count_dict.values()):
            self.__compute_performance_metrics(target=layer)
        else:
            LOGGER.error(f'Not enough data for LAYER{layer}, skipping metrics computation for now.')

        self.__compute_classification_metrics()

    def __compute_classification_metrics(self):
        # normal ratio computation
        self._classification_metrics['normal_ratio'] = self._overall['normal_traffic'] / self._overall['total']

        # quarantine ratio computation
        self._classification_metrics['quarantine_ratio'] = self._overall['quarantine'] / self._overall['total']

        # l1_anomaly ratio computation
        self._classification_metrics['l1_anomaly_ratio'] = self._overall['l1_anomaly'] / self._overall['total']

        # l2_anomaly ratio computation
        self._classification_metrics['l2_anomaly_ratio'] = self._overall['l2_anomaly'] / self._overall['total']

    def update_classifications(self, tag, value):
        self._overall[tag] += value

    def analyze_metrics(self):

        LOGGER.info('Analyzing the metrics..')

        if self._metrics_1['accuracy'] < self._metrics_thresh_1['accuracy_t']:
            LOGGER.info("Accuracy for Layer 1 fell below the threshold.")
            raise accuracyException

        if self._metrics_1['precision'] < self._metrics_thresh_1['precision_t']:
            LOGGER.info("Precision for Layer 1 fell below the threshold.")
            raise precisionException

        if self._metrics_1['fscore'] < self._metrics_thresh_1['fscore_t']:
            LOGGER.info("Fscore for Layer 1 fell below the threshold.")
            raise fException

        if self._metrics_1['tpr'] < self._metrics_thresh_1['tpr_t']:
            LOGGER.info("Tpr for Layer 1 fell below the threshold.")
            raise tprException

        if self._metrics_1['fpr'] < self._metrics_thresh_1['fpr_t']:
            LOGGER.info("Fpr for Layer 1 fell below the threshold.")
            raise fprException

        if self._metrics_1['tnr'] < self._metrics_thresh_1['tnr_t']:
            LOGGER.info("Tnr for Layer 1 fell below the threshold.")
            raise tnrException

        if self._metrics_1['fnr'] < self._metrics_thresh_1['fnr_t']:
            LOGGER.info("Fnr for Layer 1 fell below the threshold.")
            raise fnrException

    def show_metrics(self):

        LOGGER.info('Accuracy for layer 1: %s', self._metrics_1['accuracy'])
        LOGGER.info('Precision for layer 1: %s', self._metrics_1['precision'])
        LOGGER.info('F-score for layer 1: %s', self._metrics_1['fscore'])
        LOGGER.info('TPR for layer 1: %s', self._metrics_1['tpr'])
        LOGGER.info('FPR for layer 1: %s', self._metrics_1['fpr'])
        LOGGER.info('TNR for layer 1: %s', self._metrics_1['tnr'])
        LOGGER.info('FNR for layer 1: %s', self._metrics_1['fnr'])
        LOGGER.info('\n')

        LOGGER.info('Accuracy for layer 2: %s', self._metrics_2['accuracy'])
        LOGGER.info('Precision for layer 2: %s', self._metrics_2['precision'])
        LOGGER.info('F-score for layer 2: %s', self._metrics_2['fscore'])
        LOGGER.info('TPR for layer 2: %s', self._metrics_2['tpr'])
        LOGGER.info('FPR for layer 2: %s', self._metrics_2['fpr'])
        LOGGER.info('TNR for layer 2: %s', self._metrics_2['tnr'])
        LOGGER.info('FNR for layer 2: %s', self._metrics_2['fnr'])
        LOGGER.info('\n')

        LOGGER.info('Normal ratio: %s', self._classification_metrics['normal_ratio'])
        LOGGER.info('L1 anomalies ratio: %s', self._classification_metrics['l1_anomaly_ratio'])
        LOGGER.info('L2 anomalies ratio: %s', self._classification_metrics['l2_anomaly_ratio'])
        LOGGER.info('Quarantined ratio: %s', self._classification_metrics['quarantine_ratio'])

    def reset(self):
        # reset the metrics and counts
        self._count_1 = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'all': 0}
        self._count_2 = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'all': 0}
        self._metrics_1 = {'accuracy': 0.0, 'precision': 0.0, 'fscore': 0.0, 'tpr': 0.0, 'fpr': 0.0, 'tnr': 0.0,
                           'fnr': 0.0}
        self._metrics_2 = {'accuracy': 0.0, 'precision': 0.0, 'fscore': 0.0, 'tpr': 0.0, 'fpr': 0.0, 'tnr': 0.0,
                           'fnr': 0.0}
        self.classification_times = []
        self.cpu_usages = []
        self._tprs_1 = []
        self._fprs_1 = []
        self._tprs_2 = []
        self._fprs_2 = []

    def add_classification_time(self, time):
        self.classification_times.append(time)

    def add_cpu_usage(self, usage):
        self.cpu_usages.append(usage)

    def get_avg_time(self):
        return np.mean(self.classification_times)

    def get_counts(self, tag):
        return self._count_1[tag] + self._count_2[tag]

    def get_metrics(self, tag):
        return (self._metrics_1[tag] + self._metrics_2[tag]) / 2
