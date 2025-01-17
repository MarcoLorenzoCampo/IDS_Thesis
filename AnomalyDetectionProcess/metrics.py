import json
import os
import threading

from Shared import utils
from Shared.msg_enum import msg_type


class Metrics:

    BEGIN_SNAPSHOTS = False

    def __init__(self):

        import detection_system_main
        self.LOGGER = detection_system_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        self.metrics_lock = threading.Lock()

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
            self.BEGIN_SNAPSHOTS = True
            self.__compute_performance_metrics(target=layer)
        else:
            self.LOGGER.error(f'Not enough data for LAYER{layer}, skipping metrics computation for now.')

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

    def get_lock(self):
        return self.metrics_lock

    def reset(self):
        # reset the metrics and counts
        self._count_1 = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'all': 0}
        self._count_2 = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'all': 0}
        self._metrics_1 = {'accuracy': 0.0, 'precision': 0.0, 'fscore': 0.0, 'tpr': 0.0, 'fpr': 0.0, 'tnr': 0.0,
                           'fnr': 0.0}
        self._metrics_2 = {'accuracy': 0.0, 'precision': 0.0, 'fscore': 0.0, 'tpr': 0.0, 'fpr': 0.0, 'tnr': 0.0,
                           'fnr': 0.0}
        self._tprs_1 = []
        self._fprs_1 = []
        self._tprs_2 = []
        self._fprs_2 = []

    def get_counts(self, tag):
        return self._count_1[tag] + self._count_2[tag]

    def get_metrics(self):
        return self._metrics_1, self._metrics_2, self._classification_metrics

    def snapshot_metrics(self):
        self.LOGGER.debug('Building a json snapshot of current metrics')

        metrics_dict = {
            "MSG_TYPE": str(msg_type.METRICS_SNAPSHOT_MSG),
            "metrics_1": {
                "accuracy": self._metrics_1['accuracy'],
                "precision": self._metrics_1['precision'],
                "tpr": self._metrics_1['tpr'],
                "fpr": self._metrics_1['fpr'],
                "tnr": self._metrics_1['tnr'],
                "fnr": self._metrics_1['fnr'],
                "fscore": self._metrics_1['fscore']
            },
            "metrics_2": {
                "accuracy": self._metrics_2['accuracy'],
                "precision": self._metrics_2['precision'],
                "tpr": self._metrics_2['tpr'],
                "fpr": self._metrics_2['fpr'],
                "tnr": self._metrics_2['tnr'],
                "fnr": self._metrics_2['fnr'],
                "fscore": self._metrics_2['fscore']
            },
            "classification_metrics": {
                "normal_ratio": self._classification_metrics['normal_ratio'],
                "l1_anomaly_ratio": self._classification_metrics['l1_anomaly_ratio'],
                "l2_anomaly_ratio": self._classification_metrics['l2_anomaly_ratio'],
                "quarantined_ratio": self._classification_metrics['quarantine_ratio']
            }
        }

        self.write_performance_log(metrics_dict)

        return metrics_dict

    def write_performance_log(self, metrics_dict):
        # Save metrics just before forwarding them

        utils.pprint_to_file(
            "performance_log.txt",
            json.dumps(metrics_dict)
        )