import numpy as np
import Logger


class Metrics:
    def __init__(self):
        self.logger = Logger.set_logger(__name__)
        self.count = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        self._metrics = {'accuracy': 0.0, 'precision': 0.0, 'fscore': 0.0,
                         'tpr': 0.0, 'fpr': 0.0, 'tnr': 0.0, 'fnr': 0.0}
        self.classification_times = []
        self.cpu_usages = []
        self.tprs = []
        self.fprs = []

    def compute_metrics(self):
        # true positive rate (recall)
        self._metrics['tpr'] = self.count['tp'] / (self.count['tp'] + self.count['fn'])
        self.tprs.append(self._metrics['tpr'])

        # false positive rate
        self._metrics['fpr'] = self.count['fp'] / (self.count['fp'] + self.count['tn'])
        self.fprs.append(self._metrics['fpr'])

        # true negative rate
        self._metrics['tnr'] = self.count['tn'] / (self.count['tn'] + self.count['fn'])

        # false negative rate
        self._metrics['fnr'] = self.count['fn'] / (self.count['tn'] + self.count['fn'])

        # computation of accuracy
        self._metrics['accuracy'] = ((self.count['tp'] + self.count['tn'])
                                     / (self.count['tp'] + self.count['tn'] + self.count['fp'] + self.count['fn']))

        # computation of precision
        self._metrics['precision'] = self.count['tp'] / (self.count['tp'] + self.count['fp'])

        # computation of fscore
        self._metrics['fscore'] = (2 * (self._metrics['precision'] * self._metrics['tpr'])
                                   / (self._metrics['precision'] + self._metrics['tpr']))

    def update_count(self, tag, value):
        self.count[tag] += value

        # compute the metrics only if enough samples have been collected
        if all(value != 0 for value in self.count.values()):
            self.compute_metrics()
        else:
            self.logger.exception('Not enough data, skipping metrics computation for now.')

    def show_metrics(self):
        print('accuracy: ', self._metrics['accuracy'])
        print('precision: ', self._metrics['precision'])
        print('fscore: ', self._metrics['fscore'])
        print('tpr: ', self._metrics['tpr'])
        print('fpr: ', self._metrics['fpr'])
        print('tnr: ', self._metrics['tnr'])
        print('fnr: ', self._metrics['fnr'])
        print('\n')

    def get_tprs(self):
        return self.tprs

    def get_fprs(self):
        return self.fprs

    def get_metrics(self, tag):
        return self._metrics[tag]

    def get_counts(self, tag):
        return self.count[tag]

    def add_classification_time(self, time):
        self.classification_times.append(time)

    def add_cpu_usage(self, usage):
        self.cpu_usages.append(usage)

    def get_avg_time(self):
        return np.mean(self.classification_times)
