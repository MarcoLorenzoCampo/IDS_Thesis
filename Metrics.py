import numpy as np
import logging


def set_logger(name):
    # Create a custom logger
    logger = logging.getLogger(name)

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler(f'Logs/{name}.log')  # File handler
    c_handler.setLevel(logging.WARNING)  # Set level for console handler
    f_handler.setLevel(logging.ERROR)  # Set level for file handler

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')  # Console format
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # File format

    c_handler.setFormatter(c_format)  # Set format for console handler
    f_handler.setFormatter(f_format)  # Set format for file handler

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


class Metrics:
    def __init__(self):
        self.count = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        self._metrics = {'accuracy': -1.0, 'precision': -1.0, 'fscore': -1.0,
                         'tpr': -1.0, 'fpr': -1.0, 'tnr': -1.0, 'fnr': -1.0}
        self.classification_times = []
        self.cpu_usages = []

    def compute_metrics(self):
        # true positive rate (recall)
        self._metrics['tpr'] = self.count['tp'] / (self.count['tp'] + self.count['fn'])

        # false positive rate
        self._metrics['fpr'] = self.count['fp'] / (self.count['fp'] + self.count['tn'])

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
        try:
            self.compute_metrics()
        except ZeroDivisionError:
            print("Error: Division by zero occurred in compute_metrics. Could be because of default values.")

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
