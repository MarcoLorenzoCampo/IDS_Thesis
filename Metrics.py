import numpy as np


class Metrics:
    def __init__(self):
        self._metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        self.classification_times = []

    def update(self, tag, value):
        self._metrics[tag] += value

    def get_dict(self, tag):
        return self._metrics[tag]

    def add_classification_time(self, time):
        self.classification_times.append(time)

    def get_avg_time(self):
        return np.mean(self.classification_times)
