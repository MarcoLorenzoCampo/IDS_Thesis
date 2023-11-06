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
        self._metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        self.classification_times = []
        self.cpu_usages = []

    def update(self, tag, value):
        self._metrics[tag] += value

    def get_dict(self, tag):
        return self._metrics[tag]

    def add_classification_time(self, time):
        self.classification_times.append(time)

    def add_cpu_usage(self, usage):
        self.cpu_usages.append(usage)

    def get_avg_time(self):
        return np.mean(self.classification_times)
