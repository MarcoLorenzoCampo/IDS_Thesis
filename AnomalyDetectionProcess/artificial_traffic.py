import copy
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTENC
from matplotlib import pyplot as plt


class Runner:
    counter = 1

    def __init__(self):
        # load the test sets as simulated traffic samples
        self.x_test = pd.read_csv('AWS Downloads/Datasets/OriginalDatasets/KDDTest+.txt', sep=",", header=None)
        self.y_test = np.load('AWS Downloads/Datasets/OriginalDatasets/KDDTest+_targets.npy', allow_pickle=True)

    def get_packet(self):
        """
        Get the next packet from the test set.

        Returns:
            tuple: A tuple containing the next packet and its corresponding label.
            If there are no more packets, it returns None.
        """
        if self.counter < self.x_test.shape[0]:
            row = self.x_test.iloc[self.counter]
            sample = pd.DataFrame(data=[row], columns=self.x_test.columns)
            actual = self.y_test[self.counter]

            self.counter += 1
            return sample, actual
        else:
            return None, None
