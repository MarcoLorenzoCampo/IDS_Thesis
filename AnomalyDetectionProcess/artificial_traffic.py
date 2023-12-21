import copy
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTENC
from matplotlib import pyplot as plt


class Runner:
    counter = 1

    def __init__(self):
        # load the test sets as simulated traffic samples
        self.x_test = pd.read_csv('AWS Downloads/Datasets/OriginalDatasets/KDDTest+.txt', sep=",", header=0)
        self.y_test = np.load('AWS Downloads/Datasets/OriginalDatasets/KDDTest+_targets.npy', allow_pickle=True)

        self.x_test_expanded = pd.read_csv('Files/KDDTest+_expanded_w_targets.csv', sep=",", header=0)
        self.y_test_expanded = np.load('Files/KDDTest+_expanded_only_targets.npy', allow_pickle=True)

    def oversample(self):

        x, y = copy.deepcopy(self.x_test), copy.deepcopy(self.y_test)

        del x['label']

        smote_nc = SMOTENC(sampling_strategy='auto', categorical_features=[1, 2, 3], random_state=42)
        x, y = smote_nc.fit_resample(x, y)

        plt.figure(figsize=(12, 6))
        plt.hist(y, bins=100)
        plt.grid(axis='y')

        x.to_csv('Files/KDDTest+_expanded_w_targets.csv')
        np.save('Files/KDDTest+_expanded_only_targets.npy', y)

    def get_packet(self):
        """
        Get the next packet from the test set.

        Returns:
            tuple: A tuple containing the next packet and its corresponding label. If there are no more packets, returns None.
        """
        if self.counter < self.x_test_expanded.shape[0]:
            row = self.x_test_expanded.iloc[self.counter]
            sample = pd.DataFrame(data=[row], columns=self.x_test_expanded.columns)
            actual = self.y_test_expanded[self.counter]

            self.counter += 1
            return sample, actual
        else:
            return None, None
