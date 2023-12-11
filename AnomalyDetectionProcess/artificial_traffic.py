import numpy as np
import pandas as pd


class Runner:
    counter = 1

    def __init__(self):
        # load the test sets as simulated traffic samples
        self.x_test = pd.read_csv('AWS Downloads/Datasets/OriginalDatasets/KDDTest+.txt', sep=",", header=0)
        self.y_test = np.load('AWS Downloads/Datasets/OriginalDatasets/KDDTest+_targets.npy', allow_pickle=True)

    def get_packet(self):
        if self.counter < self.x_test.shape[0]:
            row = self.x_test.iloc[self.counter]
            sample = pd.DataFrame(data=[row], columns=self.x_test.columns)
            actual = self.y_test[self.counter]

            self.counter += 1
            return sample, actual
        else:
            # Return None if the counter exceeds the number of rows
            return None, None
