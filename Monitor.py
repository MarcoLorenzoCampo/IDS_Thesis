import pickle

import numpy as np
import pandas as pd


def load_models():
    with open('Models/NSL_l1_classifier.pkl', "rb") as f:
        l1 = pickle.load(f)
    with open('Models/NSL_l2_classifier.pkl', 'rb') as f:
        l2 = pickle.load(f)
    return l1, l2


def load_tests():
    x_test = pd.read_csv('NSL-KDD Encoded Datasets/KDDTest+', header=0)
    y_test = np.load('NSL-KDD Encoded Datasets/KDDTest+_targets.npy')
    return x_test, y_test


def main():
    l1_classifier, l2_classifier = load_models()
    x_test, y_test = load_tests()


