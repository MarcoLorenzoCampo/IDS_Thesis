import pickle

import pandas as pd


def load_models():
    with open('Models/NSL_l1_classifier.pkl', "rb") as f:
        l1 = pickle.load(f)
    with open('Models/NSL_l2_classifier.pkl', 'rb') as f:
        l2 = pickle.load(f)

    return l1, l2

def load_testsets():
    x_test_l1 = pd.read_csv('NSL-KDD Encoded Datasets/K')

    # Load the NumPy array.
    array = np.load('my_array.npy')


def main():
    l1_classifier, l2_classifier = load_models()
    x_test_l1, y_testl1, x_test_l2, y_test_l2 = load_testsets()