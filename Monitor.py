import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# If l1_classifier predicts anomaly, then check if it's an anomaly.
# If l1_classifier does not predict anomaly, then feed it to l2_classifier.
# The final prediction is from l2_classifier
# Evaluate the results on the final outcome.


def load_models():
    with open('Models/NSL_l1_classifier.pkl', "rb") as f:
        l1 = pickle.load(f)
    with open('Models/NSL_l2_classifier.pkl', 'rb') as f:
        l2 = pickle.load(f)
    return l1, l2


def load_preprocessed_tests():
    x_test_l1 = pd.read_csv('NSL-KDD Encoded Datasets/X_test_l1.txt', header=0)
    x_test_l2 = pd.read_csv('NSL-KDD Encoded Datasets/X_test_l2.txt', header=0)
    y_test = np.load('NSL-KDD Encoded Datasets/y_test.npy', allow_pickle=True)

    for col in x_test_l1.columns:
        pd.to_numeric(x_test_l1[col], errors='coerce')

    for col in x_test_l2.columns:
        pd.to_numeric(x_test_l1[col], errors='coerce')

    return x_test_l1, x_test_l2, y_test


def main():
    classifier1, classifier2 = load_models()
    x_l1, x_l2, y_test = load_preprocessed_tests()

    # convert to np arrays
    x_test_l1 = x_l1.values
    x_test_l2 = x_l2.values

    result = []
    for i in range(x_test_l2.shape[0]):
        layer1 = classifier1.predict(x_test_l1[i].reshape(1, -1))[0]
        if layer1 == 1:
            result.append(layer1)
        else:
            layer2 = classifier2.predict(x_test_l2[i].reshape(1, -1))[0]
            if layer2 == 1:
                result.append(layer2)
            else:
                result.append(0)

    result = np.array(result)

    print(result)


main()
