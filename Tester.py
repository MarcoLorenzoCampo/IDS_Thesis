import copy

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from DetectionSystem import ModelMaker

pd.set_option('display.max_columns', None)
pd.options.display.max_columns = None
maker = ModelMaker()


def main():
    layer1, layer2 = maker.train_models()

    # Consider the test sets as a whole
    x_test = maker.x_test
    x_labels = x_test['label']
    del x_test['label']
    y_test = maker.y_test

    for index, row in x_test.iterrows():

        # Make each row as its own data frame and pre-process it
        sample = pd.DataFrame(data=np.array([row]), index=None, columns=x_test.columns)
        actual = y_test[index]
        output = test_pipeline(layer1, layer2, sample)

        # it's a case that is unsure for the IDS
        if output[0] == 0:
            maker.quarantine_samples = pd.concat([maker.quarantine_samples, sample], axis=1)
            print(f'([Prediction: QUARANTINE, AnomalyScore: {output[1]}], Actual: {actual})')
        # it's an anomaly signaled by l1
        elif output[0] == 1:
            maker.anomaly_by_l1 = pd.concat([maker.anomaly_by_l1, sample], axis=1)
            print(f'([Prediction: L1_ANOMALY, AnomalyScore: {output[1]}], Actual: {actual})')
        # it's an anomaly signaled by l2
        elif output[0] == 2:
            maker.anomaly_by_l2 = pd.concat([maker.anomaly_by_l2, sample], axis=1)
            print(f'([Prediction: L2_ANOMALY, AnomalyScore: {output[1]}], Actual: {actual})')
        # it's not an anomaly
        elif output[0] == 3:
            maker.normal_traffic = pd.concat([maker.normal_traffic, sample], axis=1)
            print(f'([Prediction: NORMAL, AnomalyScore: {output[1]}], Actual: {actual})')


def test_pipeline(layer1, layer2, unprocessed_sample: np.array) -> list[int, float]:
    """Tests the given sample on the given layers.

  Args:
    layer1: A random forest classifier.
    layer2: A support vector machine
    sample: A NumPy array containing the sample to test.

  Returns:
    A list containing two elements:
      * The first element is an integer indicating whether the sample is an anomaly
        0: unsure, quarantines the sample for further analysis
        1: anomaly signaled by layer1
        2: anomaly signaled by layer2
        3: not an anomaly
      * The second element is a float indicating the anomaly score of the sample.
        A higher score indicates a more likely anomaly.

    This is the strategy:
    If it's signaled as an anomaly by l1 with an anomaly_score higher than the threshold,
    then return it. If the benign_confidence is high enough, then return it as a not anomaly.
    If it's neither an anomaly nor normal traffic for l1, then forward it to l2.
    If it's signaled as an anomaly by l2 with an anomaly_core higher than the threshold,
    then return it. If the benign_confidence is high enough, then return it as a not anomaly.
    If it's neither an anomaly nor normal traffic for l2, then it's a case in which the labeling is unsure
    and so it needs further analysis.

  """
    # evaluate if the traffic is malicious
    # Start with layer1 (random forest)
    sample = maker.pipeline_data_process(unprocessed_sample, target_layer=1)
    anomaly_confidence = layer1.predict_proba(sample)[0][1]
    benign_confidence = 1 - anomaly_confidence

    if anomaly_confidence >= maker.ANOMALY_THRESHOLD1:
        # it's an anomaly for layer1
        return [1, anomaly_confidence]
    else:
        if benign_confidence >= maker.BENIGN_THRESHOLD:
            return [3, benign_confidence]

    # Continue with layer 2 if layer 1 does not detect anomalies
    sample = maker.pipeline_data_process(unprocessed_sample, target_layer=2)
    anomaly_confidence = layer2.decision_function(sample)
    benign_confidence = 1 - anomaly_confidence
    if anomaly_confidence >= maker.ANOMALY_THRESHOLD2:
        # it's an anomaly for layer2
        return [2, anomaly_confidence]
    else:
        if benign_confidence >= maker.BENIGN_THRESHOLD:
            return [3, benign_confidence]

    return [0, 0]


main()
