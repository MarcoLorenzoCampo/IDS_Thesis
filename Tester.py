import copy

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from ModelMaker import ModelMaker

pd.set_option('display.max_columns', None)
pd.options.display.max_columns = None
maker = ModelMaker()


def main():
    layer1, layer2 = maker.train_models([], [])

    # Consider the test sets as a whole
    x_test = maker.x_test
    x_labels = x_test['label']
    del x_test['label']
    y_test = maker.y_test

    for index, row in x_test.iterrows():

        # Make each row as its own data frame and pre-process it
        sample = pd.DataFrame(data=np.array([row]), index=None, columns=x_test.columns)
        actual = y_test[index]
        processed_sample = maker.pipeline_data_process(sample, target_layer=1)

        # See if it's an anomaly for layer 1
        output = test_pipeline(layer1, processed_sample)

        if output[0]:
            print(f'([Prediction: {output[0]}, AnomalyScore: {output[1]}], Actual: {actual})')
        else:
            # See if it's an anomaly for layer 2
            processed_sample = maker.pipeline_data_process(sample, target_layer=2)
            output = test_pipeline(layer2, processed_sample)

            print(f'([Prediction: {output[0]}, AnomalyScore: {output[1]}], Actual: {actual})')


def test_pipeline(layer, sample: np.array) -> list[int, float]:
    """Tests the given sample on the given layers.

  Args:
    layer: A random forest classifier.
    sample: A NumPy array containing the sample to test.

  Returns:
    A list containing two elements:
      * The first element is an integer indicating whether the sample is an anomaly
        (1) or not (0).
      * The second element is a float indicating the anomaly score of the sample.
        A higher score indicates a more likely anomaly.
  """
    # Start with layer1
    if isinstance(layer, RandomForestClassifier):
        anomaly_score = layer.predict_proba(sample)[0][1]
        is_anomaly = anomaly_score >= maker.ANOMALY_THRESHOLD1
        if is_anomaly:
            return [int(is_anomaly), anomaly_score]

    # Continue with layer 2 if layer 1 does not detect anomalies
    if isinstance(layer, SVC):
        anomaly_score = layer.decision_function(sample)
        is_anomaly = anomaly_score >= maker.ANOMALY_THRESHOLD2
        if is_anomaly:
            return [int(is_anomaly), anomaly_score]

    return [0, 0]


main()
