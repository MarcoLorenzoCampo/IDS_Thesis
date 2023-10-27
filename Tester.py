import copy

import pandas as pd
import numpy as np

from ModelMaker import ModelMaker

pd.set_option('display.max_columns', None)
maker = ModelMaker()


def main():
    layer1, layer2 = maker.train_models([], [])

    # Consider the test sets as a whole
    x_test = maker.x_test
    x_labels = x_test['label']
    del x_test['label']
    y_test = maker.y_test

    for index, row in x_test.iterrows():
        # See if it's an anomaly for layer 1
        sample = pd.DataFrame(data=np.array([row]), index=None, columns=x_test.columns)
        target = y_test[index]
        processed_sample = maker.pipeline_data_process(sample, target_layer=1)
        output = test_pipeline(layer1, layer2, processed_sample)

        print(output)


def test_pipeline(layer1, layer2, sample):
    if layer1.decision_function(sample) >= maker.ANOMALY_THRESHOLD1:
        return [1, layer1.decision_function(sample)]
    if layer2.decision_function(sample) >= maker.ANOMALY_THRESHOLD2:
        return [1, layer2.decision_function(sample)]

    return [0, 0]


main()
