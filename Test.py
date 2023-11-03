import numpy as np
import pandas as pd

from DetectionInfrastructure import DetectionInfrastructure


def launch_on_testset(detection_infrastructure: DetectionInfrastructure):
    # Load a testing dataset
    x_test = pd.read_csv('NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
    y_test = np.load('NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)

    # Test the set using this infrastructure
    for i, (index, row) in enumerate(x_test.iterrows()):

        # reduce the number of iterations for testing purposes
        if i >= 20000:
            break

        # Make each row as its own data frame and pre-process it
        sample = pd.DataFrame(data=np.array([row]), index=None, columns=x_test.columns)
        actual = y_test[index]
        output = detection_infrastructure.ids.classify(sample)
        detection_infrastructure.ids.evaluate_classification(sample, output, actual=actual)

    # let's see the output of the classification
    print('Anomalies by l1: ', detection_infrastructure.ids.anomaly_by_l1.shape[0])
    print('Anomalies by l2: ', detection_infrastructure.ids.anomaly_by_l2.shape[0])
    print('Normal traffic: ', detection_infrastructure.ids.normal_traffic.shape[0])
    print('Quarantined samples: ', detection_infrastructure.ids.quarantine_samples.shape[0])


def main():
    # Launch a new detection infrastructure instance
    detection_infrastructure = DetectionInfrastructure()
    launch_on_testset(detection_infrastructure)

    return 0


if __name__ == '__main__':
    main()
