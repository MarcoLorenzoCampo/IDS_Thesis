import numpy as np
import pandas as pd

from DetectionInfrastructure import DetectionInfrastructure


def launch_on_testset(detection_infrastructure: DetectionInfrastructure):
    # Load a testing dataset
    x_test = pd.read_csv('NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
    y_test = np.load('NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)

    iterations = 500

    # Test the set using this infrastructure
    for i, (index, row) in enumerate(x_test.iterrows()):

        # reduce the number of iterations for testing purposes
        if i >= iterations:
            break

        print(f'Sample #{i}:')

        # Make each row as its own data frame and pre-process it
        sample = pd.DataFrame(data=np.array([row]), index=None, columns=x_test.columns)
        actual = y_test[index]
        output = detection_infrastructure.ids.classify(sample)
        detection_infrastructure.ids.evaluate_classification(sample, output, actual=actual)

        if i in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
            print('accuracy: ', detection_infrastructure.ids.metrics.get_metrics('accuracy'))
            print('precision: ', detection_infrastructure.ids.metrics.get_metrics('precision'))
            print('fscore: ', detection_infrastructure.ids.metrics.get_metrics('fscore'))

    # let's see the output of the classification
    print('Anomalies by l1: ', detection_infrastructure.ids.anomaly_by_l1.shape[0])
    print('Anomalies by l2: ', detection_infrastructure.ids.anomaly_by_l2.shape[0])
    print('Normal traffic: ', detection_infrastructure.ids.normal_traffic.shape[0])
    print('Quarantined samples: ', detection_infrastructure.ids.quarantine_samples.shape[0])

    # print the outcomes
    print('Classified = ANOMALY, Actual = ANOMALY: tp -> ', detection_infrastructure.ids.metrics.get_counts('tp'))
    print('Classified = ANOMALY, Actual = NORMAL: fp -> ', detection_infrastructure.ids.metrics.get_counts('fp'))
    print('Classified = NORMAL, Actual = ANOMALY: fn -> ', detection_infrastructure.ids.metrics.get_counts('fn'))
    print('Classified = NORMAL, Actual = NORMAL: tn -> ', detection_infrastructure.ids.metrics.get_counts('tn'))

    # print the average of the computation time
    print(f'Average computation time for {iterations} samples: ', detection_infrastructure.ids.metrics.get_avg_time())


def main():
    # Launch a new detection infrastructure instance
    detection_infrastructure = DetectionInfrastructure()
    launch_on_testset(detection_infrastructure)

    return 0


if __name__ == '__main__':
    main()
