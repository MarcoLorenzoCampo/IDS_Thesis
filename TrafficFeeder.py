import numpy as np
import pandas as pd

from Infrastructure import DetectionInfrastructure


def launch_on_testset(detection_infrastructure: DetectionInfrastructure):
    # Load a testing dataset
    x_test = pd.read_csv('NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
    y_test = np.load('NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)

    iterations = 700

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
        detection_infrastructure.ids.show_classification(sample, output, actual=actual)

        if i in list(range(iterations))[::50]:
            print(f'\nAt iterations #{i}:')
            detection_infrastructure.ids.metrics.show_metrics()

    # let's see the output of the classification
    print(f'Classification of {iterations} test samples:')
    print('Anomalies by l1: ', detection_infrastructure.ids.anomaly_by_l1.shape[0])
    print('Anomalies by l2: ', detection_infrastructure.ids.anomaly_by_l2.shape[0])
    print('Normal traffic: ', detection_infrastructure.ids.normal_traffic.shape[0])
    print('Quarantined samples: ', detection_infrastructure.ids.quarantine_samples.shape[0])

    # print the outcomes
    print('\nOverall classification:')
    print('Classified = ANOMALY, Actual = ANOMALY: tp -> ', detection_infrastructure.ids.metrics.get_counts('tp'))
    print('Classified = ANOMALY, Actual = NORMAL: fp -> ', detection_infrastructure.ids.metrics.get_counts('fp'))
    print('Classified = NORMAL, Actual = ANOMALY: fn -> ', detection_infrastructure.ids.metrics.get_counts('fn'))
    print('Classified = NORMAL, Actual = NORMAL: tn -> ', detection_infrastructure.ids.metrics.get_counts('tn'))

    # print the average of the computation time
    print(f'Average computation time for {iterations} samples: ', detection_infrastructure.ids.metrics.get_avg_time())

    # plot the ROC curve
    # detection_infrastructure.plotter.plot_new(detection_infrastructure.ids.metrics.get_tprs(),
    # detection_infrastructure.ids.metrics.get_fprs())


def main():
    # Launch a new detection infrastructure instance
    detection_infrastructure = DetectionInfrastructure()

    # test the infrastructure on the test set
    launch_on_testset(detection_infrastructure)

    return 0


if __name__ == '__main__':
    main()
