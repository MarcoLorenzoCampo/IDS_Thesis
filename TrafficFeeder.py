import numpy as np
import pandas as pd

from Infrastructure import DetectionInfrastructure


def launch_on_testset(detection_infrastructure: DetectionInfrastructure):
    # Load a testing dataset
    x_test = pd.read_csv('NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
    y_test = np.load('NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)

    iterations = 1000

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

    with open('NSL-KDD Files/Results.txt', 'a') as file:
        # Write the strings to the file
        file.write('\nOverall classification:\n')
        file.write('Classified = ANOMALY, Actual = ANOMALY: tp -> ' + str(
            detection_infrastructure.ids.metrics.get_counts('tp')) + '\n')
        file.write('Classified = ANOMALY, Actual = NORMAL: fp -> ' + str(
            detection_infrastructure.ids.metrics.get_counts('fp')) + '\n')
        file.write('Classified = NORMAL, Actual = ANOMALY: fn -> ' + str(
            detection_infrastructure.ids.metrics.get_counts('fn')) + '\n')
        file.write('Classified = NORMAL, Actual = NORMAL: tn -> ' + str(
            detection_infrastructure.ids.metrics.get_counts('tn')) + '\n\n')

    # print the average of the computation time
    print(f'Average computation time for {iterations} samples: ', detection_infrastructure.ids.metrics.get_avg_time())

    # plot the ROC curve
    # detection_infrastructure.plotter.plot_new(detection_infrastructure.ids.metrics.get_tprs(),
    # detection_infrastructure.ids.metrics.get_fprs())


def artificial_tuning(detection_infrastructure: DetectionInfrastructure):
    detection_infrastructure.hp_tuning.tune()


def main():
    # empty the result.txt file before writing on it
    with open('NSL-KDD Files/Results.txt', 'w') as f:
        pass

    # Launch a new detection infrastructure instance
    with open('NSL-KDD Files/Results.txt', 'a') as f:
        f.write('BEFORE TUNING FOR FALSE POSITIVES FP:\n')
    detection_infrastructure = DetectionInfrastructure()

    # test the infrastructure on the test set
    launch_on_testset(detection_infrastructure)

    # evaluate the precision on the train set to see over fitting
    detection_infrastructure.ids.train_accuracy()

    # reset the variables used to store classification data
    detection_infrastructure.ids.reset()
    detection_infrastructure.ids.metrics.reset()

    # test if and how the tuning procedure works
    artificial_tuning(detection_infrastructure)

    # test if the new iterations produce better results
    with open('NSL-KDD Files/Results.txt', 'a') as f:
        f.write('\n\nAFTER TUNING FOR FALSE POSITIVES FP:\n')
    launch_on_testset(detection_infrastructure)

    # evaluate the precision on the train set to see over fitting
    detection_infrastructure.ids.train_accuracy()

    return 0


if __name__ == '__main__':
    main()
