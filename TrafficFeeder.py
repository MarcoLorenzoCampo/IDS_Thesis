import time
import numpy as np
import pandas as pd
import Utils
from Infrastructure import DetectionInfrastructure


class DetectionSystemLauncher:
    def __init__(self, infrastructure: DetectionInfrastructure):
        self.detection_infrastructure = infrastructure
        self.logger = Utils.set_logger(__name__)

    def launch_on_test(self, iterations=100):
        x_test = pd.read_csv('NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
        y_test = np.load('NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)

        for i, (index, row) in enumerate(x_test.iterrows()):
            if i >= iterations:
                break

            sample = pd.DataFrame(data=np.array([row]), index=None, columns=x_test.columns)
            actual = y_test[index]
            self.detection_infrastructure.ids.classify(sample, actual=actual)

            if i - 1 in list(range(iterations))[::50]:
                self.logger.info('\nAt iterations #%s:', i)
                self.detection_infrastructure.ids.metrics.show_metrics()

        self.logger.info('Classification of %s test samples:', iterations)
        self.logger.info('Anomalies by l1: %s', self.detection_infrastructure.ids.anomaly_by_l1.shape[0])
        self.logger.info('Anomalies by l2: %s', self.detection_infrastructure.ids.anomaly_by_l2.shape[0])
        self.logger.info('Normal traffic: %s', self.detection_infrastructure.ids.normal_traffic.shape[0])
        self.logger.info('Quarantined samples: %s', self.detection_infrastructure.ids.quarantine_samples.shape[0])

        with open('Required Files/Results.txt', 'a') as file:
            file.write('\nOverall classification:\n')
            file.write('Classified = ANOMALY, Actual = ANOMALY: tp -> ' + str(
                self.detection_infrastructure.ids.metrics.get_counts('tp')) + '\n')
            file.write('Classified = ANOMALY, Actual = NORMAL: fp -> ' + str(
                self.detection_infrastructure.ids.metrics.get_counts('fp')) + '\n')
            file.write('Classified = NORMAL, Actual = ANOMALY: fn -> ' + str(
                self.detection_infrastructure.ids.metrics.get_counts('fn')) + '\n')
            file.write('Classified = NORMAL, Actual = NORMAL: tn -> ' + str(
                self.detection_infrastructure.ids.metrics.get_counts('tn')) + '\n\n')

        self.logger.info('Average computation time for %s samples: %s', iterations,
                         self.detection_infrastructure.ids.metrics.get_avg_time())

    def artificial_tuning(self):
        self.detection_infrastructure.hp_tuner.tune()

    def run(self):
        with open('Required Files/Results.txt', 'w') as f:
            pass

        self.launch_on_test()

        self.detection_infrastructure.ids.train_accuracy()

        test_acc = self.detection_infrastructure.ids.metrics.get_metrics('accuracy')
        with open('Required Files/Results.txt', 'a') as f:
            f.write('BEFORE TUNING FOR FALSE POSITIVES FP:\n')
            f.write('\nSystem accuracy on the train set: ' + str(test_acc))

        self.detection_infrastructure.ids.reset()
        self.detection_infrastructure.ids.metrics.reset()

        self.artificial_tuning()

        self.launch_on_test()

        self.detection_infrastructure.ids.train_accuracy()

        test_acc = self.detection_infrastructure.ids.metrics.get_metrics('accuracy')
        val_acc1 = self.detection_infrastructure.hp_tuner.best_acc1
        val_acc2 = self.detection_infrastructure.hp_tuner.best_acc2
        with open('Required Files/Results.txt', 'a') as f:
            f.write('\n\nAFTER TUNING FOR FALSE POSITIVES FP:\n')
            f.write('\nSystem accuracy on the train set: ' + str(test_acc))
            f.write('\nSystem accuracy on the validation set: ' + str(val_acc1) + ', ' + str(val_acc2))

        return 0


if __name__ == '__main__':
    detection_infrastructure = DetectionInfrastructure()
    launcher = DetectionSystemLauncher(detection_infrastructure)
    launcher.run()
