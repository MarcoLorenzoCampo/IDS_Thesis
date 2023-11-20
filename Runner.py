import time
import numpy as np
import pandas as pd
import Utils
import threading

from CustomExceptions import quarantineRatioException, fnrException, tnrException, fprException, tprException, \
    f1Exception, precisionException, accuracyException
from Infrastructure import DetectionInfrastructure


class DetectionSystemLauncher:
    def __init__(self, infrastructure: DetectionInfrastructure):
        self.detection_infrastructure = infrastructure
        self.logger = Utils.set_logger(__name__)
        self.infrastructure = infrastructure

        # load the test sets as simulated traffic samples
        self.x_test = pd.read_csv('NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
        self.y_test = np.load('NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)

        # clean the output files before a new execution
        with open('Required Files/Results.txt', 'w') as f:
            pass

        # pause event for the threads
        self.pause_classification_event = threading.Event()

        # thread used to actually run the ids
        self.ids_thread = threading.Thread(target=self.read_packet, args=(self.pause_classification_event,))

    def read_packet(self, pause_event: threading.Event, iterations=100):

        for i, (index, row) in enumerate(self.x_test.iterrows()):

            # wait for the event to be unpaused (set)
            pause_event.wait()

            # portion the input for clarity
            time.sleep(0)

            if i >= iterations:
                break

            sample = pd.DataFrame(data=np.array([row]), index=None, columns=self.x_test.columns)
            actual = self.y_test[index]
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

        # start the classification thread
        self.ids_thread.start()

        while True:
            try:
                # start the metrics monitor thread and watch for exceptions
                self.infrastructure.ids.metrics.monitor_metrics()
                time.sleep(3)

            except accuracyException as e:
                self.logger.exception(f"Caught accuracyException: {e}")
                self.pause_classification_event.clear()    # pause the classification thread

            except precisionException as e:
                self.logger.exception(f"Caught precisionException: {e}")
                self.pause_classification_event.clear()

            except f1Exception as e:
                self.logger.exception(f"Caught f1Exception: {e}")
                self.pause_classification_event.clear()

            except tprException as e:
                self.logger.exception(f"Caught tprException: {e}")
                self.pause_classification_event.clear()

            except fprException as e:
                self.logger.exception(f"Caught fprException: {e}")
                self.pause_classification_event.clear()

            except tnrException as e:
                self.logger.exception(f"Caught tnrException: {e}")
                self.pause_classification_event.clear()

            except fnrException as e:
                self.logger.exception(f"Caught fnrException: {e}")
                self.pause_classification_event.clear()

            except quarantineRatioException as e:
                self.logger.exception(f"Caught quarantineRatioException: {e}")
                self.pause_classification_event.clear()

            self.pause_classification_event.set()   # resume the classification thread

        '''
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
        '''

        return


if __name__ == '__main__':
    detection_infrastructure = DetectionInfrastructure()
    launcher = DetectionSystemLauncher(detection_infrastructure)
    launcher.run()
