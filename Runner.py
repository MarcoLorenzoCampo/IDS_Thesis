import time
import numpy as np
import pandas as pd

import Metrics
import Utils
import threading

from CustomExceptions import (quarantineRatioException, fnrException, tnrException, fprException, tprException,
                              precisionException, accuracyException, fException)
from Infrastructure import DetectionInfrastructure


class DetectionSystemLauncher:
    def __init__(self, infrastructure: DetectionInfrastructure):
        self.detection_infrastructure = infrastructure
        self.logger = Utils.set_logger('Runner')
        self.infrastructure = infrastructure

        # load the test sets as simulated traffic samples
        self.x_test = pd.read_csv('NSL-KDD Encoded Datasets/before_pca/KDDTest+', sep=",", header=0)
        self.y_test = np.load('NSL-KDD Encoded Datasets/before_pca/y_test.npy', allow_pickle=True)

        # clean the output files before a new execution
        with open('Required Files/Results.txt', 'w'):
            pass

        # pause event for the threads
        self.pause_classification_event = threading.Event()
        self.pause_monitor_event = threading.Event()

        # thread used to actually run the ids
        self.ids_thread = threading.Thread(target=self.read_packet, args=(self.pause_classification_event,))
        self.monitor_thread = threading.Thread(target=self.monitor,
                                               args=(self.pause_monitor_event, self.detection_infrastructure.metrics))

    def read_packet(self, pause_event: threading.Event, iterations=None):

        iterations = self.x_test.shape[0]
        for i, (index, row) in enumerate(self.x_test.iterrows()):

            # wait for the event to be unpaused (set)
            pause_event.wait()

            # portion the input for clarity
            # time.sleep(0.15)

            # for debugging purposes
            if i >= iterations:
                break

            sample = pd.DataFrame(data=np.array([row]), index=None, columns=self.x_test.columns)
            actual = self.y_test[index]
            self.detection_infrastructure.ids.classify(sample, actual=actual)

            if i - 1 in list(range(iterations))[::200]:
                self.logger.info('At iterations #%s:', i)
                self.detection_infrastructure.ids.metrics.show_metrics()

        # some classification metrics after the whole set has been processed
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

    def monitor(self, pause_event: threading.Event, metrics: Metrics):
        while True:
            try:
                metrics.analyze_metrics()

            except accuracyException as e:
                # the .clear() event command
                self.pause_classification_event.clear()
                self.logger.exception(f"Caught accuracyException: {e}")
                self.infrastructure.hp_tuner.tune('accuracy', 'maximize')

            except precisionException as e:
                # the .clear() event command
                self.pause_classification_event.clear()
                self.logger.exception(f"Caught precisionException: {e}")
                self.infrastructure.hp_tuner.tune('precision', 'maximize')

            except fException as e:
                # the .clear() event command
                self.pause_classification_event.clear()
                self.logger.exception(f"Caught f1Exception: {e}")
                self.infrastructure.hp_tuner.tune('fscore', 'maximize')

            except tprException as e:
                # the .clear() event command
                self.pause_classification_event.clear()
                self.logger.exception(f"Caught tprException: {e}")
                self.infrastructure.hp_tuner.tune('tpr', 'maximize')

            except fprException as e:
                # the .clear() event command
                self.pause_classification_event.clear()
                self.logger.exception(f"Caught fprException: {e}")
                self.infrastructure.hp_tuner.tune('fpr', 'maximize')

            except tnrException as e:
                # the .clear() event command
                self.pause_classification_event.clear()
                self.logger.exception(f"Caught tnrException: {e}")
                self.infrastructure.hp_tuner.tune('tnr', 'maximize')

            except fnrException as e:
                # the .clear() event command
                self.pause_classification_event.clear()
                self.logger.exception(f"Caught fnrException: {e}")
                self.infrastructure.hp_tuner.tune('fnr', 'maximize')

            except quarantineRatioException as e:
                # the .clear() event command
                self.pause_classification_event.clear()
                self.logger.exception(f"Caught quarantineRatioException: {e}")
                self.infrastructure.hp_tuner.tune('quarantine_ratio', 'minimize')

            # the classification resumes after handling the exception and tuning
            self.pause_classification_event.set()

    def run(self):

        # start the classification thread
        self.ids_thread.start()

        # start the monitor thread
        self.monitor_thread.start()

        # wait for the ids_thread to finish before returning
        self.ids_thread.join()

        self.logger.info('No more input packets (test), closing.')
        return


if __name__ == '__main__':
    detection_infrastructure = DetectionInfrastructure()
    launcher = DetectionSystemLauncher(detection_infrastructure)
    launcher.run()
