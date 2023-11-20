import copy
import time
import threading

import Metrics
from Metrics import Metrics
import Utils

import pandas as pd
from sklearn.metrics import accuracy_score

import DataProcessor
from KnowledgeBase import KnowledgeBase
from typing import Union


class DetectionSystem:

    def __init__(self, kb: KnowledgeBase):

        # set up an instance-level logger to report on the classification performance
        self.logger = Utils.set_logger(__name__)
        self.logger.debug('Launching the DetectionSystem.')

        # set new knowledge base
        self.kb = kb

        # set the metrics from the class Metrics
        self.metrics = Metrics()

        # set up the dataframes containing the analyzed data
        self.quarantine_samples = pd.DataFrame(columns=self.kb.x_test.columns)
        self.anomaly_by_l1 = pd.DataFrame(columns=self.kb.x_test.columns)
        self.anomaly_by_l2 = pd.DataFrame(columns=self.kb.x_test.columns)
        self.normal_traffic = pd.DataFrame(columns=self.kb.x_test.columns)

        # set the classifiers
        self.layer1, self.layer2 = self.kb.layer1, self.kb.layer2

        # dictionary for classification functions
        self.clf_switcher = {
            'QUARANTINE': self.__add_to_quarantine,
            'L1_ANOMALY': self.__add_to_anomaly1,
            'L2_ANOMALY': self.__add_to_anomaly2,
            'NOT_ANOMALY1': self.__add_to_normal1,
            'NOT_ANOMALY2': self.__add_to_normal2
        }

        # dictionary for metrics functions
        self.metrics_switcher = {
            ('NOT_ANOMALY1', 1): lambda: self.metrics.update_count('fn', 1, 1),
            ('NOT_ANOMALY1', 0): lambda: self.metrics.update_count('tn', 1, 1),
            ('NOT_ANOMALY2', 1): lambda: self.metrics.update_count('fn', 1, 2),
            ('NOT_ANOMALY2', 0): lambda: self.metrics.update_count('tn', 1, 2),
            ('L1_ANOMALY', 0): lambda: self.metrics.update_count('fp', 1, 1),
            ('L1_ANOMALY', 1): lambda: self.metrics.update_count('tp', 1, 1),
            ('L2_ANOMALY', 0): lambda: self.metrics.update_count('fp', 1, 2),
            ('L2_ANOMALY', 1): lambda: self.metrics.update_count('tp', 1, 2),
        }

    def classify(self, incoming_data, actual: int = None):
        """
      Args:
        incoming_data: A NumPy array containing the sample to test.
        actual: Optional argument containing the label the incoming traffic data, if available
      """

        # Copy of the original sample
        unprocessed_sample = copy.deepcopy(incoming_data)

        # Classification for layer 1
        prediction1, computation_time, cpu_usage = self.__clf_layer1(unprocessed_sample)

        # add cpu_usage and computation_time to metrics
        self.metrics.add_cpu_usage(cpu_usage)
        self.metrics.add_classification_time(computation_time)

        if prediction1:
            # anomaly identified by layer1
            self.__finalize_clf(incoming_data, [1, 'L1_ANOMALY'], actual)
            return

        else:
            # considered as normal by layer1
            self.__finalize_clf(incoming_data, [0, 'NOT_ANOMALY1'], actual)

            # Continue with layer 2 if layer 1 does not detect anomalies
            anomaly_confidence, computation_time, cpu_usage = self.__clf_layer2(unprocessed_sample)

            # add cpu_usage and computation_time to metrics
            self.metrics.add_cpu_usage(cpu_usage)
            self.metrics.add_classification_time(computation_time)

            benign_confidence_2 = 1 - anomaly_confidence[0, 1]

            # anomaly identified by layer2
            if anomaly_confidence[0, 1] >= self.kb.ANOMALY_THRESHOLD2:
                self.__finalize_clf(incoming_data, [anomaly_confidence, 'L2_ANOMALY'], actual)
                return

            # not an anomaly identified by layer2
            if benign_confidence_2 >= self.kb.BENIGN_THRESHOLD:
                self.__finalize_clf(incoming_data, [benign_confidence_2, 'NOT_ANOMALY2'], actual)
                return

        # has not been classified yet, it's not decided
        self.__finalize_clf(incoming_data, [0, 'QUARANTINE'], actual)

    def __clf_layer1(self, unprocessed_sample):
        sample = (DataProcessor.data_process(unprocessed_sample, self.kb.scaler1, self.kb.ohe1,
                                             self.kb.pca1, self.kb.features_l1, self.kb.cat_features))

        # evaluate cpu_usage and time before classification
        # cpu_usage_before = psutil.cpu_percent(interval=1)
        start = time.time()

        # predict using the classifier for layer 1
        prediction1 = self.layer1.predict(sample)

        # evaluate cpu_usage and time immediately after classification
        # cpu_usage_after = psutil.cpu_percent(interval=1)

        # metrics to return
        cpu_usage = 0
        # cpu_usage = cpu_usage_after - cpu_usage_before
        computation_time = time.time() - start

        return prediction1, computation_time, cpu_usage

    def __clf_layer2(self, unprocessed_sample):
        sample = (DataProcessor.data_process(unprocessed_sample, self.kb.scaler2, self.kb.ohe2,
                                             self.kb.pca2, self.kb.features_l2, self.kb.cat_features))

        # evaluate cpu_usage and time before classification
        # cpu_usage_before = psutil.cpu_percent(interval=1)
        start = time.time()

        # we are interested in anomaly_confidence[0, 1], meaning the first sample and class 1 (the anomaly class)
        anomaly_confidence = self.layer2.predict_proba(sample)

        # evaluate cpu_usage and time immediately after classification
        # cpu_usage_after = psutil.cpu_percent(interval=1)

        # metrics to return
        cpu_usage = 0
        # cpu_usage = cpu_usage_after - cpu_usage_before
        computation_time = time.time() - start

        return anomaly_confidence, computation_time, cpu_usage

    def __finalize_clf(self, sample: pd.DataFrame, output: list[Union[int, str]], actual: int = None):
        """
        This function is used to evaluate whether the prediction made by the IDS itself is correct or not
        and acts accordingly
        :param actual: optional parameter, it's None in real application but not in testing
        :param sample: Traffic data to store
        :param output: What to classify
        :return:
        """
        if actual is None:
            switch_function = self.clf_switcher.get(output[1], lambda: "Invalid value")
            switch_function(sample)

        if actual is not None:
            switch_function = self.clf_switcher.get(output[1], lambda: "Invalid value")
            switch_function(sample)

        switch_function = self.metrics_switcher.get((output[1], actual), lambda: "Invalid value")
        switch_function()

    def __add_to_quarantine(self, sample: pd.DataFrame) -> None:
        """
        Add an unsure traffic sample to quarantine
        :param sample: incoming traffic
        """
        self.quarantine_samples = pd.concat([self.quarantine_samples, sample], axis=0)
        self.metrics.update_classifications('quarantine', 1)

    def __add_to_anomaly1(self, sample: pd.DataFrame) -> None:
        """
        Add an anomalous sample by layer1 to the list
        :param sample: incoming traffic
        """
        self.anomaly_by_l1 = pd.concat([self.anomaly_by_l1, sample], axis=0)
        self.metrics.update_classifications(tag='l1_anomaly', value=1)

    def __add_to_anomaly2(self, sample: pd.DataFrame) -> None:
        """
        Add an anomalous sample by layer2 to the list
        :param sample: incoming traffic
        """
        self.anomaly_by_l2 = pd.concat([self.anomaly_by_l2, sample], axis=0)
        self.metrics.update_classifications('l2_anomaly', 1)

    def __add_to_normal1(self, sample: pd.DataFrame) -> None:
        """
        Add a normal sample to the list
        :param sample: incoming traffic
        """
        self.normal_traffic = pd.concat([self.normal_traffic, sample], axis=0)
        self.metrics.update_classifications('normal_traffic', 1)

    def __add_to_normal2(self, sample: pd.DataFrame) -> None:
        """
        Add a normal sample to the list
        :param sample: incoming traffic
        """
        self.normal_traffic = pd.concat([self.normal_traffic, sample], axis=0)
        self.metrics.update_classifications('normal_traffic', 1)

    def train_accuracy(self) -> list[float, float]:
        """
        Function to see how the IDS performs on training data, useful to see if over fitting happens
        """

        l1_prediction = self.layer1.predict(self.kb.x_train_l1)
        l2_prediction = self.layer2.predict(self.kb.x_train_l2)

        # Calculate the accuracy score for layer 1.
        l1_accuracy = accuracy_score(self.kb.y_train_l1, l1_prediction)

        # Calculate the accuracy score for layer 2.
        l2_accuracy = accuracy_score(self.kb.y_train_l2, l2_prediction)

        # write accuracy scores to file
        with open('Required Files/Results.txt', 'a') as f:
            f.write("\nLayer 1 accuracy on the train set:" + str(l1_accuracy))
            f.write("\nLayer 2 accuracy on the train set:" + str(l2_accuracy))

        return [l1_accuracy, l2_accuracy]

    def reset(self):
        # reset the output storages
        self.quarantine_samples = pd.DataFrame(columns=self.kb.x_test.columns)
        self.anomaly_by_l1 = pd.DataFrame(columns=self.kb.x_test.columns)
        self.anomaly_by_l2 = pd.DataFrame(columns=self.kb.x_test.columns)
        self.normal_traffic = pd.DataFrame(columns=self.kb.x_test.columns)

    def metrics(self) -> Metrics:
        return self.metrics

    def quarantine(self) -> pd.DataFrame:
        return self.quarantine_samples

    def anomaly_l1(self) -> pd.DataFrame:
        return self.anomaly_by_l1

    def anomaly_l2(self) -> pd.DataFrame:
        return self.anomaly_by_l2

    def normal(self) -> pd.DataFrame:
        return self.normal_traffic
