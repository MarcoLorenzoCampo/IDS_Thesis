import copy
import pickle
import time
import Metrics
from Metrics import Metrics
import psutil

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import DataPreprocessingComponent
from KnowledgeBase import KnowledgeBase
from typing import List, Union


class DetectionSystem:

    def __init__(self, kb: KnowledgeBase):
        """
        This is the initialization function for the class responsible for setting up the classifiers and
        process data to make it ready for analysis.
        Data is loaded when the class is initiated, then updated when necessary, calling the function
        update_files(.)
        """

        # set up an instance-level logger to report on the classification performance
        # self.logger = Metrics.set_logger(__name__)

        # manually set the detection thresholds
        self.ANOMALY_THRESHOLD1, self.ANOMALY_THRESHOLD2, self.BENIGN_THRESHOLD = 0.9, 0.8, 0.6

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

    def train_models(self):
        """
        :return: trained models for layer 1 and 2 respectively
        """

        # Start with training classifier 1
        classifier1 = (RandomForestClassifier(n_estimators=25, criterion='gini')
                       .fit(self.kb.x_train_l1, self.kb.y_train_l1))

        # Now train classifier 2
        classifier2 = (SVC(C=0.1, gamma=0.01, kernel='rbf')
                       .fit(self.kb.x_train_l2, self.kb.y_train_l2))

        # Save models to file
        with open('Models/NSL_l1_classifier.pkl', 'wb') as model_file:
            pickle.dump(classifier1, model_file)
        with open('Models/NSL_l2_classifier.pkl', 'wb') as model_file:
            pickle.dump(classifier2, model_file)

        return classifier1, classifier2

    def classify(self, incoming_data) -> list[Union[int, str]]:
        """Tests the given sample on the given layers.

      Args:
        incoming_data: A NumPy array containing the sample to test.

      Returns:
        A list containing two elements:
          * The first element is an integer indicating whether the sample is an anomaly
            0: unsure, quarantines the sample for further analysis
            1: anomaly signaled by layer1
            2: anomaly signaled by layer2
            3: not an anomaly
          * The second element is a float indicating the anomaly score of the sample.
            A higher score indicates a more likely anomaly.
      """

        # Data process for layer 1
        unprocessed_sample = copy.deepcopy(incoming_data)
        sample = (DataPreprocessingComponent.data_process(unprocessed_sample, self.kb.scaler1, self.kb.ohe1,
                                                          self.kb.pca1, self.kb.features_l1, self.kb.cat_features))

        # evaluate cpu_usage and time before classification
        cpu_usage_before = psutil.cpu_percent(interval=1)
        start = time.time()

        # predict using the classifier for layer 1
        prediction1 = self.layer1.predict(sample)

        # evaluate cpu_usage and time immediately after classification
        cpu_usage_after = psutil.cpu_percent(interval=1)
        computation_time = time.time() - start

        # add cpu_usage and computation_time to metrics
        self.metrics.add_cpu_usage(cpu_usage_after - cpu_usage_before)
        self.metrics.add_classification_time(computation_time)

        if prediction1:
            # it's an anomaly for layer1
            return [1, 'L1_ANOMALY']

        # Continue with layer 2 if layer 1 does not detect anomalies
        sample = (DataPreprocessingComponent.data_process(unprocessed_sample, self.kb.scaler2, self.kb.ohe2,
                                                          self.kb.pca2, self.kb.features_l2, self.kb.cat_features))

        # evaluate cpu_usage and computation_time before classification
        cpu_usage_before = psutil.cpu_percent(interval=1)
        start = time.time()

        anomaly_confidence = self.layer2.decision_function(sample)

        # evaluate cpu_usage immediately after classification
        cpu_usage_after = psutil.cpu_percent(interval=1)
        computation_time = time.time() - start

        # add cpu_usage and computation_time to metrics
        self.metrics.add_cpu_usage(cpu_usage_after - cpu_usage_before)
        self.metrics.add_classification_time(computation_time)

        benign_confidence_2 = 1 - anomaly_confidence
        if anomaly_confidence >= self.ANOMALY_THRESHOLD2:
            # it's an anomaly for layer2
            return [anomaly_confidence, 'L2_ANOMALY']

        if benign_confidence_2 >= self.BENIGN_THRESHOLD:
            return [benign_confidence_2, 'NOT_ANOMALY']

        # has not been classified yet, it's not decided
        return [0, 'QUARANTINE']

    def train_accuracy(self, layer1, layer2):
        """
        Function to see how the IDS performs on training data, useful to see if over fitting happens
        :param layer1: classifier 1
        :param layer2: classifier 2
        """

        l1_prediction = layer1.predict(self.kb.x_train_l1, self.kb.y_train_l1)
        l2_prediction = layer2.predict(self.kb.x_train_l2, self.kb.y_train_l2)

        # Calculate the accuracy score for layer 1.
        l1_accuracy = accuracy_score(self.kb.y_train_l1, l1_prediction)

        # Calculate the accuracy score for layer 2.
        l2_accuracy = accuracy_score(self.kb.y_train_l2, l2_prediction)

        # Print the accuracy scores.
        print("Layer 1 accuracy:", l1_accuracy)
        print("Layer 2 accuracy:", l2_accuracy)

    def add_to_quarantine(self, sample):
        """
        Add an unsure traffic sample to quarantine
        :param sample: incoming traffic
        """
        self.quarantine_samples = pd.concat([self.quarantine_samples, sample], axis=0)

    def add_to_anomaly1(self, sample):
        """
        Add an anomalous sample by layer1 to the list
        :param sample: incoming traffic
        """
        self.anomaly_by_l1 = pd.concat([self.anomaly_by_l1, sample], axis=0)

    def add_to_anomaly2(self, sample):
        """
        Add an anomalous sample by layer2 to the list
        :param sample: incoming traffic
        """
        self.anomaly_by_l2 = pd.concat([self.anomaly_by_l2, sample], axis=0)

    def add_to_normal(self, sample):
        """
        Add an anomalous sample by layer2 to the list
        :param sample: incoming traffic
        """
        self.normal_traffic = pd.concat([self.normal_traffic, sample], axis=0)

    def evaluate_classification(self, sample, output: list[int, float], actual: int):
        """
        This function is used to evaluate whether the prediction made by the IDS itself
        :param actual: optional parameter, it's present in testing but not when running
        :param sample:
        :param output:
        :return:
        """
        if actual is None:
            if output[1] == 'QUARANTINE':
                self.add_to_quarantine(sample)
                print(f'Prediction: {output[1]}, AnomalyScore: {output[0]}')
            # it's an anomaly signaled by l1
            elif output[1] == 'L1_ANOMALY':
                self.add_to_anomaly1(sample)
                print(f'Prediction: {output[1]}, AnomalyScore: {output[0]}')
            # it's an anomaly signaled by l2
            elif output[1] == 'L2_ANOMALY':
                self.add_to_anomaly1(sample)
                print(f'Prediction: {output[1]}, AnomalyScore: {output[0]}')
            # it's not an anomaly
            elif output[1] == 'NOT_ANOMALY':
                self.add_to_normal(sample)
                print(f'Prediction: {output[1]}, AnomalyScore: {output[0]}')

            return

        # in testing only
        if output[1] == 'QUARANTINE':
            self.add_to_quarantine(sample)
            print(f'Prediction: {output[1]}, AnomalyScore: {output[0]}, actual: {actual}')
        # it's an anomaly signaled by l1
        elif output[1] == 'L1_ANOMALY':
            self.add_to_anomaly1(sample)
            print(f'Prediction: {output[1]}, AnomalyScore: {output[0]}, actual: {actual}')
        # it's an anomaly signaled by l2
        elif output[1] == 'l2_ANOMALY':
            self.add_to_anomaly1(sample)
            print(f'Prediction: {output[1]}, AnomalyScore: {output[0]}, actual: {actual}')
        # it's not an anomaly
        elif output[1] == 'NOT_ANOMALY':
            self.add_to_normal(sample)
            print(f'Prediction: {output[1]}, AnomalyScore: {output[0]}, actual: {actual}')

        # it's an anomaly and has been correctly labeled
        if (output[1] == 'L1_ANOMALY' or output[1] == 'L2_ANOMALY') and actual == 1:
            self.metrics.update('tp', 1)

        # it's an anomaly, and it has been labeled as normal traffic
        if output[1] == 'NOT_ANOMALY' and actual == 1:
            self.metrics.update('fn', 1)

        # it's normal traffic and has been correctly labeled as so
        if output[1] == 'NOT_ANOMALY' and actual == 0:
            self.metrics.update('tn', 1)

        # it's been labeled as an anomaly, but it's actually normal traffic
        if (output[1] == 'L1_ANOMALY' or output[1] == 'L2_ANOMALY') and actual == 0:
            self.metrics.update('fp', 1)

        return

    """ List of all getters from this class """

    def kb(self) -> KnowledgeBase:
        return self.kb

    def thresholds(self) -> list[float]:
        return [self.ANOMALY_THRESHOLD1, self.ANOMALY_THRESHOLD2, self.BENIGN_THRESHOLD]

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
