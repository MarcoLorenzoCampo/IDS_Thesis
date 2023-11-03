import copy
import pickle
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import DataPreprocessingComponent
from KnowledgeBase import KnowledgeBase
from Metrics import Metrics


class DetectionSystem:

    def __init__(self, kb: KnowledgeBase):
        """
        This is the initialization function for the class responsible for setting up the classifiers and
        process data to make it ready for analysis.
        Data is loaded when the class is initiated, then updated when necessary, calling the function
        update_files(.)
        """

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

    def classify(self, incoming_data) -> list[int, float]:
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

        # Start with layer1 (random forest)
        unprocessed_sample = copy.deepcopy(incoming_data)
        sample = (DataPreprocessingComponent.data_process(unprocessed_sample, self.kb.scaler1, self.kb.ohe1,
                                                          self.kb.pca1, self.kb.features_l1, self.kb.cat_features))

        start = time.time()
        anomaly_confidence = self.layer1.predict_proba(sample)[0][1]
        end = time.time()
        computation_time = end - start

        benign_confidence = 1 - anomaly_confidence

        if anomaly_confidence >= self.ANOMALY_THRESHOLD1:
            # it's an anomaly for layer1
            self.metrics.add_classification_time(computation_time)
            return [1, anomaly_confidence]
        else:
            if benign_confidence >= self.BENIGN_THRESHOLD:
                self.metrics.add_classification_time(computation_time)
                return [3, benign_confidence]

        # Continue with layer 2 if layer 1 does not detect anomalies
        sample = (DataPreprocessingComponent.data_process(unprocessed_sample, self.kb.scaler2, self.kb.ohe2,
                                                          self.kb.pca2, self.kb.features_l2, self.kb.cat_features))

        start = time.time()
        anomaly_confidence = self.layer2.decision_function(sample)
        end = time.time()
        computation_time = end - start

        benign_confidence = 1 - anomaly_confidence
        if anomaly_confidence >= self.ANOMALY_THRESHOLD2:
            # it's an anomaly for layer2
            self.metrics.add_classification_time(computation_time)
            return [2, anomaly_confidence]
        else:
            if benign_confidence >= self.BENIGN_THRESHOLD:
                self.metrics.add_classification_time(computation_time)
                return [3, benign_confidence]

        # should not return here
        return [-1111, -1111]

    def train_accuracy(self, layer1, layer2):
        """
        Function to see how the IDS performs on training data, useful to see if overfitting happens
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
            if output[0] == 0:
                self.add_to_quarantine(sample)
                print(f'Prediction: QUARANTINE, AnomalyScore: {output[1]}')
            # it's an anomaly signaled by l1
            elif output[0] == 1:
                self.add_to_anomaly1(sample)
                print(f'Prediction: L1_ANOMALY, AnomalyScore: {output[1]}')
            # it's an anomaly signaled by l2
            elif output[0] == 2:
                self.add_to_anomaly1(sample)
                print(f'Prediction: L2_ANOMALY, AnomalyScore: {output[1]}')
            # it's not an anomaly
            elif output[0] == 3:
                self.add_to_normal(sample)
                print(f'Prediction: NORMAL, AnomalyScore: {output[1]}')

            return

        # in testing only
        if output[0] == 0:
            self.add_to_quarantine(sample)
            print(f'Prediction: QUARANTINE, AnomalyScore: {output[1]}, actual: {actual}')
        # it's an anomaly signaled by l1
        elif output[0] == 1:
            self.add_to_anomaly1(sample)
            print(f'Prediction: L1_ANOMALY, AnomalyScore: {output[1]}, actual: {actual}')
        # it's an anomaly signaled by l2
        elif output[0] == 2:
            self.add_to_anomaly1(sample)
            print(f'Prediction: L2_ANOMALY, AnomalyScore: {output[1]}, actual: {actual}')
        # it's not an anomaly
        elif output[0] == 3:
            self.add_to_normal(sample)
            print(f'Prediction: NORMAL, AnomalyScore: {output[1]}, actual: {actual}')

        # it's an anomaly and has been correctly labeled
        if output[0] in [1, 2] and actual == 1:
            self.metrics.update('tp', 1)

        # it's an anomaly, and it has been labeled as normal traffic
        if output[0] == 3 and actual == 1:
            self.metrics.update('fn', 1)

        # it's normal traffic and has been correctly labeled as so
        if output[0] == 3 and actual == 0:
            self.metrics.update('tn', 1)

        # it's been labeled as an anomaly, but it's actually normal traffic
        if output[0] in [1, 2] and actual == 0:
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
