import copy
import logging
import time

import boto3
import joblib

import Metrics
from Metrics import Metrics

import pandas as pd
from sklearn.metrics import accuracy_score

import DataProcessor
from typing import Union
from Loader import Loader

LOGGER = logging.getLogger('DetectionSystem')
LOG_FORMAT = '%(levelname) -10s %(name) -45s %(funcName) -35s %(lineno) -5d: %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER.info('Creating an instance of DetectionSystem.')


class DetectionSystem:

    def __init__(self):

        self.ANOMALY_THRESHOLD1, self.ANOMALY_THRESHOLD2, self.BENIGN_THRESHOLD = 0.9, 0.8, 0.6
        self.cat_features = ['flag', 'protocol_type', 'service']

        # Load necessary data from AWS s3
        self.__s3_setup_and_load()

        self.__load_data_instances()

        # set the metrics from the class Metrics
        self.metrics = Metrics()

        # set up the dataframes containing the analyzed data
        self.quarantine_samples = pd.DataFrame(columns=None)
        self.anomaly_by_l1 = pd.DataFrame(columns=None)
        self.anomaly_by_l2 = pd.DataFrame(columns=None)
        self.normal_traffic = pd.DataFrame(columns=None)

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

    def __s3_setup_and_load(self):
        self.s3_resource = boto3.client('s3')
        self.loader = Loader(s3_resource=self.s3_resource)

        LOGGER.info('Loading models.')
        self.loader.s3_load()

        LOGGER.info('Loading from S3 bucket complete.')

    def __load_data_instances(self):

        LOGGER.info('Loading one hot encoders.')
        self.ohe1, self.ohe2 = self.loader.load_encoders('OneHotEncoder_l1.pkl', 'OneHotEncoder_l2.pkl')

        LOGGER.info('Loading scalers.')
        self.scaler1, self.scaler2 = self.loader.load_scalers('Scaler_l1.pkl', 'Scaler_l2.pkl')

        LOGGER.info('Loading pca transformers.')
        self.pca1, self.pca2 = self.loader.load_pca_transformers('layer1_pca_transformer.pkl',
                                                                 'layer2_pca_transformer.pkl')

        LOGGER.info('Loading models.')
        self.layer1, self.layer2 = self.loader.load_models('NSL_l1_classifier.pkl',
                                                           'NSL_l2_classifier.pkl')

        LOGGER.info('Loading minimal features.')
        self.features_l1 = self.loader.load_features('Required Files/NSL_features_l1.txt')
        self.features_l2 = self.loader.load_features('Required Files/NSL_features_l2.txt')

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
            if anomaly_confidence[0, 1] >= self.ANOMALY_THRESHOLD2:
                self.__finalize_clf(incoming_data, [anomaly_confidence, 'L2_ANOMALY'], actual)
                return

            # not an anomaly identified by layer2
            if benign_confidence_2 >= self.BENIGN_THRESHOLD:
                self.__finalize_clf(incoming_data, [benign_confidence_2, 'NOT_ANOMALY2'], actual)
                return

        # has not been classified yet, it's not decided
        self.__finalize_clf(incoming_data, [0, 'QUARANTINE'], actual)

    def __clf_layer1(self, unprocessed_sample):
        sample = (DataProcessor.data_process(unprocessed_sample, self.scaler1, self.ohe1,
                                             self.pca1, self.features_l1, self.cat_features))

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
        sample = (DataProcessor.data_process(unprocessed_sample, self.scaler2, self.ohe2,
                                             self.pca2, self.features_l2, self.cat_features))

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
        Add a normal sample to the correspondant list
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

    def reset(self):
        # reset the output storages
        self.quarantine_samples = pd.DataFrame(columns=None)
        self.anomaly_by_l1 = pd.DataFrame(columns=None)
        self.anomaly_by_l2 = pd.DataFrame(columns=None)
        self.normal_traffic = pd.DataFrame(columns=None)

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


det = DetectionSystem()
