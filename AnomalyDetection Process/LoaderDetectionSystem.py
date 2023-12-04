import logging
import os

import joblib
import numpy as np
import pandas as pd

import LoggerConfig

LOGGER = logging.getLogger('KnowledgeBase')
logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)

class Loader:
    def __init__(self, s3_resource):
        self.bucket_name = 'nsl-kdd-datasets'
        self.s3_resource = s3_resource

    def s3_load(self):
        LOGGER.info(f'Loading data from S3 bucket {self.bucket_name}.')

        LOGGER.info('Loading minimal features.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='AdditionalFiles/MinimalFeatures',
            file_name='NSL_features_l1.txt',
            download_path="AWS Downloads/MinimalFeatures/"
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='AdditionalFiles/MinimalFeatures',
            file_name='NSL_features_l2.txt',
            download_path="AWS Downloads/MinimalFeatures/"
        )

        LOGGER.info('Loading set of one hot encoders.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='AdditionalFiles/OneHotEncoders',
            file_name='OneHotEncoder_l1.pkl',
            download_path="AWS Downloads/OneHotEncoders/"
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='AdditionalFiles/OneHotEncoders',
            file_name='OneHotEncoder_l2.pkl',
            download_path="AWS Downloads/OneHotEncoders/"
        )

        LOGGER.info('Loading set of PCA encoders.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='AdditionalFiles/PCAEncoders',
            file_name='layer1_pca_transformer.pkl',
            download_path="AWS Downloads/PCAEncoders/"
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='AdditionalFiles/PCAEncoders',
            file_name='layer2_pca_transformer.pkl',
            download_path="AWS Downloads/PCAEncoders/"
        )

        LOGGER.info('Loading set of scalers.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='AdditionalFiles/Scalers',
            file_name='Scaler_l1.pkl',
            download_path="AWS Downloads/Scalers/"
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='AdditionalFiles/Scalers',
            file_name='Scaler_l2.pkl',
            download_path="AWS Downloads/Scalers/"
        )

        LOGGER.info('Loading models.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='Models/StartingModels',
            file_name='NSL_l1_classifier.pkl',
            download_path="AWS Downloads/Models/StartingModels/"
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='Models/StartingModels',
            file_name='NSL_l2_classifier.pkl',
            download_path="AWS Downloads/Models/StartingModels/"
        )

        LOGGER.info('Loading the test set.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='OriginalDatasets',
            file_name='KDDTest+.txt',
            download_path="AWS Downloads/Test Set/"
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='OriginalDatasets',
            file_name='KDDTest+_targets.npy',
            download_path="AWS Downloads/Test Set/"
        )

    def __aws_download(self, bucket_name: str, folder_name: str, file_name: str, download_path: str):
        local_file_path = os.path.join(download_path, file_name)
        self.s3_resource.download_file(
            bucket_name,
            f'{folder_name}/{file_name}',
            local_file_path,
            Callback=self.__aws_download_callback
        )

    @staticmethod
    def __aws_download_callback(downloaded_bytes):
        LOGGER.info(f'Downloaded {downloaded_bytes} bytes')
        pass

    @staticmethod
    def load_pca_transformers(pca1_file, pca2_file):
        pca1 = joblib.load(f'AWS Downloads/PCAEncoders/{pca1_file}')
        pca2 = joblib.load(f'AWS Downloads/PCAEncoders/{pca2_file}')
        return pca1, pca2

    @staticmethod
    def load_testset(test_set, targets):
        x_test = pd.read_csv(f'AWS Downloads/Test Set/{test_set}', sep=",", header=0)
        y_test = np.load(f'AWS Downloads/Test Set/{targets}', allow_pickle=True)
        return x_test, y_test

    @staticmethod
    def load_models(model1, model2):
        model1 = joblib.load(f'AWS Downloads/Models/StartingModels/{model1}')
        model2 = joblib.load(f'AWS Downloads/Models/StartingModels/{model2}')
        return model1, model2

    @staticmethod
    def load_encoders(ohe1_file, ohe2_file):
        ohe1 = joblib.load(f'AWS Downloads/OneHotEncoders/{ohe1_file}')
        ohe2 = joblib.load(f'AWS Downloads/OneHotEncoders/{ohe2_file}')
        return ohe1, ohe2

    @staticmethod
    def load_scalers(scaler1_file, scaler2_file):
        scaler1 = joblib.load(f'AWS Downloads/Scalers/{scaler1_file}')
        scaler2 = joblib.load(f'AWS Downloads/Scalers/{scaler2_file}')
        return scaler1, scaler2

    @staticmethod
    def load_features(file_name: str):
        path = 'AWS Downloads/MinimalFeatures/'+file_name
        with open(path, 'r') as f:
            return f.read().split(',')
