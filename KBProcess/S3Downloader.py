import logging
import os

import joblib
import numpy as np
import pandas as pd

from KBProcess import LoggerConfig

LOGGER = logging.getLogger('KBLoader')
logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)

class Loader:
    def __init__(self, s3_resource, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_resource = s3_resource

    def s3_original_sets(self):
        LOGGER.info('Loading original data sets.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='OriginalDatasets',
            file_name='KDDTrain+_with_labels.txt',
            download_path='AWS Downloads/Datasets/OriginalDatasets/'
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='OriginalDatasets',
            file_name='KDDTrain+20_percent_with_labels.txt',
            download_path='AWS Downloads/Datasets/OriginalDatasets/'
        )

    def s3_original_test_set(self):
        LOGGER.info('Loading the test set.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='OriginalDatasets',
            file_name='KDDTest+.txt',
            download_path="AWS Downloads/Datasets/OriginalDatasets/"
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='OriginalDatasets',
            file_name='KDDTest+_targets.npy',
            download_path="AWS Downloads/Datasets/OriginalDatasets/"
        )

    def s3_min_features(self):
        LOGGER.info('Loading set of minimal features.')
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

    def s3_one_hot_encoders(self):
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

    def s3_pca_encoders(self):
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

    def s3_scalers(self):
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

    def s3_models(self):
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

    def s3_processed_train_sets(self):
        LOGGER.info('Loading fully processed train sets.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/PCAEncoded',
            file_name='KDDTrain+_l1_pca.pkl',
            download_path='AWS Downloads/Datasets/PCAEncoded/'
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/PCAEncoded',
            file_name='KDDTrain+_l2_pca.pkl',
            download_path='AWS Downloads/Datasets/PCAEncoded/'
        )
        LOGGER.info('Loading target variables for train sets.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/ScaledEncoded_no_pca',
            file_name='KDDTrain+_l1_targets.npy',
            download_path='AWS Downloads/Datasets/PCAEncoded/'
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/ScaledEncoded_no_pca',
            file_name='KDDTrain+_l2_targets.npy',
            download_path='AWS Downloads/Datasets/PCAEncoded/'
        )

    def s3_processed_validation_sets(self):
        LOGGER.info('Loading fully processed validation sets.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/PCAEncoded',
            file_name='KDDValidate+_l1_pca.pkl',
            download_path='AWS Downloads/Datasets/PCAEncoded/'
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/PCAEncoded',
            file_name='KDDValidate+_l2_pca.pkl',
            download_path='AWS Downloads/Datasets/PCAEncoded/'
        )
        LOGGER.info('Loading target variables for validation sets.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/ScaledEncoded_no_pca',
            file_name='KDDValidate+_l1_targets.npy',
            download_path='AWS Downloads/Datasets/PCAEncoded/'
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/ScaledEncoded_no_pca',
            file_name='KDDValidate+_l2_targets.npy',
            download_path='AWS Downloads/Datasets/PCAEncoded/'
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
    def load_test_set():
        x_test = pd.read_csv('AWS Downloads/Datasets/OriginalDatasets/KDDTest+.txt', sep=",", header=0)
        y_test = np.load('AWS Downloads/Datasets/OriginalDatasets/KDDTest+_targets.npy', allow_pickle=True)
        return x_test, y_test

    @staticmethod
    def load_og_dataset(file):
        x_df = pd.read_csv(f'AWS Downloads/Datasets/OriginalDatasets/{file}')
        return x_df

    @staticmethod
    def load_dataset(pca_file, targets_file):
        x = joblib.load(f'AWS Downloads/Datasets/PCAEncoded/{pca_file}')
        x_df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
        y = np.load(f'AWS Downloads/Datasets/PCAEncoded/{targets_file}', allow_pickle=True)
        return x_df, y

    @staticmethod
    def load_features(file_name: str):
        with open('AWS Downloads/MinimalFeatures/'+file_name, 'r') as f:
            return f.read().split(',')
