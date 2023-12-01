import logging
import os

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger('HypertunerLoader')
LOG_FORMAT = '%(asctime)-10s %(levelname)-10s %(name)-45s %(funcName)-35s %(lineno)-5d: %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

class Loader:
    def __init__(self, s3_resource):
        self.bucket_name = 'nsl-kdd-datasets'
        self.s3_resource = s3_resource

    def s3_load(self):
        LOGGER.info(f'Loading data from S3 bucket {self.bucket_name}.')

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

        LOGGER.info('Loading fully processed train sets.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/PCAEncoded',
            file_name='KDDTrain+_l1_pca.pkl',
            download_path='AWS Downloads/Datasets/'
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/PCAEncoded',
            file_name='KDDTrain+_l2_pca.pkl',
            download_path='AWS Downloads/Datasets/'
        )

        LOGGER.info('Loading target variables for train sets.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/ScaledEncoded_no_pca',
            file_name='KDDTrain+_l1_targets.npy',
            download_path='AWS Downloads/Datasets/'
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/ScaledEncoded_no_pca',
            file_name='KDDTrain+_l2_targets.npy',
            download_path='AWS Downloads/Datasets/'
        )

        LOGGER.info('Loading fully processed validation sets.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/PCAEncoded',
            file_name='KDDValidate+_l1_pca.pkl',
            download_path='AWS Downloads/Datasets/'
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/PCAEncoded',
            file_name='KDDValidate+_l2_pca.pkl',
            download_path='AWS Downloads/Datasets/'
        )

        LOGGER.info('Loading target variables for validation sets.')
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/ScaledEncoded_no_pca',
            file_name='KDDValidate+_l1_targets.npy',
            download_path='AWS Downloads/Datasets/'
        )
        self.__aws_download(
            bucket_name=self.bucket_name,
            folder_name='ProcessedDatasets/ScaledEncoded_no_pca',
            file_name='KDDValidate+_l2_targets.npy',
            download_path='AWS Downloads/Datasets/'
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
    def __aws_download_callback(bytes):
        LOGGER.info(f'Downloaded {bytes} bytes')
        pass

    @staticmethod
    def load_models(model1, model2):
        model1 = joblib.load(f'AWS Downloads/Models/StartingModels/{model1}')
        model2 = joblib.load(f'AWS Downloads/Models/StartingModels/{model2}')
        return model1, model2

    @staticmethod
    def load_dataset(pca_file, targets_file):
        x = joblib.load(f'AWS Downloads/Datasets/{pca_file}')
        x_df = pd.DataFrame(x, columns=[f'feature_{i}' for i in range(x.shape[1])])
        y = np.load(f'AWS Downloads/Datasets/{targets_file}', allow_pickle=True)
        return x_df, y
