import os
import sqlite3

import boto3
import pandas as pd

from botocore.exceptions import ClientError
from Shared import utils
from Shared.s3_wrapper import Loader


class S3Manager:

    def __init__(self, bucket_name: str):

        import hypertuner_main
        self.LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        self.bucket_name = bucket_name
        self.__s3_setup()

    def __s3_setup(self):
        self.s3_resource = boto3.client('s3')
        self.loader = Loader(s3_resource=self.s3_resource, bucket_name=self.bucket_name)

        if self.__s3_files_ok() and not utils.need_s3_update('TunerProcess/'):
            self.LOGGER.debug('S3 is already setup and loaded.')
            return

        self.__s3_load()

    def __s3_load(self):
        self.LOGGER.debug(f'Loading data from S3 bucket {self.bucket_name}.')

        self.loader.s3_processed_train_sets('ProcessedDatasets',
                                            '../TunerProcess/AWS Downloads/Datasets')
        self.loader.s3_processed_validation_sets('ProcessedDatasets',
                                                 '../TunerProcess/AWS Downloads/Datasets')
        self.loader.s3_models('Models/ModelsToUse',
                              '../TunerProcess/AWS Downloads/Models/ModelsToUse')

        self.LOGGER.debug('Loading from S3 bucket complete.')

    def __s3_files_ok(self):
        loader = self.loader
        return (
            loader.check_train_encoded() and
            loader.check_validation_encoded() and
            loader.check_models()
        )

    def get_prepared_loader(self):
        return self.loader
    
    def publish_s3_models(self):

        self.LOGGER.debug(f'Updating models in the S3 bucket {self.bucket_name}.')

        classifier1_path = 'TunedModels/l1_classifier.pkl'
        classifier2_path = 'TunedModels/l2_classifier.pkl'

        try:
            if os.path.isfile(classifier1_path):
                self.s3_resource.upload_file(
                    classifier1_path,
                    self.bucket_name,
                    self.bucket_name + '/Models/TunedModels/l1_classifier.pkl'
                )
            else:
                self.LOGGER.warning(f'File {classifier1_path} does not exist locally and will not be uploaded.')

            if os.path.isfile(classifier2_path):
                self.s3_resource.upload_file(
                    classifier2_path,
                    self.bucket_name,
                    self.bucket_name + '/Models/TunedModels/l2_classifier.pkl'
                )
            else:
                self.LOGGER.warning(f'File {classifier2_path} does not exist locally and will not be uploaded.')

        except ClientError as e:
            self.LOGGER.error(f'Error when uploading file: {e}')
            return False

        return True


class Storage:

    def __init__(self, s3_manager: S3Manager):

        import hypertuner_main
        self.LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        self.s3_manager = s3_manager
        self.loader = self.s3_manager.get_prepared_loader()
        self.__load_data_in_disk()

    def __load_data_in_disk(self):

        self.LOGGER.debug('Loading train sets.')
        self.x_train_l1, self.y_train_l1 = self.loader.load_dataset(
            'KDDTrain+_l1_pca.txt',
            'KDDTrain+_l1_targets.npy'
        )
        self.x_train_l2, self.y_train_l2 = self.loader.load_dataset(
            'KDDTrain+_l2_pca.txt',
            'KDDTrain+_l2_targets.npy'
        )
        self.LOGGER.debug('Loading validation sets.')
        self.x_validate_l1, self.y_validate_l1 = self.loader.load_dataset(
            'KDDValidate+_l1_pca.txt',
            'KDDValidate+_l1_targets.npy'
        )
        self.x_validate_l2, self.y_validate_l2 = self.loader.load_dataset(
            'KDDValidate+_l2_pca.txt',
            'KDDValidate+_l2_targets.npy'
        )
        self.LOGGER.debug('Loading models.')
        self.layer1, self.layer2 = self.loader.load_models('l1_classifier.pkl',
                                                           'l2_classifier.pkl')
        print(self.x_validate_l2.shape, self.y_validate_l2.shape)

class SQLiteManager:

    def __init__(self, storage: Storage):

        import hypertuner_main
        self.LOGGER = hypertuner_main.LOGGER.getChild(os.path.splitext(os.path.basename(__file__))[0])

        self.storage = storage
        self.__sqlite3_setup()

    def __sqlite3_setup(self):
        self.LOGGER.debug('Connecting to sqlite3 in memory database.')
        self.sql_connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.cursor = self.sql_connection.cursor()

        self.LOGGER.debug('Instantiating the needed SQL in memory tables.')
        self.__move_data_in_memory()

        self.LOGGER.debug('Removing local instances.')
        self.__clean()

        self.LOGGER.debug('Completed sqlite3 in memory databases setup.')

    def __move_data_in_memory(self):
        # create a table for each train set
        self.storage.x_train_l1.to_sql('x_train_l1', self.sql_connection, index=False, if_exists='replace')
        self.storage.x_train_l2.to_sql('x_train_l2', self.sql_connection, index=False, if_exists='replace')

        # create a table for each validation set
        self.storage.x_validate_l1.to_sql('x_validate_l1', self.sql_connection, index=False, if_exists='replace')
        self.storage.x_validate_l2.to_sql('x_validate_l2', self.sql_connection, index=False, if_exists='replace')

        # now append target variables as the last column of each table
        self.__append_to_table('x_train_l1', 'targets', self.storage.y_train_l1)
        self.__append_to_table('x_train_l2', 'targets', self.storage.y_train_l2)
        self.__append_to_table('x_validate_l1', 'targets', self.storage.y_validate_l1)
        self.__append_to_table('x_validate_l2', 'targets', self.storage.y_validate_l2)

    def __append_to_table(self, table_name, column_name, target_values):
        existing_data = pd.read_sql_query(f'SELECT * FROM {table_name}', self.sql_connection)
        existing_data[column_name] = target_values
        existing_data.to_sql(table_name, self.sql_connection, if_exists='replace', index=False)

    def __clean(self):
        # Remove instances of datasets to free up memory
        del self.storage.x_train_l1, self.storage.x_train_l2, self.storage.y_train_l1, self.storage.y_train_l2
        del self.storage.x_validate_l1, self.storage.x_validate_l2, self.storage.y_validate_l1, self.storage.y_validate_l2

    def perform_query(self, received):

        try:
            select_clause = received.get("select")
            from_clause = received.get("from")
            where_clause = received.get("where")

            if where_clause is None:
                sql_query = f'SELECT {select_clause} FROM {from_clause}'
            else:
                sql_query = f'SELECT {select_clause} FROM {from_clause} WHERE {where_clause}'

            self.LOGGER.debug(f'Executing the query: {sql_query}')

            result_df = pd.read_sql_query(sql_query, self.sql_connection)

        except pd.errors.DatabaseError:
            self.LOGGER.exception('Could not fulfill the requests.')
            return None

        self.LOGGER.debug('Query was executed correctly.')
        return result_df
