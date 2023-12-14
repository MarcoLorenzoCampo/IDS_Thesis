import os
import sqlite3

import boto3
import pandas as pd
from botocore.exceptions import ClientError

from Shared import utils
from Shared.s3_wrapper import Loader


LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])

class Storage:

    def __init__(self):
        self.bucket_name = 'nsl-kdd-datasets'
        self.__s3_setup()
        self.__load_data_instances()
        self.__sqlite3_setup()

    def __s3_setup(self):
        self.bucket_name = 'nsl-kdd-datasets'
        self.s3_resource = boto3.client('s3')
        self.loader = Loader(s3_resource=self.s3_resource, bucket_name=self.bucket_name)

        if self.__s3_files_ok() and not utils.need_s3_update():
            LOGGER.debug('S3 is already setup and loaded.')
            return

        self.__s3_load()

    def __s3_load(self):
        LOGGER.debug(f'Loading data from S3 bucket {self.bucket_name}.')

        self.loader.s3_processed_train_sets()
        self.loader.s3_processed_validation_sets()
        self.loader.s3_models()

        LOGGER.debug('Loading from S3 bucket complete.')

    def __s3_files_ok(self):
        l = self.loader
        return (
            l.check_train_encoded() and
            l.check_validation_encoded() and
            l.check_models()
        )

    def __load_data_instances(self):

        LOGGER.debug('Loading train sets.')
        self.x_train_l1, self.y_train_l1 = self.loader.load_dataset(
            'KDDTrain+_l1_pca.pkl',
            'KDDTrain+_l1_targets.npy'
        )
        self.x_train_l2, self.y_train_l2 = self.loader.load_dataset(
            'KDDTrain+_l2_pca.pkl',
            'KDDTrain+_l2_targets.npy'
        )

        LOGGER.debug('Loading validation sets.')
        self.x_validate_l1, self.y_validate_l1 = self.loader.load_dataset(
            'KDDValidate+_l1_pca.pkl',
            'KDDValidate+_l1_targets.npy'
        )
        self.x_validate_l2, self.y_validate_l2 = self.loader.load_dataset(
            'KDDValidate+_l2_pca.pkl',
            'KDDValidate+_l2_targets.npy'
        )

        LOGGER.debug('Loading models.')
        self.layer1, self.layer2 = self.loader.load_models('NSL_l1_classifier.pkl',
                                                           'NSL_l2_classifier.pkl')

    def __sqlite3_setup(self):
        LOGGER.debug('Connecting to sqlite3 in memory database.')
        self.sql_connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.cursor = self.sql_connection.cursor()

        LOGGER.debug('Instantiating the needed SQL in memory tables.')
        self.__fill_tables()

        LOGGER.debug('Removing local instances.')
        self.__clean()

        LOGGER.debug('Completed sqlite3 in memory databases setup.')

    def __fill_tables(self):
        # create a table for each train set
        self.x_train_l1.to_sql('x_train_l1', self.sql_connection, index=False, if_exists='replace')
        self.x_train_l2.to_sql('x_train_l2', self.sql_connection, index=False, if_exists='replace')

        # create a table for each validation set
        self.x_validate_l1.to_sql('x_validate_l1', self.sql_connection, index=False, if_exists='replace')
        self.x_validate_l2.to_sql('x_validate_l2', self.sql_connection, index=False, if_exists='replace')

        # now append target variables as the last column of each table
        self.__append_to_table('x_train_l1', 'targets', self.y_train_l1)
        self.__append_to_table('x_train_l2', 'targets', self.y_train_l2)
        self.__append_to_table('x_validate_l1', 'targets', self.y_validate_l1)
        self.__append_to_table('x_validate_l2', 'targets', self.y_validate_l2)

    def __append_to_table(self, table_name, column_name, target_values):
        existing_data = pd.read_sql_query(f'SELECT * FROM {table_name}', self.sql_connection)
        existing_data[column_name] = target_values
        existing_data.to_sql(table_name, self.sql_connection, if_exists='replace', index=False)

    def __clean(self):
        # Remove instances of datasets to free up memory
        del self.x_train_l1, self.x_train_l2, self.y_train_l1, self.y_train_l2
        del self.x_validate_l1, self.x_validate_l2, self.y_validate_l1, self.y_validate_l2

    def perform_query(self, received):

        try:
            select_clause = received.get("select")
            from_clause = received.get("from")
            where_clause = received.get("where")

            if where_clause is None:
                sql_query = f'SELECT {select_clause} FROM {from_clause}'
            else:
                sql_query = f'SELECT {select_clause} FROM {from_clause} WHERE {where_clause}'

            LOGGER.debug(f'Executing the query: {sql_query}')

            result_df = pd.read_sql_query(sql_query, self.sql_connection)

        except pd.errors.DatabaseError:
            LOGGER.exception('Could not fulfill the requests.')
            return None

        LOGGER.debug('Query was executed correctly.')
        return result_df

    def publish_s3_models(self):

        LOGGER.debug(f'Updating models in the S3 bucket {self.bucket_name}.')

        classifier1_path = 'AWS Downloads/Models/Tuned/NSL_l1_classifier.pkl'
        classifier2_path = 'AWS Downloads/Models/Tuned/NSL_l2_classifier.pkl'

        try:
            self.s3_resource.upload_file(
                classifier1_path,
                self.bucket_name,
                self.bucket_name+'/Models/Tuned/NSL_l1_classifier.pkl'
            )

            self.s3_resource.upload_file(
                classifier2_path,
                self.bucket_name,
                self.bucket_name+'/Models/Tuned/NSL_l2_classifier.pkl'
            )

        except ClientError as e:
            LOGGER.error(f'Error when uploading file: {e}')
            return False

        return True
