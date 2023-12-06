from datetime import datetime, timedelta
import json
import logging
import os
import sqlite3
import time
import boto3

import pandas as pd

from S3Downloader import Loader
from AnomalyDetectionProcess import SQSWrapper
import FeaturesSelector

import LoggerConfig

logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
filename = os.path.splitext(os.path.basename(__file__))[0]
LOGGER = logging.getLogger(filename)


class KnowledgeBase:
    FULL_CLOSE = False

    def __init__(self):

        LOGGER.info('Creating an instance of KnowledgeBase.')

        self.__s3_setup()
        self.__load_data_instances()
        self.__sqlite3_setup()
        self.__sqs_setup()

    def __sqs_setup(self):
        self.sqs_resource = boto3.resource('sqs')
        self.connector = SQSWrapper.Connector(
            sqs_resource=self.sqs_resource,
            queue_names=['tuner-update.fifo', 'detection-system-update.fifo']
        )

    def terminate(self):
        self.connector.close()

    def __sqlite3_setup(self):
        LOGGER.info('Connecting to sqlite3 in memory database.')
        self.sql_connection = sqlite3.connect(':memory:')
        self.cursor = self.sql_connection.cursor()

        LOGGER.info('Instantiating the needed SQL in memory tables.')
        self.__fill_tables()

        LOGGER.info('Removing local instances.')
        self.__clean()

        LOGGER.info('Completed sqlite3 in memory databases setup.')

    def __s3_setup(self):
        self.bucket_name = 'nsl-kdd-datasets'
        self.s3_resource = boto3.client('s3')
        self.loader = Loader(s3_resource=self.s3_resource, bucket_name=self.bucket_name)

        if self.__s3_files_ok() and not self.__need_s3_update():
            LOGGER.info('S3 is already setup and loaded.')
            return

        self.__s3_load()

    def __s3_load(self):
        LOGGER.info(f'Loading data from S3 bucket {self.bucket_name}.')

        self.loader.s3_processed_validation_sets()
        self.loader.s3_processed_train_sets()
        self.loader.s3_models()
        self.loader.s3_scalers()
        self.loader.s3_pca_encoders()
        self.loader.s3_one_hot_encoders()
        self.loader.s3_min_features()

        LOGGER.info('Loading from S3 bucket complete.')

    def __s3_files_ok(self):
        l = self.loader
        return (
            l.check_train_encoded() and
            l.check_train_original() and
            l.check_validation_encoded() and
            l.check_features() and
            l.check_models() and
            l.check_encoders() and
            l.check_pca_encoders() and
            l.check_scalers()
        )

    def __load_data_instances(self):
        LOGGER.info('Loading original train set.')
        self.x_train = self.loader.load_og_dataset('KDDTrain+_with_labels.txt')
        LOGGER.info('Loading original reduced train set.')
        self.x_train_20p = self.loader.load_og_dataset('KDDTrain+20_percent_with_labels.txt')

        LOGGER.info('Loading train sets.')
        self.x_train_l1, self.y_train_l1 = self.loader.load_dataset(
            'KDDTrain+_l1_pca.pkl',
            'KDDTrain+_l1_targets.npy'
        )
        self.x_train_l2, self.y_train_l2 = self.loader.load_dataset(
            'KDDTrain+_l2_pca.pkl',
            'KDDTrain+_l2_targets.npy'
        )

        LOGGER.info('Loading validation sets.')
        self.x_validate_l1, self.y_validate_l1 = self.loader.load_dataset(
            'KDDValidate+_l1_pca.pkl',
            'KDDValidate+_l1_targets.npy'
        )
        self.x_validate_l2, self.y_validate_l2 = self.loader.load_dataset(
            'KDDValidate+_l2_pca.pkl',
            'KDDValidate+_l2_targets.npy'
        )

        LOGGER.info('Loading scalers.')
        self.scaler1, self.scaler2 = self.loader.load_scalers('Scaler_l1.pkl', 'Scaler_l2.pkl')

        LOGGER.info('Loading one hot encoders.')
        self.ohe1, self.ohe2 = self.loader.load_encoders('OneHotEncoder_l1.pkl', 'OneHotEncoder_l2.pkl')

        LOGGER.info('Loading pca transformers.')
        self.pca1, self.pca2 = self.loader.load_pca_transformers('layer1_pca_transformer.pkl',
                                                                 'layer2_pca_transformer.pkl')

        LOGGER.info('Loading models.')
        self.pca1, self.pca2 = self.loader.load_models('NSL_l1_classifier.pkl',
                                                       'NSL_l2_classifier.pkl')

        LOGGER.info('Loading minimal features.')
        self.features_l1 = self.loader.load_features('NSL_features_l1.txt')
        self.features_l2 = self.loader.load_features('NSL_features_l2.txt')

    def __fill_tables(self):
        # create a table for each train set
        self.x_train_l1.to_sql('x_train_l1', self.sql_connection, index=False, if_exists='replace')
        self.x_train_l2.to_sql('x_train_l2', self.sql_connection, index=False, if_exists='replace')

        # create a table for each train set
        self.x_train.to_sql('x_train', self.sql_connection, index=False, if_exists='replace')
        self.x_train_20p.to_sql('x_train_20p', self.sql_connection, index=False, if_exists='replace')

        # create a table for each validation set
        self.x_validate_l1.to_sql('x_validate_l1', self.sql_connection, index=False, if_exists='replace')
        self.x_validate_l2.to_sql('x_validate_l2', self.sql_connection, index=False, if_exists='replace')

        # now append target variables as the last column of each table
        self.__append_to_table('x_train_l1', 'target', self.y_train_l1)
        self.__append_to_table('x_train_l2', 'target', self.y_train_l2)
        self.__append_to_table('x_validate_l1', 'target', self.y_validate_l1)
        self.__append_to_table('x_validate_l2', 'target', self.y_validate_l2)

    def __append_to_table(self, table_name, column_name, target_values):
        # Fetch the existing table from the in-memory database
        existing_data = pd.read_sql_query(f'SELECT * FROM {table_name}', self.sql_connection)
        # Append the target column to the existing table
        existing_data[column_name] = target_values
        # Update the table in the in-memory database
        existing_data.to_sql(table_name, self.sql_connection, if_exists='replace', index=False)

    def __clean(self):
        # Remove instances of datasets to free up memory
        del self.x_train_l1, self.x_train_l2, self.y_train_l1, self.y_train_l2
        del self.x_validate_l1, self.x_validate_l2, self.y_validate_l1, self.y_validate_l2
        # del self.x_test, self.y_test
        del self.x_train, self.x_train_20p

    def perform_query(self, received):
        LOGGER.info(f'Received a query.')

        try:
            select_clause = received.get("select")
            from_clause = received.get("from")
            where_clause = received.get("where")

            if where_clause is None:
                sql_query = f'SELECT {select_clause} FROM {from_clause}'
            else:
                sql_query = f'SELECT {select_clause} FROM {from_clause} WHERE {where_clause}'

            LOGGER.info(f'Executing the query: {sql_query}')

            result_df = pd.read_sql_query(sql_query, self.sql_connection)

        except pd.errors.DatabaseError:
            LOGGER.exception('Could not fulfill the requests.')
            return None

        LOGGER.info('Query was executed correctly.')
        return result_df

    def __select_features_procedure(self, feature_selection_func):

        query_dict = {
            "select": "*",
            "from": "x_train"
        }
        if feature_selection_func(self.perform_query(query_dict)):

            update_msg = 'UPDATED FEATURES'
            self.connector.send_message_to_queues(update_msg, None)

        else:
            LOGGER.error('Feature selection function failed. Retry.')

    @staticmethod
    def __need_s3_update():
        try:
            with open('last_online.txt', 'r') as last_online_file:
                last_online_str = last_online_file.read().strip()

            last_online = datetime.strptime(last_online_str, "%Y-%m-%d %H:%M:%S")

            time_difference = datetime.now() - last_online

            return time_difference > timedelta(hours=3)
        except FileNotFoundError:
            return False

    def __test(self):
        test_dict = {
            "UPDATE": ["FEATURES", "TRAIN", "VALIDATE"]
        }
        msg_body = json.dumps(test_dict)
        self.connector.send_message_to_queues(msg_body, None)

    def run_tasks(self):

        action_mapping = {
            1: FeaturesSelector.perform_icfs,
            2: FeaturesSelector.perform_sfs,
            3: FeaturesSelector.perform_bfs,
            4: FeaturesSelector.perform_fisher
        }

        LOGGER.info('Running background tasks..')
        time.sleep(2)

        try:
            while True:

                print("\nSelect the number of the action to perform:"
                      "\n1. ICFS feature selection"
                      "\n2. SFS feature selection"
                      "\n3. BFS feature selection"
                      "\n4. Fisher feature selection"
                      "\n5. Analyze the dataset for changes"
                      "\n'exit' to quit to program.")

                choice = input('>> ')

                if choice.lower() == 'exit':
                    break

                try:
                    action_number = int(choice)
                    selected_function = action_mapping.get(action_number)
                    if selected_function:
                        self.__select_features_procedure(selected_function)
                    else:
                        print("Invalid action number. Please enter a number between 1 and 4.")
                        continue

                except ValueError:
                    print("Invalid input. Please enter a valid number.")
                    continue

                # self.__test()
                time.sleep(1.5)

        except KeyboardInterrupt:
            raise KeyboardInterrupt('Keyboard interrupt. Terminating knowledge base instance.')


if __name__ == '__main__':
    kb = KnowledgeBase()

    try:
        kb.run_tasks()
    except KeyboardInterrupt:
        if kb.FULL_CLOSE:
            kb.terminate()
            LOGGER.info('Deleting queues..')

        LOGGER.info('Saving last online timestamp.')
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('last_online.txt', 'w') as file:
            file.write(current_timestamp)

        LOGGER.info('Received keyboard interrupt. Terminating knowledge base instance.')
