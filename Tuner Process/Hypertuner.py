import logging
import sqlite3
import sys
import time

import boto3
import optuna
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.svm import SVC
from LoaderHypertuner import Loader
import HPTunerConnectionHandler


LOGGER = logging.getLogger('Hypertuner')
LOG_FORMAT = '%(asctime)-10s %(levelname)-10s %(name)-45s %(funcName)-35s %(lineno)-5d: %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER.info('Creating an instance of KnowledgeBase connection handler.')

class Hypertuner:

    def __init__(self, ampq_url: str):

        self.__s3_setup_and_load()
        self.__load_data_instances()
        self.__sqlite3_setup()

        self.connection_handler = HPTunerConnectionHandler.Connector(self, ampq_url)

        # number of trials for tuning
        self.n_trials = 7

        # new optimal models
        self.new_opt_layer1 = None
        self.new_opt_layer2 = None

        # accuracy for over fitting evaluations
        self.val_accuracy_l1 = []
        self.val_accuracy_l2 = []

        # accuracy of the best hyperparameters
        self.best_acc1 = 0
        self.best_acc2 = 0

    def __s3_setup_and_load(self):
        self.s3_resource = boto3.client('s3')
        self.loader = Loader(s3_resource=self.s3_resource)

        LOGGER.info('Loading models.')
        self.loader.s3_load()

        LOGGER.info('Loading from S3 bucket complete.')

    def __load_data_instances(self):

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

        LOGGER.info('Loading models.')
        self.layer1, self.layer2 = self.loader.load_models('NSL_l1_classifier.pkl',
                                                           'NSL_l2_classifier.pkl')

    def __sqlite3_setup(self):
        LOGGER.info('Connecting to sqlite3 in memory database.')
        self.sql_connection = sqlite3.connect(':memory:')
        self.cursor = self.sql_connection.cursor()

        LOGGER.info('Instantiating the needed SQL in memory tables.')
        self.__fill_tables()

        LOGGER.info('Removing local instances.')
        self.__clean()

        LOGGER.info('Completed sqlite3 in memory databases setup.')

    def __fill_tables(self):
        # create a table for each train set
        self.x_train_l1.to_sql('x_train_l1', self.sql_connection, index=False, if_exists='replace')
        self.x_train_l2.to_sql('x_train_l2', self.sql_connection, index=False, if_exists='replace')

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

    def tune(self, to_opt: str, direction: str):
        # build the study object
        study_l1 = optuna.create_study(study_name=f'RandomForest optimization: {to_opt}', direction=direction)
        study_l2 = optuna.create_study(study_name=f'SupportVectorMachine optimization: {to_opt}', direction=direction)

        # if statement to identify what optimization function to call
        if to_opt == 'accuracy':
            LOGGER.info('Optimizing L1 for accuracy.')
            study_l1.optimize(self.__objective_accuracy_l1, n_trials=self.n_trials)
            study_l2.optimize(self.__objective_accuracy_l2, n_trials=self.n_trials)

        if to_opt == 'precision':
            LOGGER.info('Optimizing L1 for precision.')
            study_l1.optimize(self.__objective_precision_l1, n_trials=self.n_trials)
            study_l2.optimize(self.__objective_precision_l2, n_trials=self.n_trials)

        if to_opt == 'fscore':
            LOGGER.info('Optimizing L1 for fscore.')
            study_l1.optimize(self.__objective_fscore_l1, n_trials=self.n_trials)
            study_l2.optimize(self.__objective_fscore_l2, n_trials=self.n_trials)

        if to_opt == 'tpr':
            LOGGER.info('Optimizing L1 for tpr.')
            study_l1.optimize(self.__objective_tpr_l1, n_trials=self.n_trials)
            study_l2.optimize(self.__objective_tpr_l2, n_trials=self.n_trials)

        if to_opt == 'tnr':
            LOGGER.info('Optimizing L1 for tnr.')
            study_l1.optimize(self.__objective_tnr_l1, n_trials=self.n_trials)
            study_l2.optimize(self.__objective_tnr_l2, n_trials=self.n_trials)

        if to_opt == 'fpr':
            LOGGER.info('Optimizing L1 for fpr.')
            study_l1.optimize(self.__objective_fpr_l1, n_trials=self.n_trials)
            study_l2.optimize(self.__objective_fpr_l2, n_trials=self.n_trials)

        if to_opt == 'fnr':
            LOGGER.info('Optimizing L1 for fnr.')
            study_l1.optimize(self.__objective_fnr_l1, n_trials=self.n_trials)
            study_l2.optimize(self.__objective_fnr_l2, n_trials=self.n_trials)

        if to_opt == 'quarantine_ratio':
            LOGGER.info('Optimizing L1 for tpr.')
            study_l1.optimize(self.__objective_quarantine_rate_l1, n_trials=self.n_trials)
            study_l2.optimize(self.__objective_quarantine_rate_l2, n_trials=self.n_trials)

        # obtain the optimal classifiers from the studies
        self.new_opt_layer1 = self.__train_new_hps('RandomForest', study_l1.best_params)
        self.new_opt_layer2 = self.__train_new_hps('SVM', study_l2.best_params)

        # reset the storage variables
        self.reset()

        # return the newly trained models and hyperparameters
        return self.new_opt_layer1, self.new_opt_layer2

    def __objective_tpr_l1(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            tp = sum((self.y_train_l1 == 1) & (predicted == 1))
            fn = sum((self.y_train_l1 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            return tpr

    def __objective_tpr_l2(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            return tpr

    def __objective_fpr_l1(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            fp = sum((self.y_train_l2 == 0) & (predicted == 1))
            tn = sum((self.y_train_l2 == 0) & (predicted == 0))
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

            return fpr

    def __objective_fpr_l2(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            fp = sum((self.y_train_l2 == 0) & (predicted == 1))
            tn = sum((self.y_train_l2 == 0) & (predicted == 0))
            fpr = fp / (fp + tn) if (fp + tn) != 0 else 0.0

            return fpr

    def __objective_tnr_l1(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            tn = sum((self.y_train_l2 == 0) & (predicted == 0))
            fp = sum((self.y_train_l2 == 0) & (predicted == 1))
            tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            return tnr

    def __objective_tnr_l2(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            tn = sum((self.y_train_l2 == 0) & (predicted == 0))
            fp = sum((self.y_train_l2 == 0) & (predicted == 1))
            tnr = tn / (tn + fp) if (tn + fp) != 0 else 0.0

            return tnr

    def __objective_fnr_l1(self, trial: optuna.Trial) -> float:
        """
        This function defines the objective to maximize the true positive rate
        :param trial:
        :return: True positive rate
        """
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0

            return fnr

    def __objective_fnr_l2(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fnr = fn / (fn + tp) if (fn + tp) != 0 else 0.0

            return fnr

    def __objective_accuracy_l1(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            accuracy = accuracy_score(self.y_validate_l1, predicted)
            return accuracy

    def __objective_accuracy_l2(self, trial: optuna.Trial) -> float:
        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            accuracy = accuracy_score(self.y_validate_l2, predicted)
            return accuracy

    def __objective_precision_l1(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)

            return precision

    def __objective_precision_l2(self, trial: optuna.Trial) -> float:
        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            precision = precision_score(self.y_validate_l2, predicted)

            return precision

    def __objective_fscore_l1(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)
            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            fscore = 2*precision*tpr/(precision + tpr)

            return fscore

    def __objective_fscore_l2(self, trial: optuna.Trial) -> float:
        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l2)

            precision = precision_score(self.y_validate_l2, predicted)
            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            fscore = 2 * precision * tpr / (precision + tpr)

            return fscore

    def __objective_quarantine_rate_l1(self, trial: optuna.Trial) -> float:
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest'])
        if classifier_name == 'RandomForest':
            rf_n_estimators = trial.suggest_int(name='n_estimators', low=1, high=19, step=2)
            rf_max_depth = trial.suggest_int(name='max_depth', low=2, high=32, step=1)
            rf_criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
            rf_max_features = trial.suggest_int(name='min_samples_split', low=2, high=10, step=1)

            parameters = {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'criterion': rf_criterion,
                'max_features': rf_max_features
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)
            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            fscore = 2*precision*tpr/(precision + tpr)

            return fscore

    def __objective_quarantine_rate_l2(self, trial: optuna.Trial) -> float:
        # providing a choice of classifiers to use in the 'choices' array
        classifier_name = trial.suggest_categorical('classifier', ['SVC'])
        if classifier_name == 'SVC':
            # list now the hyperparameters that need tuning
            svc_c = trial.suggest_float(name='svc_c', low=1e-10, high=1e10)

            # add to parameters all the hyperparameters that need tuning
            parameters = {
                'C': svc_c
            }

            classifier = self.__train_new_hps(classifier_name, parameters)
            predicted = classifier.predict(self.x_validate_l1)

            precision = precision_score(self.y_validate_l1, predicted)
            tp = sum((self.y_train_l2 == 1) & (predicted == 1))
            fn = sum((self.y_train_l2 == 1) & (predicted == 0))
            tpr = tp / (tp + fn) if (tp + fn) != 0 else 0.0

            fscore = 2 * precision * tpr / (precision + tpr)

            return fscore

    def __train_new_hps(self, classifier_name: str, parameters: dict):

        if classifier_name == 'RandomForest':
            classifier = RandomForestClassifier(
                n_estimators=parameters.get('n_estimators', 10),
                criterion=parameters.get('criterion', 'gini'),
                max_depth=parameters.get('max_depth', None),
                min_samples_split=parameters.get('min_samples_split', 2),
                min_samples_leaf=parameters.get('min_samples_leaf', 1),
                min_weight_fraction_leaf=parameters.get('min_weight_fraction_leaf', 0.0),
                max_features=parameters.get('max_features', 'sqrt'),
                max_leaf_nodes=parameters.get('max_leaf_nodes', None),
                min_impurity_decrease=parameters.get('min_impurity_decrease', 0.0),
                bootstrap=parameters.get('bootstrap', True),
                oob_score=parameters.get('oob_score', False),
                n_jobs=parameters.get('n_jobs', None),
                random_state=parameters.get('random_state', None),
                verbose=0,
                warm_start=parameters.get('warm_start', False),
                class_weight=parameters.get('class_weight', None),
                ccp_alpha=parameters.get('ccp_alpha', 0.0),
                max_samples=parameters.get('max_samples', None)
            )

            LOGGER.info('Trained a new RandomForest classifier.')
            classifier.fit(self.x_train_l1, self.y_train_l1)
            return classifier

        else:
            classifier = SVC(
                C=parameters.get('C', 10),
                kernel=parameters.get('kernel', 'rbf'),
                degree=parameters.get('degree', 3),
                gamma=parameters.get('gamma', 0.01),
                coef0=parameters.get('coef0', 0.0),
                shrinking=parameters.get('shrinking', True),
                probability=True,
                tol=parameters.get('tol', 1e-3),
                cache_size=parameters.get('cache_size', 200),
                class_weight=parameters.get('class_weight', None),
                verbose=False,
                max_iter=parameters.get('max_iter', -1),
                decision_function_shape=parameters.get('decision_function_shape', 'ovr')
            )

            LOGGER.info('Trained a new SVM classifier.')
            classifier.fit(self.x_train_l2, self.y_train_l2)
            return classifier

    def reset(self):
        self.val_accuracy_l1 = []
        self.val_accuracy_l2 = []


class ReconnectingConsumerProducer:
    def __init__(self, amqp_url):
        self._reconnect_delay = 0
        self._amqp_url = amqp_url
        self._consumer = Hypertuner(self._amqp_url)

    def run(self):
        while True:
            try:
                self._consumer.connection_handler.run()
            except KeyboardInterrupt:
                self._consumer.connection_handler.stop()
                break
            self._maybe_reconnect()

    def _maybe_reconnect(self):
        if self._consumer.connection_handler.should_reconnect:
            self._consumer.connection_handler.stop()
            reconnect_delay = self._get_reconnect_delay()
            LOGGER.info('Reconnecting after %d seconds', reconnect_delay)
            time.sleep(reconnect_delay)
            self._consumer = Hypertuner(self._amqp_url)

    def _get_reconnect_delay(self):
        if self._consumer.connection_handler.was_consuming:
            self._reconnect_delay = 0
        else:
            self._reconnect_delay += 1
        if self._reconnect_delay > 30:
            self._reconnect_delay = 30
        return self._reconnect_delay


def main():
    param1 = sys.argv[1] if len(sys.argv) > 1 else None
    param2 = sys.argv[2] if len(sys.argv) > 2 else None
    ampq_url = sys.argv[3] if len(sys.argv) > 3 else "amqp://guest:guest@host:5672/"

    consumer = ReconnectingConsumerProducer(amqp_url=ampq_url)
    consumer.run()


if __name__ == '__main__':
    main()
