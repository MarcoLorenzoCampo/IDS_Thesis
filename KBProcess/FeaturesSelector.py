import logging

import pandas as pd

import LoggerConfig


logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
LOGGER = logging.getLogger('FeaturesSelector')

def perform_icfs(x_train: pd.DataFrame):
    LOGGER.info('Performing ICFS.')

    return 1

def perform_fisher(x_train: pd.DataFrame):
    LOGGER.info('Performing fisher score.')

    return 1

def perform_bfs(x_train: pd.DataFrame):
    LOGGER.info('Performing BFS.')

    return 1

def perform_sfs(x_train: pd.DataFrame):
    LOGGER.info('Performing SFS.')

    return 1

def __split_two_layers(x_train: pd.DataFrame):
    pass
