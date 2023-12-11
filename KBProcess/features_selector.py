import os

import pandas as pd

from Shared import utils

LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])

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

def analyze_datasets(self):
    LOGGER.info('Analyzing datasets.')
    pass

def __split_two_layers(x_train: pd.DataFrame):
    pass
