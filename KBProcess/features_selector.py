import json
import os

import pandas as pd

from KBProcess.storage import Storage
from Shared import utils

LOGGER = utils.get_logger(os.path.splitext(os.path.basename(__file__))[0])

class DataManager:

    def __init__(self, storage: Storage):
        pass

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

    def analyze_datasets(self, x_train: pd.DataFrame):
        LOGGER.info('Analyzing datasets.')

        old_dataset_properties = json.loads("dataset_properties")

        new_dataset_properties = {
            "features_num": x_train.shape[1],
            "features": x_train.columns.tolist(),
            "train_samples": x_train.shape[0],
        }

        if (old_dataset_properties["features_num"] != new_dataset_properties["features_num"] or
                old_dataset_properties["features"] != new_dataset_properties["features"] or
                old_dataset_properties["train_samples"] != new_dataset_properties["train_samples"]):
            LOGGER.info(f'New dataset properties identified.')
            return True

        return False

    def __split_two_layers(x_train: pd.DataFrame):
        pass
