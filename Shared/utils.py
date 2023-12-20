import copy
import pprint
import time
import os
import pandas as pd
import logging
import colorlog

from datetime import datetime, timedelta

from Shared import msg_enum


def get_logger(name):
    logger = colorlog.getLogger(name)

    # Set the logger level to the lowest level you want to capture
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers to avoid duplication
    logger.handlers = []

    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)-15s %(levelname)-10s %(name)-40s %(funcName)-35s %(lineno)-5d:%(reset)s %(message)s',
        log_colors={
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger


LOGGER = get_logger(os.path.splitext(os.path.basename(__file__))[0])

def pprint_to_file(path: str, file_content: str):
    LOGGER.info("Writing metrics to file.")

    with open(path, "a") as log_file:
        log_file.write("\n\n")
        pprint.pprint(file_content, log_file)


def data_process(incoming_data, scaler, ohe, pca, features, cat_features):
    data = copy.deepcopy(incoming_data)
    to_scale = data[features]

    scaled = scaler.transform(to_scale)
    scaled_data = pd.DataFrame(scaled, columns=to_scale.columns)
    label_enc = ohe.transform(data[cat_features])
    label_enc.toarray()
    new_labels = ohe.get_feature_names_out(cat_features)
    new_encoded = pd.DataFrame(data=label_enc.toarray(), columns=new_labels)
    processed = pd.concat([scaled_data, new_encoded], axis=1)
    pca_transformed = pca.transform(processed)

    return pca_transformed


def parse_update_msg(json_dict: dict):
    to_update = json_dict["UPDATE"]

    if to_update is not None:
        LOGGER.info(f'Objects to update from S3: {to_update}')
        return to_update

    return None


def parse_metrics_msg(parsed_data: dict):

    try:
        metrics_1 = parsed_data["metrics_1"]
        metrics_2 = parsed_data["metrics_2"]
        classification_metrics = parsed_data["classification_metrics"]

    except KeyError as e:
        raise ValueError(f"Missing key in JSON: {e}")

    try:
        metrics1 = {
            "accuracy": metrics_1["accuracy"],
            "precision": metrics_1["precision"],
            "fscore": metrics_1["fscore"],
            "tpr": metrics_1["tpr"],
            "fpr": metrics_1["fpr"],
            "tnr": metrics_1["tnr"],
            "fnr": metrics_1["fnr"]
        }

    except KeyError as e:
        raise ValueError(f"Missing key in metrics1: {e}")

    try:
        metrics2 = {
            "accuracy": metrics_2["accuracy"],
            "precision": metrics_2["precision"],
            "fscore": metrics_2["fscore"],
            "tpr": metrics_2["tpr"],
            "fpr": metrics_2["fpr"],
            "tnr": metrics_2["tnr"],
            "fnr": metrics_2["fnr"]
        }
    except KeyError as e:
        raise ValueError(f"Missing key in metrics_2: {e}")

    try:
        classification_metrics = {
            "normal_ratio": classification_metrics["normal_ratio"],
            "l1_anomaly_ratio": classification_metrics["l1_anomaly_ratio"],
            "l2_anomaly_ratio": classification_metrics["l2_anomaly_ratio"],
            "quarantined_ratio": classification_metrics["quarantined_ratio"]
        }

    except KeyError as e:
        raise ValueError(f"Missing key in classification_metrics: {e}")

    return metrics1, metrics2, classification_metrics

def need_s3_update():
    try:
        with open('../AnomalyDetectionProcess/last_online.txt', 'r') as last_online_file:
            last_online_str = last_online_file.read().strip()

        last_online = datetime.strptime(last_online_str, "%Y-%m-%d %H:%M:%S")

        time_difference = datetime.now() - last_online

        return time_difference > timedelta(hours=3)
    except FileNotFoundError:
        return False


def save_current_timestamp(path: str):
    LOGGER.info('Saving last online timestamp.')
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path + 'last_online.txt', 'w') as file:
        file.write(current_timestamp)


def parse_objs(data: dict):

    layer1_list = data.get("objs_layer1", [])
    layer2_list = data.get("objs_layer2", [])

    LOGGER.info(f'Parsed objectives. Layer1: {layer1_list}, Layer2: {layer2_list}')

    return {
        "layer1": layer1_list,
        "layer2": layer2_list
    }
