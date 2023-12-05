import argparse
import copy
import json
import logging
import os
import re

import pandas as pd

from KBProcess import LoggerConfig

logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
filename = os.path.splitext(os.path.basename(__file__))[0]
LOGGER = logging.getLogger(filename)

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

def parse_update_msg(message):
    LOGGER.info('Received messages: %s', message)

    pattern = re.compile(r'^UPDATE\s+([^,]+(?:,\s*[^,]+)*)\s*$', re.IGNORECASE)

    # Example usage
    input_string = message
    match = pattern.match(input_string)

    if match:
        columns = match.group(1).split(', ')
        LOGGER.info(f"Attributes to update: {columns}")
        return columns
    else:
        LOGGER.error("Message body does not match the required syntax. Discarding it.")

def parse_metrics_msg(json_string):
    try:
        parsed_data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

    try:
        metrics_1 = parsed_data["metrics1"]
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


def process_command_line_args():
    parser = argparse.ArgumentParser(description='Process command line arguments for a Python script.')

    parser.add_argument('-metrics_snapshot_timer',
                        type=float,
                        default=10,
                        help='Specify the metrics snapshot timer (float)'
                        )
    parser.add_argument('-polling_timer',
                        type=float,
                        default=5,
                        help='Specify the polling timer (float)'
                        )
    parser.add_argument('-classification_delay',
                        type=float,
                        default=1,
                        help='Specify the classification delay (float)'
                        )

    args = parser.parse_args()

    # Access the arguments using dot notation
    metrics_snapshot_timer = args.metrics_snapshot_timer
    polling_timer = args.polling_timer
    classification_delay = args.classification_delay

    # You can check if the arguments are provided and then use them in your script
    if metrics_snapshot_timer is not None:
        LOGGER.info(f'Metrics Snapshot Timer: {metrics_snapshot_timer}')

    if polling_timer is not None:
        LOGGER.info(f'Polling Timer: {polling_timer}')

    if classification_delay is not None:
        LOGGER.info(f'Classification Delay: {classification_delay}')

    return metrics_snapshot_timer, polling_timer, classification_delay
