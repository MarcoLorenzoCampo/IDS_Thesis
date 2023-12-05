import argparse
import copy
import logging
import re

import pandas as pd
import LoggerConfig

LOGGER = logging.getLogger('DataProcessor')
logging.basicConfig(level=logging.INFO, format=LoggerConfig.LOG_FORMAT)
LOGGER.info('Creating an instance of DetectionSystem.')

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

def parse_message_body(message):
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
