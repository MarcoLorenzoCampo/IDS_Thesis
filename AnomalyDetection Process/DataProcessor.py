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
