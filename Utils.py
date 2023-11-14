import configparser
import logging


def set_logger(name):
    # Create a custom logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(name)

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler(f'Logs/{name}.log')  # File handler
    c_handler.setLevel(logging.WARNING)  # Set level for console handler
    f_handler.setLevel(logging.INFO)  # Set level for file handler

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')  # Console format
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # File format

    c_handler.setFormatter(c_format)  # Set format for console handler
    f_handler.setFormatter(f_format)  # Set format for file handler

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def config_parser():
    config = configparser.ConfigParser()

    config['DEFAULT'] = {'ANOMALY_THRESHOLD1': '0.9',
                         'ANOMALY_THRESHOLD2': '0.8',
                         'BENIGN_THRESHOLD': '0.6',
                         }
