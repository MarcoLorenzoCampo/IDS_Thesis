import logging


def set_logger(name):
    # Create a custom logger
    logger = logging.getLogger(name)

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    f_handler = logging.FileHandler(f'Logs/{name}.log')  # File handler
    c_handler.setLevel(logging.WARNING)  # Set level for console handler
    f_handler.setLevel(logging.ERROR)  # Set level for file handler

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')  # Console format
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # File format

    c_handler.setFormatter(c_format)  # Set format for console handler
    f_handler.setFormatter(f_format)  # Set format for file handler

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
