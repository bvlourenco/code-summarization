from datetime import datetime
import logging
import os


def configure_logger(logger):
    '''
    Configures the logger to write to both file and console.

    Args:
        logger: The logger to be configured
    '''
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%d/%m/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if os.path.exists("../results/log.txt"):
        mode = 'a'
    else:
        mode = 'w'
    logfile = logging.FileHandler("../results/log_" +
                                  datetime.now().strftime("%Y-%m-%d_%H:%M:%S") +
                                  ".txt", mode)
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
