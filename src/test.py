from datetime import datetime
import logging
import os
import torch
from args.parse_args import parse_test_args
from train_test.test_program import TestProgram

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')


def configure_logger():
    '''
    Configures the logger to write to both file and console.
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


def main():
    '''
    Main function of the program to test.
    Parses the arguments given to the program, loads the trained model and its
    vocabulary and then tests it using the testing set.
    '''
    args = parse_test_args()

    configure_logger()

    torch.manual_seed(0)
    program = TestProgram(args)
    program.start()


if __name__ == '__main__':
    main()
