import logging
import torch
from args.parse_args import parse_test_args
from logger.logger import configure_logger
from train_test.test_program import TestProgram

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')


def main():
    '''
    Main function of the program to test.
    Parses the arguments given to the program, loads the trained model and its
    vocabulary and then tests it using the testing set.
    '''
    args = parse_test_args()

    configure_logger(logger, args.dir_iteration)

    torch.manual_seed(0)
    program = TestProgram(args)
    program.start()


if __name__ == '__main__':
    main()
