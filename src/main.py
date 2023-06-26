import torch
import logging
from args.parse_args import parse_arguments
from finetune.fine_tuning import fine_tune
from logger.logger import configure_logger
from train_test.train_program import TrainProgram

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')


def main():
    '''
    Main function of the program to train and validate.
    Parses the arguments given to the program and depending on the input argument
    `hyperparameter_tuning`, it either trains the model or performs fine-tuning on
    the parameters of the model.
    '''
    args = parse_arguments()

    configure_logger(logger)

    torch.manual_seed(0)
    if args.hyperparameter_tuning:
        fine_tune(args)
    else:
        program = TrainProgram(args)
        program.start()


if __name__ == '__main__':
    main()
