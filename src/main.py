import os
import torch
import logging
from args.parse_args import parse_arguments
from finetune.fine_tuning import fine_tune
from train_test.train_program import TrainProgram

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
    logfile = logging.FileHandler("../results/log.txt", mode)
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)


def main():
    '''
    Main function of the program to train and validate.
    Parses the arguments given to the program and depending on the input argument
    `hyperparameter_tuning`, it either trains the model or performs fine-tuning on
    the parameters of the model.
    '''
    args = parse_arguments()

    configure_logger()

    torch.manual_seed(0)
    if args.hyperparameter_tuning:
        fine_tune(args)
    else:
        program = TrainProgram(args)
        program.start()


if __name__ == '__main__':
    main()
