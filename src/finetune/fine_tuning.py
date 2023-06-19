import json
import logging
import sys
import optuna
import torch

from program import run_program

optuna.logging.get_logger("optuna").addHandler(
    logging.StreamHandler(sys.stdout))


def objective(trial, args):
    '''
    Objective function used by optuna to fine-tune the parameters of a model.

    Args:
        trial: An object used by optuna library to manage the fine-tuning.
        args: The arguments given as input to the program.
    '''
    args.max_src_length = trial.suggest_int('max_src_length', 150, 151)
    args.freq_threshold = trial.suggest_int('freq_threshold', 0, 10)
    args.dropout = trial.suggest_float('dropout', 0.0, 1.0)
    args.learning_rate = trial.suggest_float(
        'learning_rate', 1e-5, 1e-1, log=True)
    args.gradient_clipping = trial.suggest_float('gradient_clipping', 0.0, 2.0)
    args.label_smoothing = trial.suggest_float('label_smoothing', 0.0, 1.0)

    print("Arguments: ", json.dumps(vars(args), indent=4, sort_keys=True))

    run_program(args, trial.number)

    with open('../results/loss_file', 'r') as loss_file:
        for line in loss_file:
            numbers = [float(s) for s in line.split()]
            if numbers[0] == trial.number:
                validation_loss = numbers[1]
                return validation_loss


def fine_tune(args):
    '''
    Performs the fine-tuning of some parameters using the optuna library. 

    Args:
        args: The arguments given as input to the program.
    
    TODO:
        If optune cannot run in multiple GPUs, put the following argument:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            raise ValueError('Optimization must only be run by 1 GPU to get one' +
                             'validation loss per epoch')
    '''
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, args), n_trials=100)
