import joblib
import logging
import os
import sys
import optuna

from train_test.train_program import TrainProgram

optuna.logging.get_logger("optuna").addHandler(
    logging.StreamHandler(sys.stdout))

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')
# To avoid having repeated logs!
logger.propagate = False

def objective(trial, args):
    '''
    Objective function used by optuna to fine-tune the parameters of a model.

    Args:
        trial: An object used by optuna library to manage the fine-tuning.
        args: The arguments given as input to the program.
    '''
    args.dropout = trial.suggest_float('dropout', 0.0, 1.0, step=0.1)
    args.gradient_clipping = trial.suggest_float('gradient_clipping', 0.0, 10.0, 
                                                 step=0.1)
    args.label_smoothing = trial.suggest_float('label_smoothing', 0.0, 1.0, 
                                               step=0.1)

    program = TrainProgram(args, trial.number)
    program.start()

    with open('../results/loss_file', 'r') as loss_file:
        for line in loss_file:
            # Each line of the loss_file has the following format:
            # <trial number> <validation loss> (example: 1 4.56421)
            numbers = [float(s) for s in line.split()]
            if numbers[0] == trial.number:
                validation_loss = numbers[1]
                return validation_loss


def save_optuna_study(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
    '''
    Callback function invoked after each trial of the optuna study.
    It saves the study to a file.

    Args:
        study: The optuna study (stores the best trial and hyper-parameters)
        trial: An optuna trial.
    '''
    joblib.dump(study, "../results/study.pkl")


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
    # Loading optuna study if there's a study stored
    if os.path.exists("../results/study.pkl"):
        study = joblib.load("../results/study.pkl")
    else:
        study = optuna.create_study(study_name="finetune",
                                    direction="minimize")

    # Lambda function used to pass args to the objective function
    study.optimize(lambda trial: objective(trial, args),
                   n_trials=100,
                   callbacks=[save_optuna_study])

    # Printing statistics about the study
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    logger.info(df)
    logger.info("Best trial:", study.best_trial.number)
    logger.info("Best loss:", study.best_trial.value)
    logger.info("Best hyperparameters:", study.best_params)
