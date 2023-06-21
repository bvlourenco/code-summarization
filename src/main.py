import torch
from args.parse_args import parse_arguments
from finetune.fine_tuning import fine_tune
from train_test.train_program import TrainProgram

def main():
    '''
    Main function of the program to train and validate.
    Parses the arguments given to the program and depending on the input argument
    `hyperparameter_tuning`, it either trains the model or performs fine-tuning on
    the parameters of the model.
    '''
    args = parse_arguments()

    torch.manual_seed(0)
    if args.hyperparameter_tuning:
        fine_tune(args)
    else:
        program = TrainProgram(args)
        program.start()


if __name__ == '__main__':
    main()
