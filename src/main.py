import torch
from args.parse_args import parse_arguments
from finetune.fine_tuning import fine_tune
from train_test.train_program import TrainProgram

def main():
    '''
    TODO
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
