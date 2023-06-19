from args.parse_args import parse_arguments
from finetune.fine_tuning import fine_tune
from program import run_program

def main():
    '''
    Main function of the program to train and validate.
    Parses the arguments given to the program and depending on the input argument
    `hyperparameter_tuning`, it either trains the model or performs fine-tuning on
    the parameters of the model.
    '''
    args = parse_arguments()

    if args.hyperparameter_tuning:
        fine_tune(args)
    else:
        run_program(args)


if __name__ == '__main__':
    main()
