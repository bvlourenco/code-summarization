import torch
from args.parse_args import parse_test_args
from train_test.test_program import TestProgram


def main():
    '''
    Main function of the program to test.
    Parses the arguments given to the program, loads the trained model and its
    vocabulary and then tests it using the testing set.
    '''
    args = parse_test_args()

    torch.manual_seed(0)
    program = TestProgram(args)
    program.start()


if __name__ == '__main__':
    main()
