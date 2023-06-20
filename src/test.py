import torch
from args.parse_args import parse_test_args
from train_test.test_program import TestProgram


def main():
    '''
    TODO
    '''
    args = parse_test_args()

    torch.manual_seed(0)
    program = TestProgram(args)
    program.start()


if __name__ == '__main__':
    main()
