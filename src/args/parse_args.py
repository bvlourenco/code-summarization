import argparse


def parse_arguments():
    '''
    Parses the arguments given as input to the program.
    '''
    parser = argparse.ArgumentParser(
        'Natural Language description generator for code snippets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_vocab_size', type=int, required=True,
                        help='Maximum allowed length for the code snippets dictionary')
    parser.add_argument('--tgt_vocab_size', type=int, required=True,
                        help='Maximum allowed length for the summaries dictionary')
    parser.add_argument('--max_seq_length', type=int, required=True,
                        help='Maximum allowed length for each code snippet and summary')
    parser.add_argument('--freq_threshold', type=int, required=True,
                        help='Minimum times a word must occur in corpus to be treated in vocab')
    parser.add_argument('--debug_max_lines', type=int, default=-1,
                        help='number of examples we want to read from the dataset. Used for debug.')

    parser.add_argument('--d_model', type=int, required=True,
                        help='Dimensionality of the model')
    parser.add_argument('--num_heads', type=int, required=True,
                        help='Number of heads of the Multi-Head Attention')
    parser.add_argument('--num_layers', type=int, required=True,
                        help='Number of layers of the Transformer encoder and decoder')
    parser.add_argument('--d_ff', type=int, required=True,
                        help='Number of units in the position wise feed forward network')
    parser.add_argument('--dropout', type=float, required=True,
                        help='Value of the dropout probability')

    parser.add_argument('--learning_rate', type=float, required=True,
                        help='Value of the learning rate')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Number of examples processed per batch')
    parser.add_argument('--num_workers', type=int, required=True,
                        help='Number of subprocesses used for data loading')
    parser.add_argument('--num_epochs', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--gradient_clipping', type=int, required=True,
                        help='Value of maximum norm of the gradients')

    parser.add_argument('--train_code_filename', type=str, required=True,
                        help="Filename of the training set with code snippets")
    parser.add_argument('--train_summary_filename', type=str, required=True,
                        help="Filename of the training set with summaries")
    parser.add_argument('--validation_code_filename', type=str, required=True,
                        help="Filename of the validation set with code snippets")
    parser.add_argument('--validation_summary_filename', type=str, required=True,
                        help="Filename of the validation set with summaries")

    parser.add_argument('--mode', type=str, required=True,
                        choices=['translation', 'loss'],
                        help="Tells if we want to compute only validation loss or \
                              validation loss and translation of the validation set")

    return parser.parse_args()
