import argparse


def str2bool(value):
    '''
    Parses a given value to a boolean. Used to parse booleans in argument parser.

    Args:
        value: The value to be parsed to a boolean.
    '''
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    '''
    Parses the arguments given as input to the main program.
    '''
    parser = argparse.ArgumentParser(
        'Training a model to perform code summarization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_vocab_size', type=int, required=True,
                        help='Maximum allowed length for the code snippets dictionary')
    parser.add_argument('--tgt_vocab_size', type=int, required=True,
                        help='Maximum allowed length for the summaries dictionary')
    parser.add_argument('--max_src_length', type=int, required=True,
                        help='Maximum allowed length for the code snippets')
    parser.add_argument('--max_tgt_length', type=int, required=True,
                        help='Maximum allowed length for the summaries')
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
                        help='Value of the initial learning rate')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Number of examples processed per batch')
    parser.add_argument('--num_workers', type=int, required=True,
                        help='Number of subprocesses used for data loading')
    parser.add_argument('--num_epochs', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--gradient_clipping', type=float, required=True,
                        help='Value of maximum norm of the gradients')
    parser.add_argument('--label_smoothing', type=float, required=True,
                        help='Value of label smoothing in range [0.0, 1.0] \
                              to be applied in loss function.')
    parser.add_argument('--init_type', type=str, required=True,
                        choices=['kaiming', 'xavier'],
                        help="The weight initialization technique to be used in \
                              the Transformer architecture")

    parser.add_argument('--train_filename', type=str, required=True,
                        help="Filename of the training set. Each line is a \
                              JSON object with structure: \
                              {original_string: code, docstring: summary}")
    parser.add_argument('--validation_filename', type=str, required=True,
                        help="Filename of the training set. Each line is a \
                              JSON object with structure: \
                              {original_string: code, docstring: summary}")

    parser.add_argument('--mode', type=str, required=True,
                        choices=['beam', 'greedy', 'loss'],
                        help="Tells if we want to compute only validation loss or \
                              validation loss and translation of the validation set \
                                (using a greedy decoding or beam search strategy)")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Number of elements to store during beam search")

    parser.add_argument("--checkpoint", type=str2bool, required=True,
                        help="Save model + optimizer state after each epoch")
    parser.add_argument("--hyperparameter_tuning", type=str2bool, required=True,
                        help="Fine-tune some selected parameters or not")

    return parser.parse_args()


def parse_test_args():
    '''
    Parses the arguments given as input to the test program.
    '''
    parser = argparse.ArgumentParser(
        'Testing a code summarization model created previously',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--src_vocab_size', type=int, required=True,
                        help='Maximum allowed length for the code snippets dictionary')
    parser.add_argument('--tgt_vocab_size', type=int, required=True,
                        help='Maximum allowed length for the summaries dictionary')
    parser.add_argument('--max_src_length', type=int, required=True,
                        help='Maximum allowed length for the code snippets')
    parser.add_argument('--max_tgt_length', type=int, required=True,
                        help='Maximum allowed length for the summaries')
    parser.add_argument('--debug_max_lines', type=int, default=-1,
                        help='number of examples we want to read from the dataset. Used for debug.')

    parser.add_argument('--test_filename', type=str, required=True,
                        help="Filename of the training set. Each line is a \
                              JSON object with structure: \
                              {original_string: code, docstring: summary}")

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
                        help='Value of the initial learning rate')
    parser.add_argument('--batch_size', type=int, required=True,
                        help='Number of examples processed per batch')
    parser.add_argument('--num_workers', type=int, required=True,
                        help='Number of subprocesses used for data loading')
    parser.add_argument('--label_smoothing', type=float, required=True,
                        help='Value of label smoothing in range [0.0, 1.0] \
                              to be applied in loss function.')
    parser.add_argument('--init_type', type=str, required=True,
                        choices=['kaiming', 'xavier'],
                        help="The weight initialization technique to be used in \
                              the Transformer architecture")
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['beam', 'greedy'],
                        help="Tells if we want to translate the code snippets using \
                                a greedy decoding or beam search strategy")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Number of elements to store during beam search")

    return parser.parse_args()
