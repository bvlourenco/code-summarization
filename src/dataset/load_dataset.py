import json
import logging
import os
import pickle
from tqdm import tqdm

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')
# To avoid having repeated logs!
logger.propagate = False


def load_dataset_file(dataset_filename, type, debug_max_lines):
    '''
    Given a file with code snippets and a file with summaries, it loads all 
    examples of them from the respective files.

    Args:
        dataset_filename (string): The filename of the dataset.
        summary_filename (string): The filename of the file with summaries.
        type (string): Indicates whether we are loading the training set, the
                       validation set or the testing set from the files.
                       Can be one of the following: "train", "validation", "test"
        debug_max_lines (int): Represents the number of examples we want to read
                               from the dataset. If we pass a non-positive value, 
                               the whole dataset will be read.

    Returns:
        A list of code snippets examples and its respective tokens and a list of 
        summaries examples and its respective tokens.
    '''
    if (not os.path.exists(dataset_filename)):
        raise ValueError("dataset filename does not exist")

    code_texts, code_tokens, summary_texts, summary_tokens = [], [], [], []
    with open(dataset_filename) as dataset_file:
        if debug_max_lines > 0:
            num_lines = debug_max_lines
            description = "Reading {} {} entries of the dataset".format(
                debug_max_lines, type)
            lines_to_read = range(debug_max_lines)
        else:
            num_lines = sum(1 for _ in dataset_file)

            # Reset the file pointer to the beggining
            dataset_file.seek(0)

            description = "Reading {} dataset".format(type)
            lines_to_read = dataset_file

        for line in tqdm(lines_to_read, total=num_lines, desc=description):
            if debug_max_lines > 0:
                line = json.loads(next(dataset_file))
            else:
                line = json.loads(line)
            code_texts.append(line['original_string'])
            summary_texts.append(line['docstring'])
            code_tokens.append(line['code_tokens'])
            summary_tokens.append(line['docstring_tokens'])

    return code_texts, code_tokens, summary_texts, summary_tokens


def load_local_matrices(matrix_filename, type, debug_max_lines):
    '''
    Given a filename, it loads the token or statement adjacency matrices (they
    represent the local structure of the code).

    Each code snippet has a token and statement adjacency matrices.

    The token matrix tells whether two sub-tokens belong to the same token 
    originally or not. (for instance, canWalk is a token having two 
    sub-tokens: can and walk. These sub-tokens belong to the same token. 
    Similarly, can_walk has two sub-tokens belonging to the same token.)

    The statement matrix tells whether two tokens belong to the same instruction
    or not.

    Args:
        matrix_filename (string): The filename of the token/statement matrix.
        type (string): Indicates whether we are loading the training set, the
                       validation set or the testing set from the files.
                       Can be one of the following: "train token", 
                       "validation token", "test token", "train statement", 
                       "validation statement", "test statement"
        debug_max_lines (int): Represents the number of examples we want to read
                               from the dataset. If we pass a non-positive value, 
                               the whole dataset will be read.

    Returns:
        A list of token/statement matrices, each one corresponding to a 
        code snippet.
    '''

    if (not os.path.exists(matrix_filename)):
        raise ValueError(type + " matrix filename does not exist")

    matrices = []
    with open(matrix_filename) as matrix_file:
        if debug_max_lines > 0:
            num_lines = debug_max_lines
            description = "Reading {} {} matrices of the dataset"\
                          .format(debug_max_lines, type)
            lines_to_read = range(debug_max_lines)
        else:
            num_lines = sum(1 for _ in matrix_file)

            # Reset the file pointer to the beggining
            matrix_file.seek(0)

            description = "Reading {} matrices of the dataset".format(type)
            lines_to_read = matrix_file

        for line in tqdm(lines_to_read, total=num_lines, desc=description):
            if debug_max_lines > 0:
                line = next(matrix_file)
            matrices.append(line)
    return matrices


def load_global_matrices(matrix_filename, type, debug_max_lines):
    '''
    Given a filename, it loads the data flow/control flow adjacency matrices 
    (they represent the global structure of the code).

    Each code snippet has a data flow and control flow adjacency matrix.

    The data flow adjacency matrix tells whether two tokens have a data dependency
    or not.

    The control flow adjacency matrix tells whether a given token can be executed
    next to the current one (models the control dependencies).

    Args:
        matrix_filename (string): The filename of the data flow/control flow 
                                  matrix.
        type (string): Indicates whether we are loading the training set, the
                       validation set or the testing set from the files.
                       Can be one of the following: "train/validation/test 
                       data flow/control flow", 
        debug_max_lines (int): Represents the number of examples we want to read
                               from the dataset. If we pass a non-positive value, 
                               the whole dataset will be read.

    Returns:
        A list of data flow/control flow matrices, each one corresponding 
        to a code snippet.
    '''
    if (not os.path.exists(matrix_filename)):
        raise ValueError(type + " matrix filename does not exist")

    matrices = []
    with open(matrix_filename, 'rb') as matrix_file:
        if debug_max_lines > 0:
            logger.info("Reading {} {} matrices of the dataset".format(
                debug_max_lines, type))
            matrices = matrices[:debug_max_lines]
        else:
            logger.info("Reading {} matrices of the dataset".format(type))

        while True:
            try:
                matrices.append(pickle.load(matrix_file))
                if debug_max_lines > 0 and len(matrices) >= debug_max_lines:
                    break
            except EOFError:
                break
    return matrices


def load_matrices(token_filename,
                  statement_filename,
                  data_flow_filename,
                  control_flow_filename,
                  type,
                  debug_max_lines):
    '''
    Loads the token, statement, data flow and control flow adjacency matrices
    from the training/validation/testing set.

    Args:
        token_filename (string): The filename of the token matrix.
        statement_filename (string): The filename of the statement matrix.
        data_flow_filename (string): The filename of the data flow matrix.
        control_flow_filename (string): The filename of the control flow matrix.
        type (string): Indicates whether we are loading the training set, the
                       validation set or the testing set from the files.
                       Can be one of the following: "train", "validation", "test"
        debug_max_lines (int): Represents the number of examples we want to read
                               from the dataset. If we pass a non-positive value, 
                               the whole dataset will be read.

    Returns:
        The token, statement, data flow and control flow adjacency matrices. 
    '''
    token_matrices = load_local_matrices(token_filename,
                                         type + ' token',
                                         debug_max_lines)

    statement_matrices = load_local_matrices(statement_filename,
                                             type + ' statement',
                                             debug_max_lines)

    data_flow_matrices = load_global_matrices(data_flow_filename,
                                              type + ' data flow',
                                              debug_max_lines)

    control_flow_matrices = load_global_matrices(control_flow_filename,
                                                 type + ' control flow',
                                                 debug_max_lines)

    return token_matrices, statement_matrices, data_flow_matrices, \
        control_flow_matrices
