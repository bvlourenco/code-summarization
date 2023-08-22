import logging
import torch

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')
# To avoid having repeated logs!
logger.propagate = False


def get_token_positions(positions):
    '''
    Transforms the token/statement matrices token positions from strings to 
    integers.

    Args:
        positions: A list with size `batch_size` where each element has size 
                   `max_src_len`. It contains the token/statement matrices of a batch.
    
    Returns:
        position_list: The same list with each number converted to an integer.
    '''
    try:
        position_list = [list(map(int, i)) for i in positions]
    except:
        logger.info("An exception occured while building the token/statement matrix")
        position_list = [[0] * len(positions[0])
                         for _ in range(len(positions))]
    return position_list

def get_token_matrix(positions, max_src_length):
    '''
    Builds the token adjacency matrix of a batch (as a tensor) 
    given their representation as a bunch of lists.

    Args:
        positions: A list with size `batch_size` where each element has size 
                   `max_src_len`. It contains numbers where each number
                   represents a token. If the number is positive, then the token 
                   is part of a snake_case or camelCase token. Otherwise the 
                   number is 0.
        max_src_length (int): Maximum length of the source code.
    
    Returns:
        The token adjacency matrices. 
        Shape: `(batch_size, max_src_length, max_src_length)`

    Adapted from: 
    https://github.com/shuzhenggao/SG-Trans/blob/2afab8844e4f1e06c06585d80158bda947e0c720/python/c2nl/inputters/vector.py#L47
    '''
    batch_size = len(positions)
    position_list = get_token_positions(positions)
    maps = torch.ones(batch_size, max_src_length, max_src_length)
    for i, code_positions in enumerate(position_list):
        start = end = -1
        for j, token_position in enumerate(code_positions):
            if j >= max_src_length:
                break

            maps[i, j, j] = 0

            if token_position != 0 and start == -1:
                start = j

            if start >= 0 and token_position != code_positions[start]:
                end = j
                maps[i, start:end, start:end] = 0
                if token_position != 0:
                    start = j
                else:
                    start = end = -1
            elif start >= 0 and j == len(code_positions) - 1:
                end = j + 1
                maps[i, start:end, start:end] = 0
    return maps


def get_statement_matrix(positions, max_src_length):
    '''
    Builds the statement adjacency matrices of a batch (as a tensor) 
    given their representation as a bunch of lists.

    Args:
        positions: A list with size `batch_size` where each element has size 
                   `max_src_len`. It contains numbers where each number
                   represents the instruction number where the token is placed.
        max_src_length (int): Maximum length of the source code.
    
    Returns:
        The statement adjacency matrices. 
        Shape: `(batch_size, max_src_length, max_src_length)`

    Adapted from: 
    https://github.com/shuzhenggao/SG-Trans/blob/2afab8844e4f1e06c06585d80158bda947e0c720/python/c2nl/inputters/vector.py#L71
    '''
    batch_size = len(positions)
    position_list = get_token_positions(positions)
    maps = torch.ones(batch_size, max_src_length, max_src_length)
    for i, code_instruction_numbers in enumerate(position_list):
        start = end = 0
        for j, token_instruction_number in enumerate(code_instruction_numbers):
            if j >= max_src_length:
                break

            maps[i, j, j] = 0
            if token_instruction_number != code_instruction_numbers[start]:
                end = j
                maps[i, start:end, start:end] = 0
                start = end = j
            elif j == len(code_instruction_numbers) - 1:
                end = j + 1
                maps[i, start:end, start:end] = 0
    return maps
