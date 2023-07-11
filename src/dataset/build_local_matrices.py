import logging
import torch

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')
# To avoid having repeated logs!
logger.propagate = False


def build_local_matrix(positions, type, max_src_length):
    '''
    Builds the token/statement adjacency matrices of a batch (as a tensor) 
    given their representation as a bunch of lists.

    Args:
        positions: A list with size `batch_size` where each element has size 
                   `max_src_len`. It contains the token matrices of a batch.
        type (string): Tells whether we are building the token or the statement
                       adjacency matrix.
                       Can be one of the following: "token", "statement"
        max_src_length (int): Maximum length of the source code.

    From: https://github.com/gszsectan/SG-Trans/blob/2afab8844e4f1e06c06585d80158bda947e0c720/java/c2nl/inputters/vector.py#L5
    '''
    try:
        # Parsing list of strings with numbers to an int
        position_list = list(list(map(int, i)) for i in positions)
    except:
        logger.info(
            "An exception occured while building the {} matrix".format(type))
        position_list = [[0]*len(positions[0]) for _ in range(len(positions))]

    maps = torch.ones(len(position_list), max_src_length, max_src_length)

    for i in range(len(position_list)):
        start = -1
        end = -1
        if type == 'token':
            for j in range(min(max_src_length, len(position_list[i]))):
                maps[i, j, j] = 0
                if position_list[i][j] == 1 and start == -1:
                    start = j
                if start >= 0 and (position_list[i][j] == 0 or
                                   j == len(position_list[i]) - 1):
                    end = j + 1
                    maps[i, start:end, start:end] = 0
                    start = -1
                    end = -1
        elif type == 'statement':
            instruction_counter = 0
            start = 0
            for j in range(min(max_src_length, len(position_list[i]))):
                maps[i, j, j] = 0
                if start >= 0 and (j == len(position_list[i]) - 1 or
                                   (j + 1 < len(position_list[i]) and
                                   position_list[i][j + 1] != instruction_counter)):
                    end = j + 1
                    maps[i, start:end, start:end] = 0
                    instruction_counter += 1
                    start = j + 1
                    end = -1

        else:
            raise ValueError(
                'Error building token/statement adjacency matrix: Unknown type: ' + type)
    return maps
