from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch


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
                end = j
                maps[i, start:end, start:end] = 0
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
                end = j
                maps[i, start:end, start:end] = 0
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


def display_adjacency_matrices(input,
                               output,
                               adj_matrix,
                               label,
                               pdf):
    input = input[:150]
    output = output[:150]
    if label == 'in_token':
        adj_matrix = get_token_matrix([adj_matrix], 150)
        _adj_matrix = adj_matrix[0, :len(output), :len(input)].cpu().detach().numpy()
    elif label == 'in_statement':
        adj_matrix = get_statement_matrix([adj_matrix], 150)
        _adj_matrix = adj_matrix[0, :len(output), :len(input)].cpu().detach().numpy()
    else:
        _adj_matrix = adj_matrix.todense()[:len(output), :len(input)]

    height = 0.2 * _adj_matrix.shape[0]
    width = 0.25 * _adj_matrix.shape[1]
    _, ax = plt.subplots(figsize=(width, height))

    cax = ax.matshow(_adj_matrix, cmap='viridis', aspect='auto')

    cbar = ax.figure.colorbar(cax, ax=ax)
    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
    ax.set_xticks(range(len(input)))
    ax.set_yticks(range(len(output)))

    if isinstance(input[0], str):
        ax.set_xticklabels([t for t in input], rotation=90)
        ax.set_yticklabels(output)
    elif isinstance(input[0], int):
        ax.set_xticklabels(input, rotation=90)
        ax.set_yticklabels(output)

    ax.set_xticks(np.arange(-.5, len(input), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(output), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Loop over data dimensions and create text annotations.
    # for i in range(len(output)):
    #     for j in range(len(input)):
    #         ax.text(j, i, "{:.2f}".format(_adj_matrix[i, j]), ha="center", va="center", color="w")

    plt.tight_layout()
    pdf.savefig()
