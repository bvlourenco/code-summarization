import logging
import math

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')
# To avoid having repeated logs!
logger.propagate = False


def compute_heads_distribution(num_heads, layer_index, hyperparameter_hsva):
    '''
    Computes the heads distribution for a given encoder layer.
    Formula:

    number_token_heads = number_statement_heads = 
                                            floor(h * ( (k - l) / (2*k - l) ) )
    number_data_flow_heads = number_control_flow_heads = 
                                            floor((h * ( l / (2*k - l) )) / 2)

    k = hyper-parameter
    l = index of the encoder layer
    h = number of heads

    We have token heads, statement heads, data-flow heads, control-flow heads, 
    and standard heads. Each head corresponds to each adjacency matrix
    that we receive as input.

    Args:
        num_heads (int): Number of heads in current encoder layer.
        layer_index (int): The index of the encoder layer for which we're 
                           computing the distribution of heads.
        hyperparameter_hsva (int): Hyperparameter used in HSVA (Hierarchical
                                   Structure Variant Attention) to control the
                                   distribution of the heads by type.

    Returns:
        A list containing the number of token, statement, data flow, control
        flow and standard head attentions in the following format:
        [
         num_token_head_attentions,
         num_statement_head_attentions, 
         num_data_flow_head_attentions,
         num_control_flow_head_attentions,
         num_standard_head_attentions
        ]

    According to the SG-Trans paper, k = L (hyper-parameter k is equal to 
                                            number of encoder layers)
    '''
    token_heads = max(0, math.floor(num_heads *
                                    ((hyperparameter_hsva - layer_index) /
                                     (2 * hyperparameter_hsva - layer_index))))
    statement_heads = token_heads
    if token_heads == 0:
        data_flow_heads = 0
    else:
        data_flow_heads = max(0, math.floor((num_heads *
                                             (layer_index /
                                              (2 * hyperparameter_hsva - layer_index)))
                                            / 2))
    control_flow_heads = data_flow_heads
    standard_heads = num_heads - (token_heads + statement_heads + data_flow_heads
                                  + control_flow_heads)

    logger.info(f"Heads distribution of layer {layer_index}:\n" +
                f"Token Heads: {token_heads} | " +
                f"Statement Heads: {statement_heads} | " +
                f"Data flow Heads: {data_flow_heads} | " +
                f"Control flow Heads: {control_flow_heads} | " +
                f"Standard Heads: {standard_heads}")

    return [token_heads,
            statement_heads,
            data_flow_heads,
            control_flow_heads,
            standard_heads]
