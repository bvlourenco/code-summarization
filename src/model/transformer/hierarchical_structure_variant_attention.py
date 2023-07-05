
import math


def compute_heads_distribution(num_heads, num_encoder_layers, layer_index):
    '''
    Computes the heads distribution for a given encoder layer.
    Formula:

    number_token_headss = number_statement_heads = floor(h * ( (k - l) / (2*k - l) ) )
    number_data_flow_heads = number_control_flow_heads = number_AST_heads = floor((h * ( l / (2*k - l) )) / 3)

    k = hyper-parameter (in this case it's the number of encoder layers)
    l = index of the encoder layer
    h = number of heads

    We have token heads, statement heads, data-flow heads, control-flow heads, 
    AST heads and standard heads. Each head corresponds to each adjacency matrix
    that we receive as input.

    Args:
        num_heads (int): Number of heads in current encoder layer.
        num_encoder_layers (int): Number of encoder layers in Transformer.
        layer_index (int): The index of the encoder layer for which we're 
                           computing the distribution of heads. 
    
    

    TODO: try different values for hyper-parameter L
    According to the SG-Trans paper, k = L (hyper-parameter k is equal to 
                                            number of encoder layers)
    '''
    token_heads = math.floor(num_heads * 
                             ((num_encoder_layers - layer_index) / 
                              (2 * num_encoder_layers - layer_index)))
    statement_heads = token_heads
    if token_heads <= 0:
        data_flow_heads = 0
    else:
        data_flow_heads = math.floor((num_heads * 
                                      (layer_index / 
                                       (2 * num_encoder_layers - layer_index))) 
                                    / 3)
    control_flow_heads = data_flow_heads
    ast_heads = data_flow_heads
    standard_heads = num_heads - (token_heads + statement_heads + data_flow_heads 
                                  + control_flow_heads + ast_heads)
    return [token_heads, 
            statement_heads,
            data_flow_heads, 
            control_flow_heads,
            ast_heads,
            standard_heads]
