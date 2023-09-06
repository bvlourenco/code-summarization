import torch.nn as nn

from model.transformer.multi_head_attention import MultiHeadAttention
from model.transformer.position_wise_feed_forward_network import PositionWiseFeedForward
from model.transformer.sublayer_connection import SublayerConnection


class EncoderLayer(nn.Module):
    '''
    Implements the Encoder Layer building block of Transformer architecture

    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            https://nlp.seas.harvard.edu/annotated-transformer/
    '''

    def __init__(self, d_model, num_heads, d_ff, dropout):
        '''
        Args:
            d_model (int): the dimension of keys/values/queries in
                           MultiHeadedAttention, also the input size of
                           the first-layer of the PositionwiseFeedForward.
            num_heads (int): number of heads of the Multi Head Attention.
            d_ff (int): the hidden layer size of the second-layer of the Feed
                        Forward Network.
            dropout (int): dropout probability (between 0 and 1).
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = SublayerConnection(d_model, dropout)
        self.norm2 = SublayerConnection(d_model, dropout)

    def forward(self, 
                x, 
                token, 
                statement, 
                data_flow, 
                control_flow, 
                zero_matrix,
                heads_distribution, 
                hyperparameter_data_flow,
                hyperparameter_control_flow, 
                mask):
        '''
        Args:
            x: The input tensor (embeddings or from the previous layer). 
               Shape: `(batch_size, seq_length, d_model)`
            token: The token adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            statement: The statement adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            data_flow: The data flow adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            control_flow: The control flow adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            zero_matrix: A matrix of zeros used in multi-head attention to denote we're using a
                         standard head attention. Shape: `(batch_size, max_src_len, max_src_len)`
            heads_distribution: A list with 6 numbers indicating the distribution of the 
                                heads of the Multi-Head Attention. The sum of the 
                                numbers give us the number of heads.
                                The number of heads of each type is the following:
                                [TOKEN_HEADS, STATEMENT_HEADS, DATA_FLOW_HEADS, 
                                 CONTROL_FLOW_HEADS, STANDARD_HEADS]
            hyperparameter_data_flow (int): Hyperparameter used to adjust the 
                                            weight of the data flow adjacency 
                                            matrix in the self-attention.
            hyperparameter_control_flow (int): Hyperparameter used to adjust the 
                                               weight of the control flow adjacency 
                                               matrix in the self-attention.
            mask: Indicates which positions of the input sequence should have
                  attention or not. Shape: `(batch_size, 1, 1, seq_length)`

        `seq_length` represents the length of the input 

        Returns:
            The input tensor encoded. Shape: `(batch_size, seq_length, d_model)`
            attn_probs: The attention scores.
                        Shape: `(batch_size, num_heads, query_len, key_len)`
        '''
        x, attn_probs = self.norm1(x, "MultiHeadAttention", 
                                   lambda x: self.self_attn(x, x, x, 
                                                            token, 
                                                            statement, 
                                                            data_flow, 
                                                            control_flow,
                                                            zero_matrix,
                                                            heads_distribution,
                                                            hyperparameter_data_flow,
                                                            hyperparameter_control_flow,
                                                            mask))
        return self.norm2(x, "PositionWiseFeedForward", lambda x: self.feed_forward(x)), attn_probs
    