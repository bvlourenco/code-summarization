import torch.nn as nn

from transformer.multi_head_attention import MultiHeadAttention
from transformer.position_wise_feed_forward_network import PositionWiseFeedForward
from transformer.sublayer_connection import SublayerConnection


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

    def forward(self, x, mask):
        '''
        Args:
            x: The input tensor (embeddings or from the previous layer). 
               Shape: `(batch_size, seq_length, d_model)`
            mask: Indicates which positions of the input sequence should have
            attention or not. Shape: `(batch_size, 1, 1, seq_length)`

        `seq_length` represents the length of the input 

        Returns:
            The input tensor encoded. Shape: `(batch_size, seq_length, d_model)`
            attn_probs: The attention scores.
                        Shape: `(batch_size, num_heads, query_len, key_len)`
        '''
        x, attn_probs = self.norm1(x, "MultiHeadAttention", lambda x: self.self_attn(x, x, x, mask))
        return self.norm2(x, "PositionWiseFeedForward", lambda x: self.feed_forward(x)), attn_probs
    