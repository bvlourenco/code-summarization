import torch.nn as nn

from model.transformer.multi_head_attention import MultiHeadAttention
from model.transformer.position_wise_feed_forward_network import PositionWiseFeedForward
from model.transformer.sublayer_connection import SublayerConnection


class DecoderLayer(nn.Module):
    '''
    Implements the Decoder Layer building block of Transformer architecture

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
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = SublayerConnection(d_model, dropout)
        self.norm2 = SublayerConnection(d_model, dropout)
        self.norm3 = SublayerConnection(d_model, dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        '''
        Args:
            x: The input tensor (embeddings or from the previous layer). 
               Shape: `(batch_size, tgt_seq_length, d_model)`
            enc_output: The encoder output. Shape: `(batch_size, src_seq_length, d_model)`
            src_mask: Indicates which positions of the encoder input sequence should have
                      attention or not. Shape: `(batch_size, 1, 1, src_seq_length)`
            tgt_mask: Indicates which positions of the decoder input sequence should have
                      attention or not. Prevents paying attention to future tokens in the
                      decoding phase. Shape: `(batch_size, 1, tgt_seq_length, tgt_seq_length)`

        `src_seq_length` represents the length of the encoder input

        `tgt_seq_length` represents the length of the decoder input

        Returns:
            The result of decoding the input. Shape: `(batch_size, tgt_seq_length, d_model)`.
            self_attn_probs: The attention scores between the decoder input.
            cross_attn_probs: The attention scores between the decoder input and the encoder output.
        '''
        x, self_attn_probs = self.norm1(x, "MultiHeadAttention", lambda x: self.self_attn(x, x, x, tgt_mask))
        x, cross_attn_probs = self.norm2(x, "MultiHeadAttention", lambda x: self.cross_attn(x, enc_output, enc_output, src_mask))
        return self.norm3(x, "PositionWiseFeedForward", lambda x: self.feed_forward(x)), self_attn_probs, cross_attn_probs