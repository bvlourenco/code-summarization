import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    '''
    Implements the Multi-Head Attention building block of Transformer architecture

    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            https://medium.com/@hunter-j-phillips/multi-head-attention-7924371d477a
    '''

    def __init__(self, d_model, num_heads, dropout):
        '''
        Args:
            d_model (int): dimension of keys/queries/values.
            num_heads (int): number of heads of the Multi Head Attention.
            dropout (int): dropout probability (between 0 and 1).

        `d_k`/`d_v` are the size (dimensionality) of each query/key or value.
        '''
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        '''
        Computes the attention score for multiple heads using the following formula: 

        `Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V`

        Args:
            Q: The query matrix. Shape: `(batch_size, num_heads, query_len, d_k)`

            K: The key matrix. Shape: `(batch_size, num_heads, key_len, d_k)`

            V: The value matrix. Shape: `(batch_size, num_heads, key_len, d_k)`

            mask: A batch of matrices with 0/1 indicating which keys have zero
            or non-zero attention. Shape: `(batch_size, query_len, key_len)`

        `query_len`/`key_len` are the number of queries and keys/values
        for which attention is computed.
        '''
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # In the positions where the mask is 0, the attention score will be
            # 0. The softmax value is 0 when the value inside it is `-inf`
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Computing softmax over the last dimension of attention scores.
        # attn_scores has shape `(batch_size, num_heads, query_len, key_len)`
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(self.dropout(attn_probs), V)
        return output

    def split_heads(self, x):
        '''
        Split the input tensor into multiple heads.

        Args:
            x: Input tensor. Shape: `(batch_size, seq_length, d_model)`

        `seq_length` is the number of elements of the input.

        Returns:
            The reshaped and transposed input tensor. Shape: 
            `(batch_size, num_heads, seq_length, d_k)`
        '''
        batch_size, seq_length, _ = x.size()
        # x is reshaped from (batch_size, seq_length, d_model) to
        # (batch_size, seq_length, num_heads, d_k) using the view() method
        # and then is transposed to (batch_size, num_heads, seq_length, d_k)
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        '''
        Combine the output of multiple attention heads into a single tensor.

        Args:
            x: Output of attention heads. Shape: 
            `(batch_size, num_heads, seq_length, d_k)`

        Returns:
            The reshaped and transposed input tensor. Shape: 
            `(batch_size, seq_length, d_model)`
        '''
        batch_size, _, seq_length, _ = x.size()
        # x is transposed from (batch_size, num_heads, seq_length, d_k) to
        # (batch_size, seq_length, num_heads, d_k).
        # The contiguous() method assures that the tensor is stored in a
        # contiguous block of memory (may be required for subsequent operations).
        # Then, the tensor is reshaped from (batch_size, seq_length, num_heads, d_k)
        # to (batch_size, seq_length, d_model)
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        '''
        Computes the attention score for each head using the following formula: 

        `Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V`

        Then, it combines the attention scores from all heads. 

        Args:
            Q: A set of queries. Shape: `(batch_size, query_len, d_k)`

            K: A set of keys. Shape: `(batch_size, key_len, d_k)`

            V: A set of values. Shape: `(batch_size, key_len, d_v)`

            mask: A batch of matrices with 0/1 indicating which keys have zero
            or non-zero attention. Shape: `(batch, query_len, key_len)`

        Returns:
            output: The attended output. Shape: `(batch_size, query_len, d_v)`
        '''
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
