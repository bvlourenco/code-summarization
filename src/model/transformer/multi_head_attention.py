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

    def scaled_dot_product_attention(self, Q, K, V,
                                     token,
                                     statement,
                                     data_flow,
                                     control_flow,
                                     zero_matrix,
                                     heads_distribution,
                                     hyperparameter_data_flow,
                                     hyperparameter_control_flow,
                                     mask=None):
        '''
        Computes the attention score for multiple heads using the following formula: 

        `Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V`

        Args:
            Q: The query matrix. Shape: `(batch_size, num_heads, query_len, d_k)`

            K: The key matrix. Shape: `(batch_size, num_heads, key_len, d_k)`

            V: The value matrix. Shape: `(batch_size, num_heads, key_len, d_k)`

            token: The token adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
                   Only used for the self-attention of the encoder layer.
            statement: The statement adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
                       Only used for the self-attention of the encoder layer.
            data_flow: The data flow adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
                       Only used for the self-attention of the encoder layer.
            control_flow: The control flow adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
                          Only used for the self-attention of the encoder layer.
            zero_matrix: A matrix of zeros used in multi-head attention to denote we're using a
                         standard head attention. Shape: `(batch_size, max_src_len, max_src_len)`
                         Only used for the self-attention of the encoder layer.
            heads_distribution: A list with 6 numbers indicating the distribution of the 
                                heads of the Multi-Head Attention. The sum of the 
                                numbers give us the number of heads.
                                The number of heads of each type is the following:
                                [TOKEN_HEADS, STATEMENT_HEADS, DATA_FLOW_HEADS, 
                                 CONTROL_FLOW_HEADS, STANDARD_HEADS]
                                Only used for the self-attention of the encoder layer.
            hyperparameter_data_flow (int): Hyperparameter used to adjust the 
                                            weight of the data flow adjacency 
                                            matrix in the self-attention.
                                            Only used for the self-attention of 
                                            the encoder layer.
            hyperparameter_control_flow (int): Hyperparameter used to adjust the 
                                               weight of the control flow adjacency 
                                               matrix in the self-attention.
                                               Only used for the self-attention of 
                                               the encoder layer.

            mask: A batch of matrices with 0/1 indicating which keys have zero
            or non-zero attention. Shape: `(batch_size, query_len, key_len)`

        `query_len`/`key_len` are the number of queries and keys/values
        for which attention is computed.

        Returns:
            output: The result of adding attention scores to the input.
                    Shape: `(batch_size, num_heads, query_len, d_k)`
            attn_probs: The attention scores.
                        Shape: `(batch_size, num_heads, query_len, key_len)`
        '''
        attn_scores = torch.matmul(
            Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if token is not None:
            # In the token adjacency matrix, elements who are 1 will be -inf
            # (to make the attention 0 in those places). Elements who are already
            # 0 will remain the same.
            # The same happens with the statement adjacency matrix
            token = -1e9 * token
            statement = -1e9 * statement

            # The matrices have shape `(batch_size, max_src_len, max_src_len)`
            # torch.stack(..., 1) will concatenate the different adjacency matrices
            # and create a new dimension for that in index 1.
            # So, local_mask_map and global_enhance_map will have shapes of 
            # `(batch_size, num_heads, max_src_len, max_src_len)` since the sum
            # of all entries of `heads_distribution` is the number of heads.
            local_mask_map = torch.stack([token for _ in range(heads_distribution[0])] +
                                         [statement for _ in range(heads_distribution[1])] +
                                         [zero_matrix for _ in range(heads_distribution[2])] +
                                         [zero_matrix for _ in range(heads_distribution[3])] +
                                         [zero_matrix for _ in range(heads_distribution[4])], 1)

            global_enhance_map = torch.stack([zero_matrix for _ in range(heads_distribution[0])] +
                                             [zero_matrix for _ in range(heads_distribution[1])] +
                                             [hyperparameter_data_flow * data_flow for _ in range(heads_distribution[2])] +
                                             [hyperparameter_control_flow * control_flow for _ in range(heads_distribution[3])] +
                                             [zero_matrix for _ in range(heads_distribution[4])], 1)

            global_enhance_map = torch.abs(global_enhance_map.mul(attn_scores))
            attn_scores = attn_scores + local_mask_map + global_enhance_map

        if mask is not None:
            # In the positions where the mask is 0, the attention score will be
            # 0. The softmax value is 0 when the value inside it is `-inf`
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Computing softmax over the last dimension of attention scores.
        # attn_scores has shape `(batch_size, num_heads, query_len, key_len)`
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(self.dropout(attn_probs), V)

        # attn_probs is returned to visualize the attention scores
        return output, attn_probs

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

    def forward(self, Q, K, V,
                token=None,
                statement=None,
                data_flow=None,
                control_flow=None,
                zero_matrix=None,
                heads_distribution=None,
                hyperparameter_data_flow=None,
                hyperparameter_control_flow=None,
                mask=None):
        '''
        Computes the attention score for each head using the following formula: 

        `Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V`

        Then, it combines the attention scores from all heads. 

        Args:
            Q: A set of queries. Shape: `(batch_size, query_len, d_k)`

            K: A set of keys. Shape: `(batch_size, key_len, d_k)`

            V: A set of values. Shape: `(batch_size, key_len, d_v)`

            token: The token adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
                   Only used for the self-attention of the encoder layer.
            statement: The statement adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
                       Only used for the self-attention of the encoder layer.
            data_flow: The data flow adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
                       Only used for the self-attention of the encoder layer.
            control_flow: The control flow adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
                          Only used for the self-attention of the encoder layer.
            zero_matrix: A matrix of zeros used in multi-head attention to denote we're using a
                         standard head attention. Shape: `(batch_size, max_src_len, max_src_len)`
                         Only used for the self-attention of the encoder layer.
            heads_distribution: A list with 6 numbers indicating the distribution of the 
                                heads of the Multi-Head Attention. The sum of the 
                                numbers give us the number of heads.
                                The number of heads of each type is the following:
                                [TOKEN_HEADS, STATEMENT_HEADS, DATA_FLOW_HEADS, 
                                 CONTROL_FLOW_HEADS, STANDARD_HEADS]
                                Only used for the self-attention of the encoder layer.
            hyperparameter_data_flow (int): Hyperparameter used to adjust the 
                                            weight of the data flow adjacency 
                                            matrix in the self-attention.
                                            Only used for the self-attention of 
                                            the encoder layer.
            hyperparameter_control_flow (int): Hyperparameter used to adjust the 
                                               weight of the control flow adjacency 
                                               matrix in the self-attention.
                                               Only used for the self-attention of 
                                               the encoder layer.

            mask: A batch of matrices with 0/1 indicating which keys have zero
                  or non-zero attention. Shape: `(batch, query_len, key_len)`

        Returns:
            output: The attended output. Shape: `(batch_size, query_len, d_v)`
            attn_probs: The attention scores.
                        Shape: `(batch_size, num_heads, query_len, key_len)`
        '''
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output, attn_probs = self.scaled_dot_product_attention(Q, K, V,
                                                                    token,
                                                                    statement,
                                                                    data_flow,
                                                                    control_flow,
                                                                    zero_matrix,
                                                                    heads_distribution,
                                                                    hyperparameter_data_flow,
                                                                    hyperparameter_control_flow,
                                                                    mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_probs
