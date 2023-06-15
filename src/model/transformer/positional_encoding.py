import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    '''
    Injects the position information of each token in the input sequence.

    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
    '''

    def __init__(self, d_model, max_length, dropout):
        '''
        Args:
            max_length (int): maximum length of the input
            d_model (int): embedding size
            dropout (int): dropout probability (between 0 and 1).
        '''
        super(PositionalEncoding, self).__init__()

        # Positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        # Generates a sequence of values from 0 to max_length - 1 and adds
        # a new dimension. Tensor will have shape (max_length, 1)
        position = torch.arange(
            0, max_length, dtype=torch.float).unsqueeze(1)
        # Creates a 1D tensor from 0 to d_model with a step size of 2.
        # Multiplies it by a step size to control the range and spacing of the
        # positional encodings.
        # The torch.exp does the function element-wise.
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        # Having sin values on even columns and cos values on odd columns e
        # ensures that neighboring positions in the sequence have
        # different representations

        # Selects even columns of pe
        pe[:, 0::2] = torch.sin(position * div_term)

        # Selects odd columns of pe
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe gets the shape (1, max_length, d_model) with the `unsqueeze`
        # `register_bufer` registers the tensor as a buffer in the class
        # instance.
        # It can be accessed later during the forward pass or other operations.
        # Registering as a buffer ensures that the state of pe is preserved
        # when the model is saved or loaded.
        self.register_buffer('pe', pe.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Adds information about the position of tokens to the input.

        Args:
            x: An embedded input. Shape: `(batch_size, seq_length, d_model)`

        `seq_length` represents the length of the input 

        Returns:
            A tensor with the same shape.
        '''
        # pe is sliced to match the length of the input sentence
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)
