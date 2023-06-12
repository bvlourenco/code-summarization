import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    '''
    Implements the Feed Forward Network building block of Transformer architecture

    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            https://medium.com/@hunter-j-phillips/position-wise-feed-forward-network-ffn-d4cc9e997b4c
    '''

    def __init__(self, d_model, d_ff, dropout):
        '''
        Args:
            d_model (int): the size of input for the first-layer of the Feed
            Forward Network.
            d_ff (int): the hidden layer size of the second-layer of the Feed
            Forward Network.
            dropout (int): dropout probability (between 0 and 1).
        '''
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        Args:
            x: The input tensor. Shape: `(batch_size, input_len, d_model)`

        `input_len` is the number of elements of the input sequence.

        Returns:
            A tensor with shape `(batch_size, input_len, d_model)`
        '''
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
