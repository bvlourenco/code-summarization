import torch.nn as nn


class SublayerConnection(nn.Module):
    '''
    A residual connection followed by a layer norm.

    Source: https://nlp.seas.harvard.edu/annotated-transformer/
            https://medium.com/@hunter-j-phillips/layer-normalization-e9ae93eb3c9c
    '''

    def __init__(self, d_model, dropout):
        '''
        Args:
            d_model (int): dimensionality of the model.
            dropout (int): dropout probability (between 0 and 1).
        '''
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        '''
        Apply residual connection to any sublayer with the same size.
        Applies pre layer-norm (opposed to post-layer norm). More about this in:
        https://github.com/harvardnlp/annotated-transformer/issues/92#issuecomment-1132966376
        https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab

        Args:
            x: The input before normalization. 
               Shape: `(batch_size, seq_length, d_model)`
            sublayer: The layer to be applied to input after the normalization.
                It's an anonymous function that receives the input and returns it
                after passing through the sublayer.
        
        `seq_length` is the number of elements of the input.

        Returns:
            The input normalized, after passing through the sublayer.
            Shape: `(batch_size, seq_length, d_model)`
        '''
        return x + self.dropout(sublayer(self.norm(x)))
