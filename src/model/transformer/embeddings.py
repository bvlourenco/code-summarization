import math
import torch.nn as nn

class Embeddings(nn.Module):
    '''
    Implements an embedding layer.
    nn.Embedding does not have the scaling factor (math.sqrt(self.d_model)) in
    the forward() method, hence we have this custom class for embeddings. More
    info in the forward() method.

    Source: https://medium.com/@hunter-j-phillips/the-embedding-layer-27d9c980d124
    '''
    def __init__(self, vocab_size, d_model):
        '''
        Args:
            vocab_size (int): size of vocabulary
            d_model (int): embeddings size
        '''
        super(Embeddings, self).__init__()

        # embedding look-up table (lut)
        self.lut = nn.Embedding(vocab_size, d_model)

        # dimension of embeddings
        self.d_model = d_model

    def forward(self, x):
        '''
        Args:
            x: The transformer input . Shape: `(batch_size, seq_length)`
        
        `seq_length` represents the length of the input 

        Returns:
            An embedding tensor. Shape: `(batch_size, seq_length, d_model)`
        '''
        # embeddings by constant sqrt(d_model)
        # The reason we increase the embedding values before the addition is 
        # to make the positional encoding relatively smaller. 
        # This means the original meaning in the embedding vector won’t be 
        # lost when we add them together.
        return self.lut(x) * math.sqrt(self.d_model)
