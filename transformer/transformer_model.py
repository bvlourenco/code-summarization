import torch
import torch.nn as nn

from transformer.embeddings import Embeddings
from transformer.positional_encoding import PositionalEncoding
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer


class Transformer(nn.Module):
    '''
    Implements the Transformer architecture

    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            https://nlp.seas.harvard.edu/annotated-transformer/
    '''

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        '''
        Args:
            src_vocab_size (int): size of the source vocabulary.
            tgt_vocab_size (int): size of the target vocabulary.
            d_model (int): dimensionality of the model.
            num_heads (int): number of heads of the Multi Head Attention.
            num_layers (int): number of encoder and decoder layers.
            d_ff (int): the hidden layer size of the second-layer of the Feed
                        Forward Network (in encoder and decoder).
            max_seq_length (int): maximum length of the input.
            dropout (int): dropout probability (between 0 and 1).

        Note: The log-softmax function is not applied here due to the later use 
              of CrossEntropyLoss, which requires the inputs to be unnormalized 
              logits.
        '''
        super(Transformer, self).__init__()
        self.encoder_embedding = Embeddings(src_vocab_size, d_model)
        self.decoder_embedding = Embeddings(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm1 = nn.LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm2 = nn.LayerNorm(d_model)

        # Produce a probability distribution over output words,
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def generate_mask(self, src, tgt):
        '''
        Creates a mask for encoder input and decoder input to prevent
        the decoder from looking at future words during training and to prevent
        the encoder from paying attention to padding tokens.

        Args:
            src: The encoder input. Shape: `(batch_size, src_len)`
            tgt: The decoder input. Shape: `(batch_size, tgt_len)`

        `src_len` is the length of the encoder input
        `tgt_len` is the length of the decoder input

        Returns:
            src_mask: Shape: `(batch_size, 1, 1, src_len)`
            tgt_mask: Shape: `(batch_size, 1, tgt_len, tgt_len)`
        '''
        # Creates two masks where each element is True if the corresponding
        # element in src/tht is not equal to zero, and False otherwise.
        # unsequeeze() will add a new dimension at the specified index.
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        # Getting the target length
        seq_length = tgt.size(1)

        # Creating an lower triangular part of the matrix to mask padding and
        # future words in decoding
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        '''
        Args:
            src: The encoder input. Shape: `(batch_size, src_len)`
            tgt: The decoder input. Shape: `(batch_size, tgt_len)`

        Returns:
            output: The output of the Transformer model. 
                    Shape: `(batch_size, tgt_len, tgt_vocab_size)`
        '''
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.positional_encoding(self.encoder_embedding(src))
        tgt_embedded = self.positional_encoding(self.decoder_embedding(tgt))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        # Final encoder layer normalization according to Pre-LN
        enc_output = self.norm1(enc_output)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        # Final decoder layer normalization according to Pre-LN
        dec_output = self.norm2(dec_output)

        output = self.fc(dec_output)
        return output
