import logging
from prettytable import PrettyTable
import torch
import torch.nn as nn
from evaluation.graphs import display_attention
from model.modules.copy_generator import CopyGenerator

from model.transformer.embeddings import Embeddings
from model.transformer.positional_encoding import PositionalEncoding
from model.transformer.encoder_layer import EncoderLayer
from model.transformer.decoder_layer import DecoderLayer

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')
# To avoid having repeated logs!
logger.propagate = False


class Transformer(nn.Module):
    '''
    Implements the Transformer architecture

    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            https://nlp.seas.harvard.edu/annotated-transformer/
    '''

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 num_heads,
                 num_layers,
                 d_ff,
                 max_src_length,
                 max_tgt_length,
                 dropout,
                 device,
                 init_type,
                 copy_attn,
                 pad_idx):
        '''
        Args:
            src_vocab_size (int): size of the source vocabulary.
            tgt_vocab_size (int): size of the target vocabulary.
            d_model (int): dimensionality of the model.
            num_heads (int): number of heads of the Multi Head Attention.
            num_layers (int): number of encoder and decoder layers.
            d_ff (int): the hidden layer size of the second-layer of the Feed
                        Forward Network (in encoder and decoder).
            max_src_length (int): maximum length of the source code.
            max_tgt_length (int): maximum length of the summaries.
            dropout (int): dropout probability (between 0 and 1).
            device: The device where the model and tensors are inserted (GPU or CPU).
            init_type (string): The weight initialization technique to be used
                                with the Transformer architecture.
            copy_attn (bool): Tells whether we want to use a copy generator in the
                              Transformer architecture or not (to copy source code tokens
                              to the summary).
            pad_idx (int): index of the <PAD> token

        Note: The log-softmax function is not applied here due to the later use 
              of CrossEntropyLoss, which requires the inputs to be unnormalized 
              logits.
        '''
        super(Transformer, self).__init__()
        self.encoder_embedding = Embeddings(src_vocab_size, d_model)
        self.decoder_embedding = Embeddings(tgt_vocab_size, d_model)
        self.encoder_positional_encoding = PositionalEncoding(
            d_model, max_src_length, dropout)
        self.decoder_positional_encoding = PositionalEncoding(
            d_model, max_tgt_length, dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm1 = nn.LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm2 = nn.LayerNorm(d_model)

        if copy_attn:
            self.fc = CopyGenerator(d_model, tgt_vocab_size, pad_idx)
        else:
            # Produce a probability distribution over output words,
            self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        self.copy_attn = copy_attn
        self.device = device

        self.init_weights(init_type)
        # self.print_transformer_information()

    def init_weights(self, init_type):
        '''
        Initialize parameters in the transformer model.

        Args:
            init_type (string): The weight initialization technique to be used
                                with the Transformer architecture.

        Source: https://nlp.seas.harvard.edu/annotated-transformer/
        '''
        for p in self.parameters():
            if p.dim() > 1:
                if init_type == 'kaiming':
                    nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(p)

    def count_parameters(self):
        '''
        Count the number of parameters of the transformer model.

        numel() returns the total number of elements in the tensor.

        Source: https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
        '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_transformer_information(self):
        '''
        Creates a table with information regarding the shapes of the layers of the
        Transformer and the number of parameters of each layer.

        Source: https://github.com/wasiahmad/NeuralCodeSum/blob/master/c2nl/models/transformer.py#L624
        '''
        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row(
                    [name, str(list(parameters.shape)), parameters.numel()])

        logger.info('Transformer parameters:\n%s' % table)

    def generate_src_mask(self, src):
        '''
        Creates a mask for the encoder input to prevent the encoder from paying 
        attention to padding token.

        Args:
            src: The encoder input. Shape: `(batch_size, src_len)`

        `src_len` is the length of the encoder input

        Returns:
            src_mask: Shape: `(batch_size, 1, 1, src_len)`
        '''
        # Creates two masks where each element is True if the corresponding
        # element in src is not equal to zero, and False otherwise.
        # unsequeeze() will add a new dimension at the specified index.
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def generate_tgt_mask(self, tgt):
        '''
        Creates a mask for the decoder input to prevent the decoder from looking 
        at future words during training.

        Args:
            tgt: The decoder input. Shape: `(batch_size, tgt_len)`

        `tgt_len` is the length of the decoder input

        Returns:
            tgt_mask: Shape: `(batch_size, 1, tgt_len, tgt_len)`
        '''
        # Creates two masks where each element is True if the corresponding
        # element in tgt is not equal to zero, and False otherwise.
        # unsequeeze() will add a new dimension at the specified index.
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        # Getting the target length
        seq_length = tgt.size(1)

        # Creating an lower triangular part of the matrix to mask padding and
        # future words in decoding
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(self.device)
        tgt_mask = tgt_mask & nopeak_mask
        return tgt_mask

    def encode(self, src, src_mask, display_attn=False):
        '''
        Encodes the input and returns the result of the last encoder layer.

        Args:
            src: The encoder input. Shape: `(batch_size, max_src_len)`
            src_mask: The encoder input mask (to avoid paying attention to padding).
                      Shape: `(batch_size, 1, 1, src_len)`
            display_attn (bool): Tells whether we want to save the attention of
                                 the last layer encoder multi-head attention.

        Returns:
            The resulting encoding from the last layer. 
            Shape: `(batch_size, max_src_len, d_model)`
        '''
        src_embedded = self.encoder_positional_encoding(
            self.encoder_embedding(src))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output, enc_attn_score = enc_layer(enc_output, src_mask)
        # Final encoder layer normalization according to Pre-LN
        enc_output = self.norm1(enc_output)

        if display_attn:
            # Displaying attention scores of the last encoder layer for the first
            # example passed as input
            display_attention(
                src[0], src[0], enc_attn_score[0], "encoder_self_attn")

        return enc_output

    def decode(self, src, src_mask, tgt, tgt_mask, enc_output, display_attn=False):
        '''
        Decodes the input and returns the result of the last decoder layer.

        Args:
            src: The encoder input (used to display cross attention). 
                 Shape: `(batch_size, max_src_len)`
            src_mask: The encoder input mask (to avoid paying attention to padding).
                      Shape: `(batch_size, 1, 1, src_len)`
            tgt: The decoder input. Shape: `(batch_size, max_tgt_len)`
            tgt_mask: The decoder input mask (to prevent the decoder from looking 
                      at future words during training). 
                      Shape: `(batch_size, 1, tgt_len, tgt_len)`
            enc_output: The output of the last encoder layer. 
                        Shape: `(batch_size, max_src_len, d_model)`
            display_attn (bool): Tells whether we want to save the attention of
                                 the last layer decoder multi-head attentions.

        Returns:
            - The resulting decoding from the last layer. 
              Shape: `(batch_size, tgt_seq_length, d_model)`
            - The attention scores between the decoder input and the encoder output
              (to be used in the copy generator). Shape: `(batch_size, num_heads, query_len, key_len)`
        '''
        tgt_embedded = self.decoder_positional_encoding(
            self.decoder_embedding(tgt))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output, dec_self_attn_score, dec_cross_attn_score = dec_layer(
                dec_output, enc_output, src_mask, tgt_mask)
        # Final decoder layer normalization according to Pre-LN
        dec_output = self.norm2(dec_output)

        if display_attn:
            # Displaying self and cross attention scores of the last decoder layer
            # for the first example passed as input
            display_attention(
                tgt[0], tgt[0], dec_self_attn_score[0], "decoder_self_attn")
            display_attention(
                src[0], tgt[0], dec_cross_attn_score[0], "decoder_cross_attn")

        return dec_output, dec_cross_attn_score

    def forward(self, src, tgt, src_map=None, display_attn=False):
        '''
        Args:
            src: The encoder input. Shape: `(batch_size, max_src_len)`
            tgt: The decoder input. Shape: `(batch_size, max_tgt_len)`
            src_map: TODO
            display_attn (bool): Tells whether we want to save the attention of
                                 the last layer encoder and decoder 
                                 multi-head attentions.

        Returns:
            output: The output of the Transformer model. 
                    Shape: `(batch_size, max_tgt_len, tgt_vocab_size)`
        '''
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)

        enc_output = self.encode(src, src_mask, display_attn)

        dec_output, dec_cross_attn_score = self.decode(src, 
                                                       src_mask, 
                                                       tgt, 
                                                       tgt_mask, 
                                                       enc_output, 
                                                       display_attn)

        if self.copy_attn:
            attn_last_head = dec_cross_attn_score[:, -1, :, :]
            attn = attn_last_head.contiguous().view(-1, attn_last_head.shape[-1])
            hidden = dec_output.contiguous().view(-1, dec_output.shape[-1])
            output = self.fc(hidden, attn, src_map)
        else:
            output = self.fc(dec_output)
    
        return output
