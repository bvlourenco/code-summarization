import logging
from prettytable import PrettyTable
import torch
import torch.nn as nn

from model.transformer.embeddings import Embeddings
from model.transformer.hierarchical_structure_variant_attention import compute_heads_distribution
from model.transformer.positional_encoding import PositionalEncoding
from model.transformer.encoder_layer import EncoderLayer
from model.transformer.decoder_layer import DecoderLayer
from transformers import (RobertaConfig, RobertaModel)

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
                 hyperparameter_hsva,
                 hyperparameter_data_flow,
                 hyperparameter_control_flow,
                 hyperparameter_ast):
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
            hyperparameter_hsva (int): Hyperparameter used in HSVA (Hierarchical
                                       Structure Variant Attention) to control the
                                       distribution of the heads by type.
            hyperparameter_data_flow (int): Hyperparameter used to adjust the 
                                            weight of the data flow adjacency 
                                            matrix in the self-attention.
            hyperparameter_control_flow (int): Hyperparameter used to adjust the 
                                               weight of the control flow adjacency 
                                               matrix in the self-attention.
            hyperparameter_ast (int): Hyperparameter used to adjust the 
                                      weight of the ast adjacency matrix in the 
                                      self-attention.

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

        self.structural_encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm1 = nn.LayerNorm(d_model)

        config = RobertaConfig.from_pretrained("microsoft/codebert-base", do_lower_case=False)
        self.code_encoder_layers = RobertaModel.from_pretrained("microsoft/codebert-base", config=config)
        self.code_encoder_linear_layer = nn.Linear(config.hidden_size, d_model)

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm2 = nn.LayerNorm(d_model)  

        self.num_heads = num_heads
        self.num_layers = num_layers

        # Produce a probability distribution over output words,
        self.fc = nn.Linear(d_model, tgt_vocab_size)

        self.device = device

        self.init_weights(init_type)
        # self.print_transformer_information()

        self.hyperparameter_hsva = hyperparameter_hsva
        self.hyperparameter_data_flow = hyperparameter_data_flow
        self.hyperparameter_control_flow = hyperparameter_control_flow
        self.hyperparameter_ast = hyperparameter_ast

        self.heads_distribution = []
        for layer_idx, _ in enumerate(self.structural_encoder_layers):
            heads_layer = compute_heads_distribution(self.num_heads,
                                                     layer_idx + 1,
                                                     self.hyperparameter_hsva)
            self.heads_distribution.append(heads_layer)

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
                    nn.init.kaiming_uniform_(
                        p, mode='fan_in', nonlinearity='relu')
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

    def encode(self,
               src,
               src_mask,
               source_ids,
               source_mask,
               token,
               statement,
               data_flow,
               control_flow,
               ast):
        '''
        Encodes the input and returns the result of the last encoder layer.

        Args:
            src: The encoder input. Shape: `(batch_size, max_src_len)`
            src_mask: The encoder input mask (to avoid paying attention to padding).
                      Shape: `(batch_size, 1, 1, src_len)`
            token: The token adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            statement: The statement adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            data_flow: The data flow adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            control_flow: The control flow adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            ast: The ast adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`

        Returns:
            The resulting encoding from the last layer and the encoder self-attention.
            Shapes: encoding: `(batch_size, max_src_len, d_model)`
                    attention: `(batch_size, num_heads, max_src_len, max_src_len)`        
        '''
        src_embedded = self.encoder_positional_encoding(
            self.encoder_embedding(src))
        enc_output = src_embedded

        # Used in multi-head attention to denote we're using a standard head attention
        zero_matrix = torch.zeros(token.shape[0], token.shape[1], token.shape[2],
                                  device=self.device)

        for layer_idx, enc_layer in enumerate(self.structural_encoder_layers):
            enc_output, enc_attn = enc_layer(enc_output, token, statement,
                                             data_flow, control_flow, ast,
                                             zero_matrix,
                                             self.heads_distribution[layer_idx],
                                             self.hyperparameter_data_flow,
                                             self.hyperparameter_control_flow,
                                             self.hyperparameter_ast,
                                             src_mask)
        # Final encoder layer normalization according to Pre-LN
        structural_enc_output = self.norm1(enc_output)

        code_enc_output = self.code_encoder_layers(source_ids, 
                                                   attention_mask=source_mask)
        # Adjusting the size of code encoder output to match d_model
        code_enc_linear_output = self.code_encoder_linear_layer(code_enc_output["last_hidden_state"])

        enc_output = 0.5 * structural_enc_output + 0.5 * code_enc_linear_output

        return enc_output, enc_attn

    def decode(self, src_mask, tgt, tgt_mask, enc_output):
        '''
        Decodes the input and returns the result of the last decoder layer.

        Args:
            src_mask: The encoder input mask (to avoid paying attention to padding).
                      Shape: `(batch_size, 1, 1, src_len)`
            tgt: The decoder input. Shape: `(batch_size, max_tgt_len)`
            tgt_mask: The decoder input mask (to prevent the decoder from looking 
                      at future words during training). 
                      Shape: `(batch_size, 1, tgt_len, tgt_len)`
            enc_output: The output of the last encoder layer. 
                        Shape: `(batch_size, max_src_len, d_model)`

        Returns:
            The resulting decoding from the last layer, the decoder 
            self-attention and the decoder cross-attention. 
            Shape: decoding - `(batch_size, tgt_seq_length, d_model)`
                   decoder self-attention - `(batch_size, num_heads, max_tgt_len, max_tgt_len)`
                   decoder cross-attention - `(batch_size, num_heads, max_tgt_len, max_src_len)`
        '''
        tgt_embedded = self.decoder_positional_encoding(
            self.decoder_embedding(tgt))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output, dec_self_attn, dec_cross_attn = dec_layer(
                dec_output, enc_output, src_mask, tgt_mask)
        # Final decoder layer normalization according to Pre-LN
        dec_output = self.norm2(dec_output)

        return dec_output, dec_self_attn, dec_cross_attn

    def forward(self, src, source_ids, source_mask, tgt, token, statement, 
                data_flow, control_flow, ast):
        '''
        Args:
            src: The encoder input. Shape: `(batch_size, max_src_len)`
            tgt: The decoder input. Shape: `(batch_size, max_tgt_len)`
            token: The token adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            statement: The statement adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            data_flow: The data flow adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            control_flow: The control flow adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`
            ast: The ast adjacency matrices. Shape: `(batch_size, max_src_len, max_src_len)`

        Returns:
            output: The output of the Transformer model, the encoder self-attention,
                    the decoder self-attention and the decoder cross-attention.
                    Shape: output: `(batch_size, max_tgt_len, tgt_vocab_size)`
                           encoder self-attention: `(batch_size, num_heads, max_src_len, max_src_len)`
                           decoder self-attention: `(batch_size, num_heads, max_tgt_len, max_tgt_len)`
                           decoder cross-attention: `(batch_size, num_heads, max_tgt_len, max_src_len)`
        '''
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)

        enc_output, enc_attn = self.encode(src,
                                           src_mask,
                                           source_ids,
                                           source_mask,
                                           token,
                                           statement,
                                           data_flow,
                                           control_flow,
                                           ast)

        dec_output, dec_self_attn, dec_cross_attn = self.decode(src_mask,
                                                                tgt,
                                                                tgt_mask,
                                                                enc_output)

        output = self.fc(dec_output)
        return output, enc_attn, dec_self_attn, dec_cross_attn
