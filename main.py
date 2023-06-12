import math
import torch
import torch.nn as nn
import torch.optim as optim
from evaluation import create_loss_plot
from transformer.transformer_model import Transformer


src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 10
dropout = 0.1

if __name__ == '__main__':
    '''
    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
    '''
    transformer = Transformer(src_vocab_size, 
                              tgt_vocab_size, 
                              d_model, 
                              num_heads, 
                              num_layers, 
                              d_ff, 
                              max_seq_length, 
                              dropout)
    
    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    epoch_loss = []
    for epoch in range(100):

        # Zeroing the gradients of transformer parameters
        optimizer.zero_grad()

        # Slicing the last element of tgt_data because it is used to compute the
        # loss.
        output = transformer(src_data, tgt_data[:, :-1])

        # output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
        # tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
        # criterion(prediction_labels, target_labels) 
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))

        # Computing gradients of loss through backpropagation
        loss.backward()

        # Update the model's parameters based on the computed gradients
        optimizer.step()
        
        print(f"Epoch: {epoch+1}")
        print(f'\tTrain Loss: {loss.item():.3f} | Train Perplexity: {math.exp(loss.item()):7.3f}')

        epoch_loss.append(loss.item())
    
    create_loss_plot(epoch_loss)

    '''
    TODO: For model training, check:
    https://nlp.seas.harvard.edu/annotated-transformer/
    https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe

    Idea: Create Trainer class (from https://wingedsheep.com/building-a-language-model/)

    TODO: For evaluation, check:
    https://github.com/DeepSoftwareAnalytics/CodeSumEvaluation
    https://arxiv.org/pdf/2107.07112.pdf

    TODO: In inference time, use beam search instead of greedy decoding. Explanation:
    https://datascience.stackexchange.com/a/93146

    TODO: Check tokenization techniques 
    https://towardsdatascience.com/word-subword-and-character-based-tokenization-know-the-difference-ea0976b64e17
    Good paper: https://openreview.net/pdf?id=htL4UZ344nF
                https://openreview.net/forum?id=htL4UZ344nF
    
    General tutorial with Transformer model, training, evaluation and inference:
    https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer

    Save and load model: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    Basics tutorials: https://pytorch.org/tutorials/beginner/basics/intro.html
    Training Transformer models in multiple GPUs: https://pytorch.org/tutorials/advanced/ddp_pipeline.html?highlight=transformer
    Datasets and Dataloaders: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=transformer
    Deploying Pytorch in python via a REST API with Flask: https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html?highlight=transformer
    Training with Pytorch: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html?highlight=transformer
    '''