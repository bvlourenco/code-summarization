import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset.custom_collate_fn import MyCollate
from evaluation import create_loss_plot
from dataset.train_dataset import TrainDataset
from transformer.transformer_model import Transformer
from timeit import default_timer as timer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset.validation_dataset import ValidationDataset

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 10
dropout = 0.1
learning_rate = 0.0001
batch_size = 32
num_epochs = 1
gradient_clipping = 1
train_code_filename = '../data/train_code.txt'
train_summary_filename = '../data/train_summary.txt'
validation_code_filename = '../data/validation_code.txt'
validation_summary_filename = '../data/validation_summary.txt'
freq_threshold = 0
num_workers = 0


def evaluate(model, criterion, device, val_dataloader):
    '''
    Validates the model to see how well he generalizes. 
    Does not update the weights of the model.

    Args:
        model: The model (an instance of Transformer).
        criterion: The function used to compute the loss (an instance of nn.CrossEntropyLoss).
        device: The device where the model and tensors are inserted (GPU or CPU).
        val_dataloader: A Dataloader object that contains the validation set.

    Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
    '''
    # Set the model to evaluation mode
    model.eval()
    losses = 0

    # evaluate without updating gradients
    # Tells pytorch to not calculate the gradients
    with torch.no_grad():
        for src, tgt in tqdm(val_dataloader, desc="Validating"):
            src = src.to(device)
            tgt = tgt.to(device)

            # Slicing the last element of tgt_data because it is used to compute the
            # loss.
            output = model(src, tgt[:, :-1])

            # - output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
            # - tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
            # - criterion(prediction_labels, target_labels)
            # - tgt[:, 1:] removes the first token from the target sequence since we are training the
            # model to predict the next word based on previous words.
            loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                             tgt[:, 1:].contiguous().view(-1))

            losses += loss.item()

    return losses / len(list(val_dataloader))


def train_epoch(model, optimizer, criterion, device, train_dataloader):
    '''
    Trains the model during one epoch, using the examples of the dataset for training.
    Computes the loss during this epoch.

    Args:
        model: The model (an instance of Transformer).
        optimizer: The optimizer used in training (an instance of Adam optimizer).
        criterion: The function used to compute the loss (an instance of nn.CrossEntropyLoss).
        device: The device where the model and tensors are inserted (GPU or CPU). 
        train_dataloader: A Dataloader object that contains the training set.

    Returns:
        The average loss across all examples during one epoch.

    Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
            https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
    '''
    # Set the model to training mode
    model.train()
    losses = 0

    for src, tgt in tqdm(train_dataloader, desc="Training"):
        # Passing vectors to GPU if it's available
        src.to(device)
        tgt.to(device)

        # Zeroing the gradients of transformer parameters
        optimizer.zero_grad()

        # Slicing the last element of tgt_data because it is used to compute the
        # loss.
        output = model(src, tgt[:, :-1])

        # - output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
        # - tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
        # - criterion(prediction_labels, target_labels)
        # - tgt[:, 1:] removes the first token from the target sequence since we are training the
        # model to predict the next word based on previous words.
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                         tgt[:, 1:].contiguous().view(-1))

        # Computing gradients of loss through backpropagation
        loss.backward()

        # clip the weights
        clip_grad_norm_(model.parameters(), gradient_clipping)

        # Update the model's parameters based on the computed gradients
        optimizer.step()

        losses += loss.item()

    return losses / len(list(train_dataloader)), train_dataloader

def get_dataloader(dataset, train_dataset=None):
    '''
    Get a dataloader given a dataset.

    Args:
        dataset: The dataset from which we will build the dataloader.
                 Instance of TrainDataset or ValidationDataset.
        train_dataset (optional): The training dataset (instance of TrainDataset)

    Returns:
        A new dataloader.
    
    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''
    # If we're getting the dataloader for the validation dataset, `train_dataset`
    # will be received as argument. But if we're getting the dataloader for the 
    # training dataset, the variable will be None to avoid passing two times the
    # same argument in the function.
    if train_dataset is None:
        train_dataset = dataset
    pad_idx = train_dataset.source_vocab.token_to_idx['<PAD>']
    bos_idx = train_dataset.source_vocab.token_to_idx['<BOS>']
    eos_idx = train_dataset.source_vocab.token_to_idx['<EOS>']
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=True,
                      drop_last=True,
                      pin_memory=False,
                      collate_fn=MyCollate(pad_idx, bos_idx, eos_idx, device, max_seq_length))

def create_dataloaders():
    '''
    Creates a training and validation dataset and dataloaders.

    Returns:
        The training and validation dataloaders (instances of Dataloader).
    
    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''
    train_dataset = TrainDataset(train_code_filename, train_summary_filename,
                                 freq_threshold, src_vocab_size, tgt_vocab_size)
    train_dataloader = get_dataloader(train_dataset)

    validation_dataset = ValidationDataset(
        validation_code_filename, validation_summary_filename, train_dataset)
    val_dataloader = get_dataloader(validation_dataset, train_dataset)
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    '''
    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
    '''
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer = Transformer(src_vocab_size,
                              tgt_vocab_size,
                              d_model,
                              num_heads,
                              num_layers,
                              d_ff,
                              max_seq_length,
                              dropout,
                              device)

    # Passing the model and all its layers to GPU if available
    transformer = transformer.to(device)

    train_dataloader, val_dataloader = create_dataloaders()

    # ignore_index is the padding token index
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(),
                           lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    train_epoch_loss, val_epoch_loss = [], []
    for epoch in range(1, num_epochs + 1):
        start_time = timer()
        train_loss, train_dataset = train_epoch(
            transformer, optimizer, criterion, device, train_dataloader)
        val_loss = evaluate(transformer, criterion, device, val_dataloader)
        end_time = timer()

        print(f"Epoch: {epoch} | Time = {(end_time - start_time):.3f}s")
        print(f'Train Loss: {train_loss:.3f} | Train Perplexity: {math.exp(train_loss):7.3f}')
        print(f'Validation Loss: {val_loss:.3f} | Validation Perplexity: {math.exp(val_loss):7.3f}')

        train_epoch_loss.append(train_loss)
        val_epoch_loss.append(val_loss)

    create_loss_plot(train_epoch_loss, val_epoch_loss)

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
    
    TODO: Check Kaiming initialization
    https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    
    General tutorial with Transformer model, training, evaluation and inference:
    https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer

    Save and load model: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    Basics tutorials: https://pytorch.org/tutorials/beginner/basics/intro.html
    Training Transformer models in multiple GPUs: https://pytorch.org/tutorials/advanced/ddp_pipeline.html?highlight=transformer
    Datasets and Dataloaders: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=transformer
    Deploying Pytorch in python via a REST API with Flask: https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html?highlight=transformer
    Training with Pytorch: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html?highlight=transformer
    '''
