import math
import os
import torch
from dataset.dataloader import create_dataloaders
from evaluation.graphs import create_loss_plot
from model.model import Model
from timeit import default_timer as timer


def train_validate_model(model,
                         num_epochs,
                         train_dataloader,
                         val_dataloader,
                         tgt_vocab_size,
                         gradient_clipping,
                         mode,
                         target_vocab,
                         max_tgt_length,
                         checkpoint,
                         device,
                         gpu_rank):
    '''
    Performs the model training. In each epoch, it is done an evaluation using the
    validation set. In the end, it plots a graph with the training and validation loss.

    Args:
        model: The model (an instance of Transformer). 
        num_epochs (int): The number of training epochs.
        train_dataloader: A Dataloader object that contains the training set.
        val_dataloader: A Dataloader object that contains the validation set.
        tgt_vocab_size (int): size of the target vocabulary.
        gradient_clipping (int): Maximum norm of the gradient.
        mode (string): Indicates whether we only want to compute validation loss or if we also
                       want to translate the source sentences in the validation set.
                       Can be one of the following: "loss", "translation"
        source_vocab: The vocabulary built from the code snippets in training set.
        target_vocab: The vocabulary built from the summaries in training set.
        max_tgt_length (int): Maximum length of the generated summaries.
        checkpoint (bool): Flag that tells whether we want to save a checkpoint of the model
                           at the end of each epoch or not.
        device: The device where the model and tensors are inserted (GPU or CPU).
        gpu_rank (int): The rank of the GPU.
                        It has the value of -1 if no GPUs are avaiable.

    Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
    '''

    # Load model checkpoint (loading model parameters and optimizer state)
    # if the checkpoint exists
    if os.path.isfile('../results/model_weights_checkpoint.pth'):
        start_epoch, train_epoch_loss, val_epoch_loss = model.load_checkpoint(gpu_rank)
        start_epoch += 1
    else:
        train_epoch_loss, val_epoch_loss = [], []
        start_epoch = 1

    best_val_loss = float('inf')
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch: {epoch}")

        if device == torch.device('cuda'):
            train_dataloader.sampler.set_epoch(epoch)
            val_dataloader.sampler.set_epoch(epoch)

        start_time = timer()
        train_loss = model.train_epoch(train_dataloader,
                                       tgt_vocab_size,
                                       gradient_clipping)
        val_loss = model.validate(val_dataloader,
                                  mode,
                                  target_vocab,
                                  tgt_vocab_size,
                                  max_tgt_length)
        end_time = timer()

        print(f"Epoch: {epoch} | Time = {(end_time - start_time):.3f}s")
        print(
            f'Train Loss: {train_loss:.3f} | Train Perplexity: {math.exp(train_loss):7.3f}')
        print(
            f'Validation Loss: {val_loss:.3f} | Validation Perplexity: {math.exp(val_loss):7.3f}')

        train_epoch_loss.append(train_loss)
        val_epoch_loss.append(val_loss)

        # We only want to store the model (and its checkpoints) if we're using 
        # the CPU (where gpu_rank is -1) or if we're using the GPU and the 
        # gpu_rank is 0 (to avoid having multiple processed storing the model 
        # when using multiple GPUs)
        if val_loss < best_val_loss and gpu_rank in [-1, 0]:
            print(f"Saving model with validation loss of {val_loss:7.3f}")
            best_val_loss = val_loss
            model.save()

        if checkpoint and gpu_rank in [-1, 0]:
            model.save_checkpoint(epoch, train_epoch_loss, val_epoch_loss)
    
    create_loss_plot(train_epoch_loss, val_epoch_loss, gpu_rank)


def create_train_model(gpu_rank,
                       world_size,
                       device,
                       src_vocab_size,
                       tgt_vocab_size,
                       d_model,
                       num_heads,
                       num_layers,
                       d_ff,
                       max_src_length,
                       max_tgt_length,
                       dropout,
                       learning_rate,
                       pad_idx,
                       num_epochs,
                       gradient_clipping,
                       label_smoothing,
                       mode,
                       source_vocab,
                       target_vocab,
                       checkpoint,
                       train_code_texts,
                       train_summary_texts,
                       val_code_texts,
                       val_summary_texts,
                       batch_size,
                       num_workers):
    '''
    Creates the training and validation dataloaders, the models and performs
    the training and validation of the model.

    Args:
        gpu_rank (int): The rank of the GPU.
                  It has the value of -1 if no GPUs are avaiable.
        world_size (int): The number of GPUs available in the machine.
                    It has the value of -1 if no GPUs are avaiable.
        device: The device where the model and tensors are inserted (GPU or CPU).
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
        learning_rate (int): Value of the initial learning rate.
        pad_idx (int): index of the <PAD> token
        num_epochs (int): The number of training epochs. 
        gradient_clipping (int): Maximum norm of the gradient.
        label_smoothing (int): Value of label smoothing to be applied in loss function.
        mode (string): Indicates whether we only want to compute validation loss or if we also
                       want to translate the source sentences in the validation set.
                       Can be one of the following: "loss", "translation" 
        source_vocab: The vocabulary built from the code snippets in training set.
        target_vocab: The vocabulary built from the summaries in training set.
        checkpoint (bool): Flag that tells whether we want to save a checkpoint of the model
                           at the end of each epoch or not. 
        train_code_texts: A list of code snippets examples belonging to the training set.
        train_summary_texts: A list of summaries examples belonging to the training set.
        val_code_texts: A list of code snippets examples belonging to the validation set.
        val_summary_texts: A list of summaries examples belonging to the validation set.
        batch_size (int): how many samples per batch to load
        num_workers (int): how many subprocesses to use for data loading.
    '''
    train_dataloader, val_dataloader = create_dataloaders(train_code_texts,
                                                          train_summary_texts,
                                                          val_code_texts,
                                                          val_summary_texts,
                                                          source_vocab,
                                                          target_vocab,
                                                          batch_size,
                                                          num_workers,
                                                          device,
                                                          max_src_length,
                                                          max_tgt_length,
                                                          world_size,
                                                          gpu_rank)

    if device == torch.device('cuda'):
        model_device = gpu_rank
    else:
        model_device = device

    model = Model(src_vocab_size,
                  tgt_vocab_size,
                  d_model,
                  num_heads,
                  num_layers,
                  d_ff,
                  max_src_length,
                  max_tgt_length,
                  dropout,
                  learning_rate,
                  label_smoothing,
                  pad_idx,
                  model_device,
                  gpu_rank)

    train_validate_model(model,
                         num_epochs,
                         train_dataloader,
                         val_dataloader,
                         tgt_vocab_size,
                         gradient_clipping,
                         mode,
                         target_vocab,
                         max_tgt_length,
                         checkpoint,
                         device,
                         gpu_rank)
