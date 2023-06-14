import math
import os
from timeit import default_timer as timer
from evaluation.graphs import create_loss_plot


def train_validate_model(model,
                         num_epochs,
                         train_dataloader,
                         val_dataloader,
                         tgt_vocab_size,
                         gradient_clipping,
                         mode,
                         source_vocab,
                         target_vocab,
                         max_seq_length,
                         checkpoint):
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
        max_seq_length (int): Maximum length of the source code and summary.
        checkpoint (bool): Flag that tells whether we want to save a checkpoint of the model
                           at the end of each epoch or not.

    Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
    '''
    
    # Load model checkpoint (loading model parameters and optimizer state) 
    # if the checkpoint exists
    if os.path.isfile('../results/model_weights_checkpoint.pth'):
        start_epoch, train_epoch_loss, val_epoch_loss = model.load_checkpoint()
        start_epoch += 1
    else:
        train_epoch_loss, val_epoch_loss = [], []
        start_epoch = 1

    best_val_loss = float('inf')
    for epoch in range(start_epoch, num_epochs + 1):
        start_time = timer()
        train_loss = model.train_epoch(train_dataloader,
                                       tgt_vocab_size,
                                       gradient_clipping)
        val_loss = model.evaluate(val_dataloader,
                                  mode,
                                  source_vocab,
                                  target_vocab,
                                  tgt_vocab_size,
                                  max_seq_length)
        end_time = timer()

        print(f"Epoch: {epoch} | Time = {(end_time - start_time):.3f}s")
        print(
            f'Train Loss: {train_loss:.3f} | Train Perplexity: {math.exp(train_loss):7.3f}')
        print(
            f'Validation Loss: {val_loss:.3f} | Validation Perplexity: {math.exp(val_loss):7.3f}')

        train_epoch_loss.append(train_loss)
        val_epoch_loss.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save()

        if checkpoint:
            model.save_checkpoint(epoch, train_epoch_loss, val_epoch_loss)

    create_loss_plot(train_epoch_loss, val_epoch_loss)
