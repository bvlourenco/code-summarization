from datetime import datetime
import json
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from tqdm import tqdm
from evaluation.translation import greedy_decode, translate_tokens
from model.transformer.transformer_model import Transformer


class Model:
    '''
    Class that represents a model and common operations to them (training, 
    evaluation and other).
    '''

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model,
                 num_heads,
                 num_layers,
                 d_ff,
                 max_seq_length,
                 dropout,
                 learning_rate,
                 pad_idx,
                 device):
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
            learning_rate (int): Value of the learning rate.
            pad_idx (int): index of the <PAD> token
            device: The device where the model and tensors are inserted (GPU or CPU).
        '''
        self.model = Transformer(src_vocab_size,
                                 tgt_vocab_size,
                                 d_model,
                                 num_heads,
                                 num_layers,
                                 d_ff,
                                 max_seq_length,
                                 dropout,
                                 device)

        # Passing the model and all its layers to GPU if available
        self.model = self.model.to(device)

        # Function used to compute the loss. ignore_index is the padding token index
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        # TODO: Test with SGD optimizer
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=learning_rate,
                                    betas=(0.9, 0.98),
                                    eps=1e-9)

        self.device = device

    def train_epoch(self, train_dataloader, tgt_vocab_size, grad_clipping):
        '''
        Trains the model during one epoch, using the examples of the dataset for training.
        Computes the loss during this epoch.

        Args:
            train_dataloader: A Dataloader object that contains the training set.
            tgt_vocab_size (int): size of the target vocabulary.
            grad_clipping (int): Maximum norm of the gradient.

        Returns:
            The average loss across all examples during one epoch.

        Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
                https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
        '''
        # Set the model to training mode
        self.model.train()
        losses = 0

        for src, tgt in tqdm(train_dataloader, desc="Training"):
            # Passing vectors to GPU if it's available
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Zeroing the gradients of transformer parameters
            self.optimizer.zero_grad()

            # Slicing the last element of tgt_data because it is used to compute the
            # loss.
            output = self.model(src, tgt[:, :-1])

            # - output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
            # - tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
            # - criterion(prediction_labels, target_labels)
            # - tgt[:, 1:] removes the first token from the target sequence since we are training the
            # model to predict the next word based on previous words.
            loss = self.criterion(output.contiguous().view(-1, tgt_vocab_size),
                                  tgt[:, 1:].contiguous().view(-1))

            # Computing gradients of loss through backpropagation
            loss.backward()

            # clip the weights
            clip_grad_norm_(self.model.parameters(), grad_clipping)

            # Update the model's parameters based on the computed gradients
            self.optimizer.step()

            losses += loss.item()

        return losses / len(list(train_dataloader))

    def evaluate(self,
                 val_dataloader,
                 mode,
                 source_vocab,
                 target_vocab,
                 tgt_vocab_size,
                 max_seq_length):
        '''
        Validates the model after a training epoch to see how well he generalizes. 
        Does not update the weights of the model.

        Args:
            val_dataloader: A Dataloader object that contains the validation set.
            mode (string): Indicates whether we only want to compute validation loss or if we also
                           want to translate the source sentences in the validation set.
                           Can be one of the following: "loss", "translation"
            source_vocab: The vocabulary built from the code snippets in training set.
            target_vocab: The vocabulary built from the summaries in training set.
            tgt_vocab_size (int): size of the target vocabulary.
            max_seq_length (int): The maximum length of the summaries.

        Returns:
            The average loss across all examples during one epoch.

        Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
        '''
        # Set the model to evaluation mode
        self.model.eval()
        losses = 0

        # evaluate without updating gradients
        # Tells pytorch to not calculate the gradients
        with torch.no_grad():
            with open('../results/validation_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.json', 'w') as log:
                for src, tgt in tqdm(val_dataloader, desc="Validating"):
                    src = src.to(self.device)
                    tgt = tgt.to(self.device)

                    if mode == 'translation':
                        start_symbol_idx = target_vocab.token_to_idx['<BOS>']
                        end_symbol_idx = target_vocab.token_to_idx['<EOS>']

                        # Translating a batch of source sentences
                        for i in range(src.shape[0]):
                            # src[i] has shape (max_src_length, )
                            # Performing an unsqueeze on src[i] will make its shape (1, max_src_length)
                            # which is the correct shape since batch_size = 1 in this case
                            tgt_preds_idx = greedy_decode(self.model,
                                                          src[i].unsqueeze(0),
                                                          self.device,
                                                          start_symbol_idx,
                                                          end_symbol_idx,
                                                          max_seq_length)

                            # Passing the tensor to 1 dimension
                            tgt_preds_idx.flatten()

                            # Translating the indexes of tokens to the textual representation of tokens
                            # Replacing <BOS> and <EOS> with empty string
                            tgt_pred_tokens = translate_tokens(tgt_preds_idx,
                                                               target_vocab)

                            example = {}
                            example["prediction"] = tgt_pred_tokens
                            example["reference"] = translate_tokens(tgt[i],
                                                                    target_vocab)
                            example["code"] = translate_tokens(src[i],
                                                               source_vocab)

                            log.write(json.dumps(example, indent=4) + ',\n')

                    # Slicing the last element of tgt_data because it is used to compute the
                    # loss.
                    output = self.model(src, tgt[:, :-1])

                    # - output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
                    # - tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
                    # - criterion(prediction_labels, target_labels)
                    # - tgt[:, 1:] removes the first token from the target sequence since we are training the
                    # model to predict the next word based on previous words.
                    loss = self.criterion(output.contiguous().view(-1, tgt_vocab_size),
                                          tgt[:, 1:].contiguous().view(-1))

                    losses += loss.item()

        return losses / len(list(val_dataloader))

    def save(self):
        '''
        Stores the model learned parameters (weights) in a file.
        '''
        try:
            torch.save(self.model.state_dict(), '../results/model_weights.pth')
        except Exception:
            print("It was not possible to save the model weights. Continuing...")

    def save_checkpoint(self, epoch, training_losses, validation_losses):
        '''
        Stores a checkpoint of the model, the optimizer state, current epoch and
        training and validation losses from first epoch to current one.

        Args:
            epoch (int): Current epoch
            training_losses: List of training losses
            validation_losses: List of validation losses
        '''
        try:
            params = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_losses': training_losses,
                'validation_losses': validation_losses
            }
            torch.save(params, '../results/model_weights_checkpoint.pth')
        except Exception:
            print("It was not possible to save a model checkpoint. Continuing...")

    def load(self):
        '''
        Loads the model learned parameters from a saved file.
        '''
        try:
            self.model.load_state_dict(torch.load('../results/model_weights.pth'))
        except Exception:
            print("It was not possible to load the model weights from the \
                  specified filename. Continuing...")

    def load_checkpoint(self):
        '''
        Loads the last model checkpoint.

        Returns:
            epoch (int): Epoch of the checkpoint
            training_losses: List with the losses during training from first epoch
                             to epoch of the checkpoint
            validation_losses: List with the losses during validation from first epoch
                               to epoch of the checkpoint
        '''
        try:
            checkpoint = torch.load('../results/model_weights_checkpoint.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            training_losses = checkpoint['training_losses']
            validation_losses = checkpoint['validation_losses']
            epoch = checkpoint['epoch']
            return epoch, training_losses, validation_losses
        except Exception:
            print("It was not possible to load the model from the \
                   checkpoint. Continuing...")
