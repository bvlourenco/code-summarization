import math
import os
import torch
from dataset.build_vocab import create_vocabulary
from dataset.dataloader import create_dataloaders
from dataset.load_dataset import load_dataset_file
from evaluation.graphs import create_loss_plot
from model.model import Model
from train_test.program import Program
from timeit import default_timer as timer


class TrainProgram(Program):
    '''
    TODO
    '''

    def __init__(self, args, trial_number=None):
        '''
        TODO
        '''
        super(TrainProgram, self).__init__(args, trial_number)
        self.num_epochs = args.num_epochs
        self.gradient_clipping = args.gradient_clipping
        self.freq_threshold = args.freq_threshold
        self.train_filename = args.train_filename
        self.validation_filename = args.validation_filename
        self.checkpoint = args.checkpoint

    def execute_operation(self, gpu_rank=None):
        '''
        TODO
        '''
        train_code_texts, train_summary_texts = load_dataset_file(self.train_filename,
                                                                  'train',
                                                                  self.debug_max_lines)

        val_code_texts, val_summary_texts = load_dataset_file(self.validation_filename,
                                                              'validation',
                                                              self.debug_max_lines)

        source_vocab, target_vocab = create_vocabulary(train_code_texts,
                                                       train_summary_texts,
                                                       self.freq_threshold,
                                                       self.src_vocab_size,
                                                       self.tgt_vocab_size)

        train_dataloader, val_dataloader = create_dataloaders(train_code_texts,
                                                              train_summary_texts,
                                                              val_code_texts,
                                                              val_summary_texts,
                                                              source_vocab,
                                                              target_vocab,
                                                              self.batch_size,
                                                              self.num_workers,
                                                              self.device,
                                                              self.max_src_length,
                                                              self.max_tgt_length,
                                                              self.world_size,
                                                              gpu_rank)

        if self.device == torch.device('cuda'):
            model_device = gpu_rank
        else:
            model_device = self.device

        model = Model(self.src_vocab_size,
                      self.tgt_vocab_size,
                      self.d_model,
                      self.num_heads,
                      self.num_layers,
                      self.d_ff,
                      self.max_src_length,
                      self.max_tgt_length,
                      self.dropout,
                      self.learning_rate,
                      self.label_smoothing,
                      source_vocab.token_to_idx['<PAD>'],
                      model_device,
                      gpu_rank)

        self.train_validate_model(model,
                                  self.num_epochs,
                                  train_dataloader,
                                  val_dataloader,
                                  self.tgt_vocab_size,
                                  self.gradient_clipping,
                                  self.mode,
                                  self.beam_size,
                                  target_vocab,
                                  self.max_tgt_length,
                                  self.checkpoint,
                                  self.device,
                                  gpu_rank,
                                  self.trial_number)

    def train_validate_model(self,
                             model,
                             num_epochs,
                             train_dataloader,
                             val_dataloader,
                             tgt_vocab_size,
                             gradient_clipping,
                             mode,
                             beam_size,
                             target_vocab,
                             max_tgt_length,
                             checkpoint,
                             device,
                             gpu_rank,
                             trial_number):
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
                        want to translate the source sentences in the validation set
                        (either using greedy decoding or beam search).
                        Can be one of the following: "loss", "greedy" or "beam"
            beam_size (int): Number of elements to store during beam search
                            Only applicable if `mode == 'beam'`
            source_vocab: The vocabulary built from the code snippets in training set.
            target_vocab: The vocabulary built from the summaries in training set.
            max_tgt_length (int): Maximum length of the generated summaries.
            checkpoint (bool): Flag that tells whether we want to save a checkpoint of the model
                            at the end of each epoch or not.
            device: The device where the model and tensors are inserted (GPU or CPU).
            gpu_rank (int): The rank of the GPU.
                            It has the value of -1 if no GPUs are avaiable.
            trial_number (int): If we are fine-tuning the parameters of the program, it is the
                                number of trial that are we are doing using optuna. 
                                Otherwise, it is None.

        Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
        '''

        # Load model checkpoint (loading model parameters and optimizer state)
        # if the checkpoint exists
        if os.path.isfile('../results/model_weights_checkpoint.pth'):
            start_epoch, train_epoch_loss, val_epoch_loss = model.load_checkpoint(
                gpu_rank)
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
                                      max_tgt_length,
                                      epoch,
                                      beam_size)
            end_time = timer()

            print(f"Epoch: {epoch} | Time = {(end_time - start_time):.3f}s")
            print(
                f'Train Loss: {train_loss:.3f} | Train Perplexity: {math.exp(train_loss):7.3f}')
            print(
                f'Validation Loss: {val_loss:.3f} | Validation Perplexity: {math.exp(val_loss):7.3f}')

            train_epoch_loss.append(train_loss)
            val_epoch_loss.append(val_loss)

            # We only want to store the model (and its checkpoints) if we're using
            # the CPU (where gpu_rank is None) or if we're using the GPU and the
            # gpu_rank is 0 (to avoid having multiple processed storing the model
            # when using multiple GPUs)
            if val_loss < best_val_loss and gpu_rank in [None, 0]:
                print(f"Saving model with validation loss of {val_loss:7.3f}")
                best_val_loss = val_loss
                model.save()

            if checkpoint and gpu_rank in [None, 0]:
                model.save_checkpoint(epoch, train_epoch_loss, val_epoch_loss)

        # Only create graph if we're not doing hyperparameter fine-tuning
        if num_epochs > 1 and trial_number is None:
            create_loss_plot(train_epoch_loss, val_epoch_loss, gpu_rank)
        else:
            print(f"Training loss of epoch 1: {train_epoch_loss[0]}\n" +
                  f"Validation loss of epoch 1: {val_epoch_loss[0]}")

        if trial_number is not None:
            with open('../results/loss_file', 'a') as loss_file:
                loss_file.write(str(trial_number) + ' ' +
                                str(val_epoch_loss[-1]) + '\n')
