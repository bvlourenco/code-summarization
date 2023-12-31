import logging
import math
import os
from dataset.build_vocab import create_vocabulary
from dataset.dataloader import create_dataloaders
from dataset.load_dataset import load_dataset_file, load_local_matrices, load_matrices
from evaluation.graphs import create_loss_plot
from model.model import Model
from train_test.program import Program
from timeit import default_timer as timer

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')
# To avoid having repeated logs!
logger.propagate = False


class TrainProgram(Program):
    '''
    Represents a program where we're loading the training and validation set,
    creating the vocabulary with the training set and then training and 
    validating the model.
    '''

    def __init__(self, args, trial_number=None):
        '''
        Args:
            args: The arguments passed to the program.
            trial_number: If we're fine-tuning the model, it is the number of
                          trial of the optuna study (each trial corresponds to a
                          different parameter configuration).
        '''
        super(TrainProgram, self).__init__(args)
        self.num_epochs = args.num_epochs
        self.freq_threshold = args.freq_threshold
        self.train_filename = args.train_filename
        self.validation_filename = args.validation_filename
        self.checkpoint = args.checkpoint

        self.train_token_matrix = args.train_token_matrix
        self.train_statement_matrix = args.train_statement_matrix
        self.train_data_flow_matrix = args.train_data_flow_matrix
        self.train_control_flow_matrix = args.train_control_flow_matrix

        self.validation_token_matrix = args.validation_token_matrix
        self.validation_statement_matrix = args.validation_statement_matrix
        self.validation_data_flow_matrix = args.validation_data_flow_matrix
        self.validation_control_flow_matrix = args.validation_control_flow_matrix

        self.trial_number = trial_number

    def execute_operation(self, gpu_rank=None):
        '''
        Executes the operation necessary to test the model:
        (1) Load the training set
        (2) Load the validation set
        (3) Create the vocabulary from the training set
        (3) Create the dataloaders for the training set and validation set
        (4) Create the model
        (5) Train the model for some epochs and validate it after each epoch

        Args:
            gpu_rank: The rank of the GPU.
                      It has the value of None if no GPUs are available or
                      only 1 GPU is available.
        '''
        train_code_texts, \
            train_code_tokens, \
            train_summary_texts, \
            train_summary_tokens = load_dataset_file(self.train_filename,
                                                     'train',
                                                     self.debug_max_lines)

        val_code_texts, \
            val_code_tokens, \
            val_summary_texts, \
            val_summary_tokens = load_dataset_file(self.validation_filename,
                                                   'validation',
                                                   32)

        train_token_matrices, train_statement_matrices, \
            train_data_flow_matrices, train_control_flow_matrices \
                 = load_matrices(self.train_token_matrix,
                                 self.train_statement_matrix,
                                 self.train_data_flow_matrix,
                                 self.train_control_flow_matrix,
                                 'train',
                                 self.debug_max_lines)

        val_token_matrices, val_statement_matrices, \
            val_data_flow_matrices, val_control_flow_matrices, \
             = load_matrices(self.validation_token_matrix,
                             self.validation_statement_matrix,
                             self.validation_data_flow_matrix,
                             self.validation_control_flow_matrix,
                             'validation',
                             self.debug_max_lines)

        source_vocab, target_vocab = create_vocabulary(train_code_tokens,
                                                       train_summary_tokens,
                                                       self.freq_threshold,
                                                       self.src_vocab_size,
                                                       self.tgt_vocab_size,
                                                       self.dir_iteration)

        train_dataloader, val_dataloader = create_dataloaders(train_code_texts,
                                                              train_code_tokens,
                                                              train_summary_texts,
                                                              train_summary_tokens,
                                                              train_token_matrices,
                                                              train_statement_matrices,
                                                              train_data_flow_matrices,
                                                              train_control_flow_matrices,
                                                              val_code_texts,
                                                              val_code_tokens,
                                                              val_summary_texts,
                                                              val_summary_tokens,
                                                              val_token_matrices,
                                                              val_statement_matrices,
                                                              val_data_flow_matrices,
                                                              val_control_flow_matrices,
                                                              source_vocab,
                                                              target_vocab,
                                                              self.batch_size,
                                                              self.num_workers,
                                                              self.device,
                                                              self.max_src_length,
                                                              self.max_tgt_length,
                                                              self.world_size,
                                                              gpu_rank)

        if self.world_size is not None:
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
                      gpu_rank,
                      self.init_type,
                      self.optimizer,
                      self.hyperparameter_hsva,
                      self.hyperparameter_data_flow,
                      self.hyperparameter_control_flow,
                      self.dir_iteration)

        self.train_validate_model(model,
                                  train_dataloader,
                                  val_dataloader,
                                  target_vocab,
                                  gpu_rank)

    def train_validate_model(self,
                             model,
                             train_dataloader,
                             val_dataloader,
                             target_vocab,
                             gpu_rank):
        '''
        Performs the model training. In each epoch, it is done an evaluation 
        using the validation set. In the end, it plots a graph with the training 
        and validation loss.

        Args:
            model: The model (an instance of Transformer). 
            train_dataloader: A Dataloader object that contains the training set.
            val_dataloader: A Dataloader object that contains the validation set.
            target_vocab: The vocabulary built from the summaries in training set.
            gpu_rank (int): The rank of the GPU.
                            It has the value of None if no GPUs are avaiable or
                            only 1 GPU is available.

        Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
        '''

        # Load model checkpoint (loading model parameters and optimizer state)
        # if the checkpoint exists
        if os.path.isfile('../results/' + self.dir_iteration + '/model_weights_checkpoint.pth'):
            start_epoch, train_epoch_loss, \
                val_epoch_loss, best_bleu = model.load_checkpoint(gpu_rank)
            best_val_loss = min(val_epoch_loss)
            start_epoch += 1
        else:
            train_epoch_loss, val_epoch_loss = [], []
            start_epoch = 1
            best_val_loss = float('inf')
            best_bleu = 0

        for epoch in range(start_epoch, self.num_epochs + 1):
            logger.info(f"Epoch: {epoch}")

            if gpu_rank is not None:
                train_dataloader.sampler.set_epoch(epoch)
                val_dataloader.sampler.set_epoch(epoch)

            start_time = timer()
            train_loss = model.train_epoch(train_dataloader)
            val_loss, bleu = model.validate(val_dataloader,
                                            self.mode,
                                            target_vocab,
                                            epoch,
                                            self.beam_size)
            end_time = timer()

            logger.info(
                f"Epoch: {epoch} | Time = {(end_time - start_time):.3f}s")
            logger.info(f'Train Loss: {train_loss:.3f} | ' +
                        f'Train Perplexity: {math.exp(train_loss):7.3f}')
            logger.info(f'Validation Loss: {val_loss:.3f} | ' +
                        f'Validation Perplexity: {math.exp(val_loss):7.3f}')

            train_epoch_loss.append(train_loss)
            val_epoch_loss.append(val_loss)

            # We only want to store the model (and its checkpoints) if we're using
            # the CPU (where gpu_rank is None) or if we're using the GPU and the
            # gpu_rank is 0 (to avoid having multiple processed storing the model
            # when using multiple GPUs)
            if epoch == self.num_epochs and gpu_rank in [None, 0]:
                logger.info(f"Saving last epoch of model with loss of {val_loss:7.3f}")
                model.save(last_epoch=True)
            elif bleu is not None and bleu > best_bleu and gpu_rank in [None, 0]:
                logger.info(
                    f"Saving model with BLEU metric of {bleu:7.3f}")
                best_bleu = bleu
                model.save()
            elif bleu is None and val_loss < best_val_loss and gpu_rank in [None, 0]:
                logger.info(
                    f"Saving model with validation loss of {val_loss:7.3f}")
                best_val_loss = val_loss
                model.save()

            if self.checkpoint and gpu_rank in [None, 0]:
                model.save_checkpoint(epoch, train_epoch_loss, val_epoch_loss, best_bleu)

        # Only create graph if we're not doing hyperparameter fine-tuning
        if self.num_epochs > 1 and self.trial_number is None:
            create_loss_plot(train_epoch_loss, val_epoch_loss, gpu_rank, self.dir_iteration)
        else:
            logger.info(f"Training loss of epoch 1: {train_epoch_loss[0]}")
            logger.info(f"Validation loss of epoch 1: {val_epoch_loss[0]}")

        if self.trial_number is not None:
            with open('../results/' + self.dir_iteration + '/loss_file', 'a') as loss_file:
                loss_file.write(str(self.trial_number) + ' ' +
                                str(val_epoch_loss[-1]) + '\n')
