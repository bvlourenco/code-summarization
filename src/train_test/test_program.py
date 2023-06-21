import torch
from dataset.build_vocab import load_vocab
from dataset.dataloader import load_evaluation_dataloader
from dataset.load_dataset import load_dataset_file
from model.model import Model
from train_test.program import Program


class TestProgram(Program):
    '''
    Represents a program where we're loading the vocabulary and the model and
    testing it using the testing set.
    '''

    def __init__(self, args):
        '''
        Args:
            args: The arguments given to the program.
        '''
        super(TestProgram, self).__init__(args)
        self.test_filename = args.test_filename

    def execute_operation(self, gpu_rank=None):
        '''
        Executes the operation necessary to test the model:
        (1) Load the testing set
        (2) Load the vocabulary of the training set
        (3) Create a dataloader for the testing set
        (4) Create the model
        (5) Load the weights of the trained model 
        (6) Test the model

        Args:
            gpu_rank: The rank of the GPU.
                      It has the value of None if no GPUs are available or
                      only 1 GPU is available.
        '''
        test_code_texts, test_summary_texts = load_dataset_file(self.test_filename,
                                                                'test',
                                                                self.debug_max_lines)

        source_vocab, target_vocab = load_vocab('../results/src_vocab.pkl',
                                                '../results/tgt_vocab.pkl')

        test_dataloader = load_evaluation_dataloader(test_code_texts,
                                                     test_summary_texts,
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
                      gpu_rank)

        model.load(gpu_rank)
        model.test(test_dataloader,
                   target_vocab,
                   self.max_tgt_length,
                   self.mode,
                   self.beam_size)