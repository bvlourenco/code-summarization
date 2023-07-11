from abc import abstractmethod
import json
import logging
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

# Multiple calls to getLogger() with the same name will return a reference
# to the same Logger object, which saves us from passing the logger objects
# to every part where it’s needed.
# Source: https://realpython.com/python-logging/
logger = logging.getLogger('main_logger')
# To avoid having repeated logs!
logger.propagate = False

class Program:
    '''
    Class that represents an instance of a program. It is only used as a 
    superclass of the classes `TrainProgram` and `TestProgram`.
    '''

    def __init__(self, args):
        '''
        Args:
            args: The arguments given to the program.
        '''
        self.src_vocab_size = args.src_vocab_size
        self.tgt_vocab_size = args.tgt_vocab_size
        self.max_src_length = args.max_src_length
        self.max_tgt_length = args.max_tgt_length
        self.debug_max_lines = args.debug_max_lines

        self.d_model = args.d_model
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.dropout = args.dropout

        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.label_smoothing = args.label_smoothing
        self.init_type = args.init_type

        self.hyperparameter_hsva = args.hyperparameter_hsva
        self.hyperparameter_data_flow = args.hyperparameter_data_flow
        self.hyperparameter_control_flow = args.hyperparameter_control_flow
        self.hyperparameter_ast = args.hyperparameter_ast

        self.mode = args.mode
        self.beam_size = args.beam_size

        self.optimizer = args.optimizer

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # world_size is the number of processes to be launched
        self.world_size = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # If we're running in GPU, we'll have 1 process per GPU
            self.world_size = torch.cuda.device_count()

        # Storing all arguments to be print at the start of the program
        self.args = args

    def start(self):
        '''
        Starts the program by checking if we're going to launch only 1 process
        or several processes (depending on the number of GPUs being used. We have
        1 process per GPU).
        '''
        logger.info("Arguments:\n%s" % json.dumps(vars(self.args),
                                                  indent=4,
                                                  sort_keys=True))
        if self.world_size is not None:
            self.run_parallel()
        else:
            self.execute_operation()

    def run_parallel(self):
        '''
        Starts multiple processes of the program. The number of processes is the
        number of GPUs available to the program. 
        '''
        mp.spawn(self.init_multiple_processes,
                 args=(),
                 nprocs=self.world_size,
                 join=True)

    @abstractmethod
    def execute_operation(self, gpu_rank=None):
        '''
        Executes the operation of the program: it either trains and validates the
        model or it tests it. It depends on the class we've created (if it's a 
        `TrainProgram` or `TestProgram` class).

        Args:
            gpu_rank (int): The rank of the GPU.
                            It has the value of None if no GPUs are available or
                            only 1 GPU is available.
        '''
        raise NotImplementedError

    def init_multiple_processes(self, gpu_rank):
        '''
        If we're running multiple processes of this program, it initializes all
        of those processes and executes the operation that the program does.

        Args:
            gpu_rank (int): The rank of the GPU.
        '''
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(
            backend='nccl',
            world_size=self.world_size,
            rank=gpu_rank
        )
        torch.cuda.set_device(gpu_rank)

        self.execute_operation(gpu_rank)

        dist.destroy_process_group()
