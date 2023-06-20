from abc import abstractmethod
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


class Program:
    '''
    TODO
    '''

    def __init__(self, args, trial_number=None):
        '''
        TODO
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

        self.mode = args.mode
        self.beam_size = args.beam_size

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # world_size is the number of processes to be launched
        self.world_size = None
        if torch.cuda.is_available():
            # If we're running in GPU, we'll have 1 process per GPU
            self.world_size = torch.cuda.device_count()

        self.trial_number = trial_number

    def start(self):
        '''
        TODO
        '''
        if self.world_size is not None:
            self.run_parallel()
        else:
            self.execute_operation()

    def run_parallel(self):
        '''
        TODO
        '''
        mp.spawn(self.init_multiple_processes,
                 args=(),
                 nprocs=self.world_size,
                 join=True)

    @abstractmethod
    def execute_operation(self, gpu_rank=None):
        '''
        TODO
        '''
        pass

    def init_multiple_processes(self, gpu_rank):
        '''
        TODO
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
