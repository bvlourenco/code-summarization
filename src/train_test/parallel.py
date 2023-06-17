import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from train_test.test_model import test_model
from train_test.train import create_train_model


def train_model_parallel(gpu_rank,
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
    Initializes the process group and then creates and trains the model.

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

    Source: https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
            https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#id1
            https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    '''

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=gpu_rank
    )
    torch.cuda.set_device(gpu_rank)

    create_train_model(gpu_rank,
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
                       num_workers)

    dist.destroy_process_group()


def test_model_parallel(gpu_rank,
                        world_size,
                        test_code_texts,
                        test_summary_texts,
                        source_vocab,
                        target_vocab,
                        batch_size,
                        num_workers,
                        device,
                        max_src_length,
                        max_tgt_length,
                        src_vocab_size,
                        tgt_vocab_size,
                        d_model,
                        num_heads,
                        num_layers,
                        d_ff,
                        dropout,
                        learning_rate,
                        label_smoothing):
    '''
    Initializes the process group and tests the model.

    Args:
        gpu_rank (int): The rank of the GPU.
                        It has the value of -1 if no GPUs are avaiable.
        world_size (int): The number of GPUs available in the machine.
                          It has the value of -1 if no GPUs are avaiable.
        test_code_texts: A list of code snippets examples belonging to the testing set.
        test_summary_texts: A list of summaries examples belonging to the testing set.
        source_vocab: The vocabulary built from the code snippets in training set.
        target_vocab: The vocabulary built from the summaries in training set.
        batch_size (int): how many samples per batch to load
        num_workers (int): how many subprocesses to use for data loading.
        device: The device where the model and tensors are inserted (GPU or CPU).
        max_src_length (int): maximum length of the source code.
        max_tgt_length (int): maximum length of the summaries.
        src_vocab_size (int): size of the source vocabulary.
        tgt_vocab_size (int): size of the target vocabulary.
        d_model (int): dimensionality of the model.
        num_heads (int): number of heads of the Multi Head Attention.
        num_layers (int): number of encoder and decoder layers.
        d_ff (int): the hidden layer size of the second-layer of the Feed
                    Forward Network (in encoder and decoder).
        dropout (int): dropout probability (between 0 and 1).
        learning_rate (int): Value of the initial learning rate.
        label_smoothing (int): Value of label smoothing to be applied in 
                               loss function.
    '''
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=gpu_rank
    )
    torch.cuda.set_device(gpu_rank)

    test_model(test_code_texts,
               test_summary_texts,
               source_vocab,
               target_vocab,
               batch_size,
               num_workers,
               device,
               max_src_length,
               max_tgt_length,
               world_size,
               gpu_rank,
               src_vocab_size,
               tgt_vocab_size,
               d_model,
               num_heads,
               num_layers,
               d_ff,
               dropout,
               learning_rate,
               label_smoothing)

    dist.destroy_process_group()


def train_parallel(world_size,
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
                   num_workers
                   ):
    '''
    Creates several processes used to run the model. The dataset is split among
    the processes.

    Args:
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

    Source: https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
            https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#id1
            https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    '''
    mp.spawn(train_model_parallel,
             args=(world_size,
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
                   num_workers
                   ),
             nprocs=world_size,
             join=True)


def test_parallel(world_size,
                  test_code_texts,
                  test_summary_texts,
                  source_vocab,
                  target_vocab,
                  batch_size,
                  num_workers,
                  device,
                  max_src_length,
                  max_tgt_length,
                  src_vocab_size,
                  tgt_vocab_size,
                  d_model,
                  num_heads,
                  num_layers,
                  d_ff,
                  dropout,
                  learning_rate,
                  label_smoothing):
    '''
    Creates several processes used to test the model. The dataset is split among
    the processes.

    Args:
        world_size (int): The number of GPUs available in the machine.
                          It has the value of -1 if no GPUs are avaiable.
        test_code_texts: A list of code snippets examples belonging to the testing set.
        test_summary_texts: A list of summaries examples belonging to the testing set.
        source_vocab: The vocabulary built from the code snippets in training set.
        target_vocab: The vocabulary built from the summaries in training set.
        batch_size (int): how many samples per batch to load
        num_workers (int): how many subprocesses to use for data loading.
        device: The device where the model and tensors are inserted (GPU or CPU).
        max_src_length (int): maximum length of the source code.
        max_tgt_length (int): maximum length of the summaries.
        src_vocab_size (int): size of the source vocabulary.
        tgt_vocab_size (int): size of the target vocabulary.
        d_model (int): dimensionality of the model.
        num_heads (int): number of heads of the Multi Head Attention.
        num_layers (int): number of encoder and decoder layers.
        d_ff (int): the hidden layer size of the second-layer of the Feed
                    Forward Network (in encoder and decoder).
        dropout (int): dropout probability (between 0 and 1).
        learning_rate (int): Value of the initial learning rate.
        label_smoothing (int): Value of label smoothing to be applied in 
                               loss function.
    '''
    mp.spawn(test_model_parallel,
             args=(world_size,
                   test_code_texts,
                   test_summary_texts,
                   source_vocab,
                   target_vocab,
                   batch_size,
                   num_workers,
                   device,
                   max_src_length,
                   max_tgt_length,
                   src_vocab_size,
                   tgt_vocab_size,
                   d_model,
                   num_heads,
                   num_layers,
                   d_ff,
                   dropout,
                   learning_rate,
                   label_smoothing
                   ),
             nprocs=world_size,
             join=True)
