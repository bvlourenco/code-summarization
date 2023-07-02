import torch
from dataset.domain.custom_collate_fn import CustomCollate
from dataset.domain.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_dataloader(dataset,
                   source_vocab,
                   batch_size,
                   num_workers,
                   device,
                   max_src_length,
                   max_tgt_length, 
                   src_vocab_size,
                   tgt_vocab_size,
                   type,
                   world_size,
                   gpu_rank):
    '''
    Get a dataloader given a dataset.

    Args:
        dataset: The dataset from which we will build the dataloader.
                 Instance of CustomDataset.
        source_vocab: The vocabulary built from the code snippets in training set.
        batch_size (int): how many samples per batch to load
        num_workers (int): how many subprocesses to use for data loading.
        device: The device where the model and tensors are inserted (GPU or CPU).
        max_src_length (int): Maximum length of the source code.
        max_tgt_length (int): Maximum length of the summaries.
        src_vocab_size (int): size of the source vocabulary.
        tgt_vocab_size (int): size of the target vocabulary.
        type (string): Indicates whether we are loading the training set or the
                       validation set.
                       Can be one of the following: "train", "evaluation"
        world_size (int): The number of GPUs available in the machine.
                          It has the value of -1 if no GPUs are avaiable.
        gpu_rank (int): The rank of the GPU.
                        It has the value of None if no GPUs are avaiable or
                        only 1 GPU is available.

    Returns:
        A new dataloader.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''
    # pad_idx, bos_idx and eos_idx are the same between code snippet and summary
    pad_idx = source_vocab.token_to_idx['<PAD>']
    bos_idx = source_vocab.token_to_idx['<BOS>']
    eos_idx = source_vocab.token_to_idx['<EOS>']

    # A sampler tells the order in which the data from a dataset is accessed
    # and distributed across different processes and threads.
    if gpu_rank is not None:
        sampler = DistributedSampler(dataset, 
                                     num_replicas=world_size, 
                                     rank=gpu_rank,
                                     shuffle=False,
                                     drop_last=False)
    else:
        sampler = None

    # shuffle is disabled because we are running the model in multiple GPUs, hence
    # we want to split the data among the GPUs and avoid having repeated data on
    # multiple GPUs
    # A dataloader loads data samples in parallel from the dataset and applies 
    # transformations on the fly (like inserting <BOS> and <EOS> at the beggining and
    # end of sequences).
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=False,
                      drop_last=False,
                      pin_memory=False,
                      sampler=sampler,
                      collate_fn=CustomCollate(pad_idx, bos_idx, eos_idx, 
                                               device, max_src_length, 
                                               max_tgt_length, src_vocab_size, 
                                               tgt_vocab_size, type))


def create_dataloaders(source_code_texts,
                       summary_texts,
                       val_code_texts,
                       val_summary_texts,
                       source_vocab,
                       target_vocab,
                       batch_size,
                       num_workers,
                       device,
                       max_src_length,
                       max_tgt_length,
                       src_vocab_size,
                       tgt_vocab_size,
                       world_size,
                       gpu_rank):
    '''
    Creates a training and validation dataset and dataloaders.

    Args:
        source_code_texts: A list with size of training set containing code snippets.
        summary_texts: A list with the summaries of the training set.
        val_code_texts: A list with size of validation set containing code snippets.
        val_summary_texts: A list with size of validation set containing summaries.
        source_vocab: The vocabulary built from the code snippets in training set.
        target_vocab: The vocabulary built from the summaries in training set.
        batch_size (int): how many samples per batch to load
        num_workers (int):how many subprocesses to use for data loading.
        device: The device where the model and tensors are inserted (GPU or CPU).
        max_src_length (int): Maximum length of the source code.
        max_tgt_length (int): Maximum length of the summaries.
        src_vocab_size (int): size of the source vocabulary.
        tgt_vocab_size (int): size of the target vocabulary.
        world_size (int): The number of GPUs available in the machine.
                          It has the value of -1 if no GPUs are avaiable.
        gpu_rank (int): The rank of the GPU.
                        It has the value of None if no GPUs are avaiable or
                        only 1 GPU is available.

    Returns:
        The training and validation dataloaders (instances of Dataloader).

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''
    train_dataset = CustomDataset(source_code_texts,
                                  summary_texts,
                                  source_vocab,
                                  target_vocab,
                                  'train',
                                  device)
    train_dataloader = get_dataloader(train_dataset,
                                      source_vocab,
                                      batch_size,
                                      num_workers,
                                      device,
                                      max_src_length,
                                      max_tgt_length,
                                      src_vocab_size,
                                      tgt_vocab_size,
                                      'train',
                                      world_size,
                                      gpu_rank)

    val_dataloader = load_evaluation_dataloader(val_code_texts,
                                                val_summary_texts,
                                                source_vocab,
                                                target_vocab,
                                                batch_size,
                                                num_workers,
                                                device,
                                                max_src_length,
                                                max_tgt_length,
                                                src_vocab_size,
                                                tgt_vocab_size,
                                                world_size,
                                                gpu_rank)
    return train_dataloader, val_dataloader


def load_evaluation_dataloader(code_texts,
                               summary_texts,
                               source_vocab,
                               target_vocab,
                               batch_size,
                               num_workers,
                               device,
                               max_src_length,
                               max_tgt_length,
                               src_vocab_size,
                               tgt_vocab_size,
                               world_size,
                               gpu_rank):
    '''
    Creates a dataloader for the validation set or the testing set.

    Args:
        source_code_texts: A list with size of validation/testing set containing code snippets.
        summary_texts: A list with the summaries of the validation/testing  set.
        source_vocab: The vocabulary built from the code snippets in training set.
        target_vocab: The vocabulary built from the summaries in training set.
        batch_size (int): how many samples per batch to load
        num_workers (int):how many subprocesses to use for data loading.
        device: The device where the model and tensors are inserted (GPU or CPU).
        max_src_length (int): Maximum length of the source code.
        max_tgt_length (int): Maximum length of the summaries.
        src_vocab_size (int): size of the source vocabulary.
        tgt_vocab_size (int): size of the target vocabulary.
        world_size (int): The number of GPUs available in the machine.
                          It has the value of -1 if no GPUs are avaiable.
        gpu_rank (int): The rank of the GPU.
                        It has the value of None if no GPUs are avaiable or
                        only 1 GPU is available.
        
    Returns:
        A new dataloader with the validation or testing set.
    '''
    evaluation_dataset = CustomDataset(code_texts,
                                       summary_texts,
                                       source_vocab,
                                       target_vocab,
                                       'evaluation',
                                       device)
    return get_dataloader(evaluation_dataset,
                          source_vocab,
                          batch_size,
                          num_workers,
                          device,
                          max_src_length,
                          max_tgt_length,
                          src_vocab_size,
                          tgt_vocab_size,
                          'evaluation',
                          world_size,
                          gpu_rank)
