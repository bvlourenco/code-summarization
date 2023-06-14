from dataset.domain.custom_collate_fn import MyCollate
from dataset.load_dataset import load_dataset_file
from dataset.domain.training_dataset import TrainDataset
from dataset.domain.validation_dataset import ValidationDataset
from torch.utils.data import DataLoader


def get_dataloader(dataset, source_vocab, batch_size, num_workers, device, max_seq_length):
    '''
    Get a dataloader given a dataset.

    Args:
        dataset: The dataset from which we will build the dataloader.
                 Instance of TrainDataset or ValidationDataset.
        source_vocab: The vocabulary built from the code snippets in training set.
        batch_size (int): how many samples per batch to load
        num_workers (int): how many subprocesses to use for data loading.
        device: The device where the model and tensors are inserted (GPU or CPU).
        max_seq_length (int): Maximum length of the source code and summary.

    Returns:
        A new dataloader.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''
    # pad_idx, bos_idx and eos_idx are the same between code snippet and summary
    pad_idx = source_vocab.token_to_idx['<PAD>']
    bos_idx = source_vocab.token_to_idx['<BOS>']
    eos_idx = source_vocab.token_to_idx['<EOS>']
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=True,
                      drop_last=True,
                      pin_memory=False,
                      collate_fn=MyCollate(pad_idx, bos_idx, eos_idx, device, max_seq_length))


def create_dataloaders(source_code_texts,
                       summary_texts,
                       source_vocab,
                       target_vocab,
                       val_code_filename,
                       val_summary_filename,
                       batch_size,
                       num_workers,
                       device,
                       max_seq_length,
                       debug_max_lines):
    '''
    Creates a training and validation dataset and dataloaders.

    Args:
        source_code_texts: A list with size of training set containing code snippets.
        summary_texts: A list with the summaries of the training set.
        source_vocab: The vocabulary built from the code snippets in training set.
        target_vocab: The vocabulary built from the summaries in training set.
        val_code_filename: The validation set filename with code snippets.
        val_summary_filename: The validation set filename with summaries.
        batch_size (int): how many samples per batch to load
        num_workers (int):how many subprocesses to use for data loading.
        device: The device where the model and tensors are inserted (GPU or CPU).
        max_seq_length (int): Maximum length of the source code and summary.
        debug_max_lines (int): Represents the number of examples we want to read
                               from the dataset. If we pass a non-positive value, 
                               the whole dataset will be read.

    Returns:
        The training and validation dataloaders (instances of Dataloader).

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''
    train_dataset = TrainDataset(
        source_code_texts, summary_texts, source_vocab, target_vocab)
    train_dataloader = get_dataloader(train_dataset,
                                      source_vocab,
                                      batch_size,
                                      num_workers,
                                      device,
                                      max_seq_length)

    # Loading the validation set
    val_code_texts, val_summary_texts = load_dataset_file(val_code_filename,
                                                          val_summary_filename,
                                                          'validation',
                                                          debug_max_lines)

    validation_dataset = ValidationDataset(
        val_code_texts, val_summary_texts, source_vocab, target_vocab)
    val_dataloader = get_dataloader(validation_dataset,
                                    source_vocab,
                                    batch_size,
                                    num_workers,
                                    device,
                                    max_seq_length)
    return train_dataloader, val_dataloader
