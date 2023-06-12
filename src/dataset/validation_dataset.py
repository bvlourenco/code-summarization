import os
import torch
from tqdm import tqdm


class ValidationDataset:
    '''
    Custom validation dataset used to load the dataset through a dataloader.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''

    def __init__(self, code_filename, summary_filename, train_dataset):
        '''
        Args:
            code_filename (string): The name of the file where the code snippets are
            summary_filename (string): The name of the file where the summaries are
            train_dataset (TrainDataset): The custom training dataset
        '''
        if (not os.path.exists(code_filename)):
            raise ValueError("code snippet filename does not exist")
        
        if (not os.path.exists(summary_filename)):
            raise ValueError("summary filename does not exist")
        
        # train dataset will be used as lookup for vocab
        self.train_dataset = train_dataset

        with open(code_filename) as code_file:
            num_code_lines = sum(1 for _ in open(code_filename, 'r'))
            self.source_code_texts = [line.strip() for line in tqdm(
                code_file, total=num_code_lines, desc="Reading train code snippets")]

        with open(summary_filename) as summary_file:
            num_summary_lines = sum(1 for _ in open(summary_filename, 'r'))
            self.summary_texts = [line.strip() for line in tqdm(
                summary_file, total=num_summary_lines, desc="Reading train summaries")]
        
        assert num_code_lines == num_summary_lines
        self.dataset_size = num_code_lines

    def __len__(self):
        '''
        Returns the number of samples in our dataset.
        '''
        return self.dataset_size

    def __getitem__(self, index):
        '''
        __getitem__ runs on 1 example at a time. Here, we get an example at index 
        and return its numericalize source and target values using the training 
        vocabulary

        Args:
            index (int): index of the example we want to numericalize
        
        Returns:
            Two tensors with the source code and summary translated to numbers
        '''
        source_text = self.source_code_texts[index]
        target_text = self.summary_texts[index]

        # numericalize texts ['<BOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        numerialized_source = [
            self.train_dataset.source_vocab.token_to_idx["<BOS>"]]
        numerialized_source += self.train_dataset.source_vocab.numericalize(
            source_text)
        numerialized_source.append(
            self.train_dataset.source_vocab.token_to_idx["<EOS>"])

        numerialized_target = [
            self.train_dataset.target_vocab.token_to_idx["<BOS>"]]
        numerialized_target += self.train_dataset.target_vocab.numericalize(
            target_text)
        numerialized_target.append(
            self.train_dataset.target_vocab.token_to_idx["<EOS>"])
        return torch.tensor(numerialized_source), torch.tensor(numerialized_target)
