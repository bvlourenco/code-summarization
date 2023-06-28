import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    '''
    Custom dataset used to load the dataset through a dataloader.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''

    def __init__(self, source_code_texts, summary_texts, source_vocab, target_vocab, type, device):
        '''
        Args:
            source_code_texts: A list containing code snippets.
            summary_texts: A list containing summaries.
            source_vocab: The vocabulary built from the code snippets in training set.
            target_vocab: The vocabulary built from the summaries in training set.
            type (string): Indicates whether we are loading the training set or the
                           validation/testing set.
                           Can be one of the following: "train", "evaluation"
            TODO: FINISH COMMENT ARGS
        '''
        super(CustomDataset, self).__init__()

        self.source_code_texts = source_code_texts
        self.summary_texts = summary_texts

        assert len(source_code_texts) == len(summary_texts)
        self.dataset_size = len(source_code_texts)

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.type = type
        self.device = device

    def __len__(self):
        '''
        Returns the number of samples in our dataset.
        '''
        return self.dataset_size

    def __getitem__(self, index):
        '''
        __getitem__ runs on 1 example at a time. Here, we get an example at index 
        and return its numericalize source and target values using the 
        vocabulary objects we created in __init__

        Args:
            index (int): index of the example we want to numericalize

        Returns:
            If this represents the training set:
            - Two tensors with the source code and summary translated to numbers
            Otherwise (if this represents the validation or testing set):
            - The source code, summary and their translation to numbers
        '''
        source_text = self.source_code_texts[index]
        target_text = self.summary_texts[index]

        # numericalize texts ['<BOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        numericalized_source = [self.source_vocab.token_to_idx["<BOS>"]]
        numericalized_source += self.source_vocab.numericalize(source_text)
        numericalized_source.append(self.source_vocab.token_to_idx["<EOS>"])

        numericalized_target = [self.target_vocab.token_to_idx["<BOS>"]]
        numericalized_target += self.target_vocab.numericalize(target_text)
        numericalized_target.append(self.target_vocab.token_to_idx["<EOS>"])

        alignment = [self.source_vocab.token_to_idx["<BOS>"]]
        alignment += self.source_vocab.numericalize(target_text)
        alignment.append(self.source_vocab.token_to_idx["<EOS>"])

        if self.type == 'train':
            # convert the list to tensor and return
            return torch.tensor(numericalized_source, device=self.device), \
                   torch.tensor(numericalized_target, device=self.device), \
                   torch.tensor(alignment, device=self.device)
        elif self.type == 'evaluation':
            return source_text, target_text, \
                   torch.tensor(numericalized_source, device=self.device), \
                   torch.tensor(numericalized_target, device=self.device), \
                   torch.tensor(alignment, device=self.device)
        else:
            raise ValueError("Invalid type: " + self.type +
                             ". It can only be: train, evaluation.")
