import torch


class ValidationDataset:
    '''
    Custom validation dataset used to load the dataset through a dataloader.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''

    def __init__(self, val_code_texts, val_summary_texts, source_vocab, target_vocab):
        '''
        Args:
            val_code_texts: A list with size of validation set containing code snippets.
            val_summary_texts: A list with the summaries of the validation set.
            source_vocab: The vocabulary built from the code snippets in training set.
            target_vocab: The vocabulary built from the summaries in training set.
        '''
        self.source_code_texts = val_code_texts
        self.summary_texts = val_summary_texts

        assert len(val_code_texts) == len(val_summary_texts)
        self.dataset_size = len(val_code_texts)

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

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
        numericalized_source = [self.source_vocab.token_to_idx["<BOS>"]]
        numericalized_source += self.source_vocab.numericalize(source_text)
        numericalized_source.append(self.source_vocab.token_to_idx["<EOS>"])

        numericalized_target = [self.target_vocab.token_to_idx["<BOS>"]]
        numericalized_target += self.target_vocab.numericalize(target_text)
        numericalized_target.append(self.target_vocab.token_to_idx["<EOS>"])

        return source_text, target_text, torch.tensor(numericalized_source), torch.tensor(numericalized_target)
