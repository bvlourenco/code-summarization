import torch
from torch.nn.functional import pad


class CustomCollate:
    '''
    Class to add padding to the batches.
    It also adds <BOS> to the beggining and <EOS> to the end of each example.

    It's a class to allow us receiving the indexes of <BOS>, <EOS> and <PAD> tokens.

    collate_fn in dataloader is used for post processing on a single batch. 
    Like __getitem__ in dataset class, it is used on a single example.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''

    def __init__(self, 
                 pad_idx, 
                 bos_idx, 
                 eos_idx, 
                 device, 
                 max_src_length, 
                 max_tgt_length,
                 src_vocab_size, 
                 type):
        '''
        Args:
            pad_idx (int): index of the <PAD> token
            bos_idx (int): index of the <BOS> token
            eos_idx (int): index of the <EOS> token
            device: The device where the model and tensors are inserted (GPU or CPU).
            max_src_length (int): Maximum length of the source code.
            max_tgt_length (int): Maximum length of the summaries.
            src_vocab_size (int): size of the source vocabulary.
            type (string): Indicates whether we are working with the training 
                           set or the validation/testing set.
                           Can be one of the following: "train", "evaluation"
        '''
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.device = device
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.src_vocab_size = src_vocab_size
        self.type = type

    def call_evaluation(self, batch):
        '''
        Pads and adds <BOS> and <EOS> tokens to the validation/testing set batch

        Args:
            batch: Contains source code, summaries, source code numericalized 
                       and summaries numericalized.

        Returns:
            A Source Code Batch, a Summary Batch, a source code batch 
            numericalized and a summary batch numericalized. 
            Shapes: `(batch_size, max_src_len)`
                    `(batch_size, max_tgt_len)`
                    `(batch_size, max_src_len)`
                    `(batch_size, max_tgt_len)`

        Source: https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
        '''
        code_batch, summary_batch, code_idxs_batch, summary_idxs_batch = [], [], [], []

        batch_size = len(batch)

        # Maps a source code word to its respective index in the extended vocabulary
        # (because of the copy mechanism)
        src_map_batch = torch.zeros(batch_size, self.max_src_length, self.src_vocab_size, 
                                    device=self.device)

        # Alignment refers to the mapping between the target tokens and 
        # their corresponding positions in the source sequence
        alignment_batch = torch.zeros(batch_size, self.max_tgt_length, device=self.device).long()

        # for each code snippet
        for i, (source_code, summary, code_idxs, summary_idxs, alignment) in enumerate(batch):
            alignment = alignment[:self.max_tgt_length]     

            alignment_batch[i, : len(alignment)] = alignment.clone().detach()

            for j, t in enumerate(code_idxs[:self.max_src_length]):
                src_map_batch[i, j, t] = 1

            code_batch.append(source_code)
            summary_batch.append(summary)

            # add padding
            code_idxs_batch.append(
                pad(code_idxs, (0, self.max_src_length - len(code_idxs)), 
                    value=self.pad_idx))

            # add padding
            summary_idxs_batch.append(
                pad(summary_idxs, (0, self.max_tgt_length - len(summary_idxs)), 
                    value=self.pad_idx))

        return code_batch, summary_batch, \
               torch.stack(code_idxs_batch).to(self.device), \
               torch.stack(summary_idxs_batch).to(self.device), \
               src_map_batch, alignment_batch

    def call_train(self, batch):
        '''
        Pads and adds <BOS> and <EOS> tokens to the training set batch

        Args:
            batch: Source Code - Summary pairs

        Returns:
            A Source Code Batch and a Summary Batch. 
            src_map: TODO
            Shapes: `(batch_size, max_src_len)`
                    `(batch_size, max_tgt_len)`

        Source: https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
        '''
        code_batch, summary_batch = [], []

        batch_size = len(batch)

        # Maps a source code word to its respective index in the extended vocabulary
        # (because of the copy mechanism)
        src_map_batch = torch.zeros(batch_size, self.max_src_length, self.src_vocab_size, 
                                    device=self.device)

        # Alignment refers to the mapping between the target tokens and 
        # their corresponding positions in the source sequence
        alignment_batch = torch.zeros(batch_size, self.max_tgt_length, device=self.device).long()

        # for each code snippet
        for i, (source_code, summary, alignment) in enumerate(batch):
            alignment = alignment[:self.max_tgt_length]     

            alignment_batch[i, : len(alignment)] = alignment.clone().detach()

            for j, t in enumerate(source_code[:self.max_src_length]):
                src_map_batch[i, j, t] = 1

            # add padding
            code_batch.append(
                pad(source_code, (0, self.max_src_length - len(source_code)), 
                    value=self.pad_idx))

            # add padding
            summary_batch.append(
                pad(summary, (0, self.max_tgt_length - len(summary)), 
                    value=self.pad_idx))

        return torch.stack(code_batch).to(self.device), \
               torch.stack(summary_batch).to(self.device), \
               src_map_batch, alignment_batch

    def __call__(self, batch):
        '''
        Default method that pads and adds <BOS> and <EOS> tokens to the example

        First the obj is created using the constructor

        Then if obj(batch) is called -> __call__ runs by default

        Args:
            If this represents the training set:
                batch: Source Code - Summary pairs
            Otherwise (if this represents the validation or testing set):
                batch: Contains source code, summaries, source code numericalized 
                       and summaries numericalized.
            TODO: Refactor comment

        Returns:
            If this represents the training set:
                A Source Code Batch and a Summary Batch. 
                Shapes: `(batch_size, max_src_len)`
                        `(batch_size, max_tgt_len)`
            Otherwise (if this represents the validation or testing set):
                A Source Code Batch, a Summary Batch, a source code batch numericalized
                and a summary batch numericalized. 
                Shapes: `(batch_size, max_src_len)`
                        `(batch_size, max_tgt_len)`
                        `(batch_size, max_src_len)`
                        `(batch_size, max_tgt_len)`
        '''
        if self.type == 'train':
            return self.call_train(batch)
        elif self.type == 'evaluation':
            return self.call_evaluation(batch)
        else:
            raise ValueError("Invalid type: " + self.type + ". It can only be: train, evaluation.")