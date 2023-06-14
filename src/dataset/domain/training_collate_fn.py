import torch
from torch.nn.functional import pad


class TrainCollate:
    '''
    Class to add padding to the batches.
    It also adds <BOS> to the beggining and <EOS> to the end of each example.

    It's a class to allow us receiving the indexes of <BOS>, <EOS> and <PAD> tokens.

    collate_fn in dataloader is used for post processing on a single batch. 
    Like __getitem__ in dataset class, it is used on a single example.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''

    def __init__(self, pad_idx, bos_idx, eos_idx, device, max_seq_length):
        '''
        Args:
            pad_idx (int): index of the <PAD> token
            bos_idx (int): index of the <BOS> token
            eos_idx (int): index of the <EOS> token
            device: The device where the model and tensors are inserted (GPU or CPU).
            max_seq_length (int): Maximum length of the source code and summary.
        '''
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.device = device
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        '''
        Default method that pads and adds <BOS> and <EOS> tokens to the example

        First the obj is created using the constructor

        Then if obj(batch) is called -> __call__ runs by default

        Args:
            batch: Source Code - Summary pairs

        Returns:
            A Source Code Batch and a Summary Batch. 
            Shapes: `(batch_size, max_src_len)`
                    `(batch_size, max_tgt_len)`

        Source: https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
        '''
        code_batch, summary_batch = [], []

        # for each code snippet
        for (source_code, summary) in batch:
            # add padding
            code_batch.append(
                pad(source_code, (0, self.max_seq_length - len(source_code)), value=self.pad_idx))

            # add padding
            summary_batch.append(
                pad(summary, (0, self.max_seq_length - len(summary)), value=self.pad_idx))

        return torch.stack(code_batch), torch.stack(summary_batch)
