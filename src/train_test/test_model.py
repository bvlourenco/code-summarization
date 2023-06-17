from dataset.dataloader import load_evaluation_dataloader
from model.model import Model


def test_model(test_code_texts,
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
               label_smoothing):
    '''
    Creates the testing dataloader, the model and tests the model.

    Args:
        test_code_texts: A list of code snippets examples belonging to the testing set.
        test_summary_texts: A list of summaries examples belonging to the testing set.
        source_vocab: The vocabulary built from the code snippets in training set.
        target_vocab: The vocabulary built from the summaries in training set.
        batch_size (int): how many samples per batch to load
        num_workers (int): how many subprocesses to use for data loading.
        device: The device where the model and tensors are inserted (GPU or CPU). 
        max_src_length (int): maximum length of the source code.
        max_tgt_length (int): maximum length of the summaries. 
        world_size (int): The number of GPUs available in the machine.
                          It has the value of -1 if no GPUs are avaiable.
        gpu_rank (int): The rank of the GPU.
                        It has the value of -1 if no GPUs are avaiable.
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
    test_dataloader = load_evaluation_dataloader(test_code_texts,
                                                 test_summary_texts,
                                                 source_vocab,
                                                 target_vocab,
                                                 batch_size,
                                                 num_workers,
                                                 device,
                                                 max_src_length,
                                                 max_tgt_length,
                                                 world_size,
                                                 gpu_rank)

    model = Model(src_vocab_size,
                  tgt_vocab_size,
                  d_model,
                  num_heads,
                  num_layers,
                  d_ff,
                  max_src_length,
                  max_tgt_length,
                  dropout,
                  learning_rate,
                  label_smoothing,
                  source_vocab.token_to_idx['<PAD>'],
                  device,
                  gpu_rank)

    model.load(gpu_rank)
    model.test(test_dataloader,
               target_vocab,
               max_tgt_length)
