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
               learning_rate):
    '''
    TODO
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
                  source_vocab.token_to_idx['<PAD>'],
                  device,
                  gpu_rank)

    model.load()
    model.test(test_dataloader,
               target_vocab,
               max_tgt_length)
