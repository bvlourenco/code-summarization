import torch
from args.parse_args import parse_test_args
from dataset.build_vocab import load_vocab
from dataset.load_dataset import load_dataset_file
from train_test.test_model import test_model
from train_test.parallel import test_parallel


def main():
    args = parse_test_args()

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device == torch.device('cuda'):
        # If we're running in GPU, we'll have 1 process per GPU
        # world_size is the number of processes to be launched
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
    else:
        # If we're running in CPU, we will only have 1 main process
        n_gpus, world_size = -1, -1

    test_code_texts, test_summary_texts = load_dataset_file(args.test_filename,
                                                            'test',
                                                            args.debug_max_lines)

    source_vocab, target_vocab = load_vocab('../results/src_vocab.pkl',
                                            '../results/tgt_vocab.pkl')

    if device == torch.device('cuda'):
        test_parallel(world_size,
                      test_code_texts,
                      test_summary_texts,
                      source_vocab,
                      target_vocab,
                      args.batch_size,
                      args.num_workers,
                      device,
                      args.max_src_length,
                      args.max_tgt_length,
                      args.src_vocab_size,
                      args.tgt_vocab_size,
                      args.d_model,
                      args.num_heads,
                      args.num_layers,
                      args.d_ff,
                      args.dropout,
                      args.learning_rate)
    else:
        test_model(test_code_texts,
                   test_summary_texts,
                   source_vocab,
                   target_vocab,
                   args.batch_size,
                   args.num_workers,
                   device,
                   args.max_src_length,
                   args.max_tgt_length,
                   world_size,
                   n_gpus,
                   args.src_vocab_size,
                   args.tgt_vocab_size,
                   args.d_model,
                   args.num_heads,
                   args.num_layers,
                   args.d_ff,
                   args.dropout,
                   args.learning_rate)


if __name__ == '__main__':
    main()
