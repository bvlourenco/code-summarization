import torch
from args.parse_args import parse_test_args
from dataset.build_vocab import load_vocab
from dataset.dataloader import load_evaluation_dataloader
from dataset.load_dataset import load_dataset_file
from model.model import Model


def main():
    args = parse_test_args()

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_code_texts, test_summary_texts = load_dataset_file(args.test_code_filename,
                                                            args.test_summary_filename,
                                                            'test',
                                                            args.debug_max_lines)

    source_vocab, target_vocab = load_vocab('../results/src_vocab.pkl', 
                                            '../results/tgt_vocab.pkl')

    test_dataloader = load_evaluation_dataloader(test_code_texts,
                                                 test_summary_texts,
                                                 source_vocab,
                                                 target_vocab,
                                                 args.batch_size,
                                                 args.num_workers,
                                                 device,
                                                 args.max_src_length,
                                                 args.max_tgt_length)
    
    model = Model(args.src_vocab_size,
                  args.tgt_vocab_size,
                  args.d_model,
                  args.num_heads,
                  args.num_layers,
                  args.d_ff,
                  args.max_src_length,
                  args.max_tgt_length,
                  args.dropout,
                  args.learning_rate,
                  source_vocab.token_to_idx['<PAD>'],
                  device)
    
    model.load()
    model.test(test_dataloader,
               target_vocab,
               args.max_tgt_length)

if __name__ == '__main__':
    main()