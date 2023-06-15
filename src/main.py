import torch
from args.parse_args import parse_arguments
from dataset.build_vocab import create_vocabulary
from dataset.dataloader import create_dataloaders
from dataset.load_dataset import load_dataset_file
from model.model import Model
from train.train import train_validate_model


def main():
    '''
    Performs the following actions:
    (1) Loads the dataset (code snippets and summaries)
    (2) Creates a vocabulary for the code snippets and another one for the summaries
    (3) Creates a dataloader for the training set and for the validation set
    (4) Creates an instance of the Transformer model
    (5) Trains and validates the model during some epochs (specified in the arguments)
        Validation is done at the end of each epoch

    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
    '''
    args = parse_arguments()

    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_code_texts, train_summary_texts = load_dataset_file(args.train_code_filename,
                                                              args.train_summary_filename,
                                                              'train',
                                                              args.debug_max_lines)

    source_vocab, target_vocab = create_vocabulary(train_code_texts,
                                                   train_summary_texts,
                                                   args.freq_threshold,
                                                   args.src_vocab_size,
                                                   args.tgt_vocab_size)

    train_dataloader, val_dataloader = create_dataloaders(train_code_texts,
                                                          train_summary_texts,
                                                          source_vocab,
                                                          target_vocab,
                                                          args.validation_code_filename,
                                                          args.validation_summary_filename,
                                                          args.batch_size,
                                                          args.num_workers,
                                                          device,
                                                          args.max_src_length,
                                                          args.max_tgt_length,
                                                          args.debug_max_lines)

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

    train_validate_model(model,
                         args.num_epochs,
                         train_dataloader,
                         val_dataloader,
                         args.tgt_vocab_size,
                         args.gradient_clipping,
                         args.mode,
                         target_vocab,
                         args.max_tgt_length,
                         args.checkpoint)


if __name__ == '__main__':
    main()
