import torch
from dataset.build_vocab import create_vocabulary
from dataset.load_dataset import load_dataset_file
from train_test.parallel import train_parallel
from train_test.train import create_train_model

def run_program(args, trial_number = None):
    '''
    Performs the following actions:
    (1) Loads the dataset (code snippets and summaries)
    (2) Creates a vocabulary for the code snippets and another one for the summaries
    (3) Creates a dataloader for the training set and for the validation set
    (4) Creates an instance of the Transformer model
    (5) Trains and validates the model during some epochs (specified in the arguments)
        Validation is done at the end of each epoch
    
    Args:
        args: The arguments given as input to the program.
        trial_number (int): If we are fine-tuning the parameters of the program, it is the
                            number of trial that are we are doing using optuna. 
                            Otherwise, it is None.

    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
    '''
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

    train_code_texts, train_summary_texts = load_dataset_file(args.train_filename,
                                                              'train',
                                                              args.debug_max_lines)

    val_code_texts, val_summary_texts = load_dataset_file(args.validation_filename,
                                                          'validation',
                                                          args.debug_max_lines)

    source_vocab, target_vocab = create_vocabulary(train_code_texts,
                                                   train_summary_texts,
                                                   args.freq_threshold,
                                                   args.src_vocab_size,
                                                   args.tgt_vocab_size)

    if device == torch.device('cuda'):
        train_parallel(world_size,
                       device,
                       args.src_vocab_size,
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
                       args.num_epochs,
                       args.gradient_clipping,
                       args.label_smoothing,
                       args.mode,
                       args.beam_size,
                       source_vocab,
                       target_vocab,
                       args.checkpoint,
                       train_code_texts,
                       train_summary_texts,
                       val_code_texts,
                       val_summary_texts,
                       args.batch_size,
                       args.num_workers,
                       trial_number)
    else:
        create_train_model(n_gpus,
                           world_size,
                           device,
                           args.src_vocab_size,
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
                           args.num_epochs,
                           args.gradient_clipping,
                           args.label_smoothing,
                           args.mode,
                           args.beam_size,
                           source_vocab,
                           target_vocab,
                           args.checkpoint,
                           train_code_texts,
                           train_summary_texts,
                           val_code_texts,
                           val_summary_texts,
                           args.batch_size,
                           args.num_workers,
                           trial_number)