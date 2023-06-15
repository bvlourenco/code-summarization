import os
import torch
from args.parse_args import parse_arguments
from dataset.build_vocab import create_vocabulary
from dataset.dataloader import create_dataloaders
from dataset.load_dataset import load_dataset_file
from model.model import Model
from train.train import train_validate_model
import torch.multiprocessing as mp
import torch.distributed as dist


def demo_model_parallel(gpu_rank,
                        world_size,
                        src_vocab_size,
                        tgt_vocab_size,
                        d_model,
                        num_heads,
                        num_layers,
                        d_ff,
                        max_src_length,
                        max_tgt_length,
                        dropout,
                        learning_rate,
                        pad_idx,
                        num_epochs,
                        gradient_clipping,
                        mode,
                        source_vocab,
                        target_vocab,
                        checkpoint,
                        train_code_texts,
                        train_summary_texts,
                        val_code_texts,
                        val_summary_texts,
                        batch_size,
                        num_workers):
    '''
    TODO

    Source: https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
            https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#id1
            https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html
    '''

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=gpu_rank
    )
    torch.cuda.set_device(gpu_rank)

    train_dataloader, val_dataloader = create_dataloaders(train_code_texts,
                                                          train_summary_texts,
                                                          val_code_texts,
                                                          val_summary_texts,
                                                          source_vocab,
                                                          target_vocab,
                                                          batch_size,
                                                          num_workers,
                                                          gpu_rank,
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
                  pad_idx,
                  gpu_rank,
                  gpu_rank)

    train_validate_model(model,
                         num_epochs,
                         train_dataloader,
                         val_dataloader,
                         tgt_vocab_size,
                         gradient_clipping,
                         mode,
                         target_vocab,
                         max_tgt_length,
                         checkpoint)

    dist.destroy_process_group()


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
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus

    train_code_texts, train_summary_texts = load_dataset_file(args.train_code_filename,
                                                              args.train_summary_filename,
                                                              'train',
                                                              args.debug_max_lines)

    val_code_texts, val_summary_texts = load_dataset_file(args.validation_code_filename,
                                                          args.validation_summary_filename,
                                                          'validation',
                                                          args.debug_max_lines)

    source_vocab, target_vocab = create_vocabulary(train_code_texts,
                                                   train_summary_texts,
                                                   args.freq_threshold,
                                                   args.src_vocab_size,
                                                   args.tgt_vocab_size)

    mp.spawn(demo_model_parallel,
             args=(world_size,
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
                   args.mode,
                   source_vocab,
                   target_vocab,
                   args.checkpoint,
                   train_code_texts,
                   train_summary_texts,
                   val_code_texts,
                   val_summary_texts,
                   args.batch_size,
                   args.num_workers,
                   ),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main()
