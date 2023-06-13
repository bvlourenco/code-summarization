from datetime import datetime
import json
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset.custom_collate_fn import MyCollate
from dataset.vocabulary import Vocabulary
from evaluation import create_loss_plot
from dataset.train_dataset import TrainDataset
from transformer.transformer_model import Transformer
from timeit import default_timer as timer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset.validation_dataset import ValidationDataset

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 150
dropout = 0.1
learning_rate = 0.0001
batch_size = 32
num_epochs = 30
gradient_clipping = 1
train_code_filename = '../data/train_code.txt'
train_summary_filename = '../data/train_summary.txt'
validation_code_filename = '../data/validation_code.txt'
validation_summary_filename = '../data/validation_summary.txt'
freq_threshold = 0
num_workers = 0
mode = 'translation'
debug_max_lines = -1


def evaluate(model, criterion, device, val_dataloader, mode, source_vocab, target_vocab):
    '''
    Validates the model to see how well he generalizes. 
    Does not update the weights of the model.

    Args:
        model: The model (an instance of Transformer).
        criterion: The function used to compute the loss (an instance of nn.CrossEntropyLoss).
        device: The device where the model and tensors are inserted (GPU or CPU).
        val_dataloader: A Dataloader object that contains the validation set.
        mode (string): Indicates whether we only want to compute validation loss or if we also
                       want to translate the source sentences in the validation set.
                       Can be one of the following: "loss", "translation"
        source_vocab: The vocabulary built from the code snippets in training set.
        target_vocab: The vocabulary built from the summaries in training set.

    Returns:
        The average loss across all examples during one epoch.

    Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
    '''
    # Set the model to evaluation mode
    model.eval()
    losses = 0

    # evaluate without updating gradients
    # Tells pytorch to not calculate the gradients
    with torch.no_grad():
        with open('../results/validation_' + str(datetime.now()) + '.json', 'w') as log:
            for src, tgt in tqdm(val_dataloader, desc="Validating"):
                src = src.to(device)
                tgt = tgt.to(device)

                if mode == 'translation':
                    start_symbol_idx = target_vocab.token_to_idx['<BOS>']
                    end_symbol_idx = target_vocab.token_to_idx['<EOS>']

                    # Translating a batch of source sentences
                    for i in range(src.shape[0]):
                        # src[i] has shape (max_src_length, )
                        # Performing an unsqueeze on src[i] will make its shape (1, max_src_length)
                        # which is the correct shape since batch_size = 1 in this case
                        tgt_preds_idx = greedy_decode(model, src[i].unsqueeze(0), device, start_symbol_idx, end_symbol_idx)

                        # Passing the tensor to 1 dimension
                        tgt_preds_idx.flatten()
                        
                        # Translating the indexes of tokens to the textual representation of tokens
                        # Replacing <BOS> and <EOS> with empty string
                        tgt_pred_tokens = translate_tokens(tgt_preds_idx, target_vocab)
                        
                        example = {}
                        example["prediction"] = tgt_pred_tokens
                        example["reference"] = translate_tokens(tgt[i], target_vocab)
                        example["code"] = translate_tokens(src[i], source_vocab)

                        log.write(json.dumps(example, indent=4) + ',\n')

                # Slicing the last element of tgt_data because it is used to compute the
                # loss.
                output = model(src, tgt[:, :-1])

                # - output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
                # - tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
                # - criterion(prediction_labels, target_labels)
                # - tgt[:, 1:] removes the first token from the target sequence since we are training the
                # model to predict the next word based on previous words.
                loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                                tgt[:, 1:].contiguous().view(-1))

                losses += loss.item()

    return losses / len(list(val_dataloader))

def translate_tokens(tokens_idx, vocabulary):
    '''
    Args:
        tokens_idx: A list of numbers where each number is the index of a token
        vocabulary: The vocabulary from where the tokens belong (code vocabulary
                    or summaries vocabulary).
    
    Returns:
        A string with the textual representation of the tokens. Tokens are 
        separated with whitespaces.
    '''
    translation = ""
    for token_idx in tokens_idx[1:]:
        token = vocabulary.idx_to_token[token_idx.item()]
        if token in ["<EOS>", "<PAD>"]:
            break 
        elif token == "<BOS>":
            raise ValueError("Unexpected <BOS>")
        elif token == "<UNK>":
            token = "?"
        translation += token + " "
    return translation.strip()

def train_epoch(model, optimizer, criterion, device, train_dataloader):
    '''
    Trains the model during one epoch, using the examples of the dataset for training.
    Computes the loss during this epoch.

    Args:
        model: The model (an instance of Transformer).
        optimizer: The optimizer used in training (an instance of Adam optimizer).
        criterion: The function used to compute the loss (an instance of nn.CrossEntropyLoss).
        device: The device where the model and tensors are inserted (GPU or CPU). 
        train_dataloader: A Dataloader object that contains the training set.

    Returns:
        The average loss across all examples during one epoch.

    Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
            https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe
    '''
    # Set the model to training mode
    model.train()
    losses = 0

    for src, tgt in tqdm(train_dataloader, desc="Training"):
        # Passing vectors to GPU if it's available
        src = src.to(device)
        tgt = tgt.to(device)

        # Zeroing the gradients of transformer parameters
        optimizer.zero_grad()

        # Slicing the last element of tgt_data because it is used to compute the
        # loss.
        output = model(src, tgt[:, :-1])

        # - output is reshaped using view() method in shape (batch_size * tgt_len, tgt_vocab_size)
        # - tgt_data is reshaped using view() method in shape (batch_size * tgt_len)
        # - criterion(prediction_labels, target_labels)
        # - tgt[:, 1:] removes the first token from the target sequence since we are training the
        # model to predict the next word based on previous words.
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                         tgt[:, 1:].contiguous().view(-1))

        # Computing gradients of loss through backpropagation
        loss.backward()

        # clip the weights
        clip_grad_norm_(model.parameters(), gradient_clipping)

        # Update the model's parameters based on the computed gradients
        optimizer.step()

        losses += loss.item()

    return losses / len(list(train_dataloader)), train_dataloader


def get_dataloader(dataset, source_vocab):
    '''
    Get a dataloader given a dataset.

    Args:
        dataset: The dataset from which we will build the dataloader.
                 Instance of TrainDataset or ValidationDataset.

    Returns:
        A new dataloader.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''
    # pad_idx, bos_idx and eos_idx are the same between code snippet and summary
    pad_idx = source_vocab.token_to_idx['<PAD>']
    bos_idx = source_vocab.token_to_idx['<BOS>']
    eos_idx = source_vocab.token_to_idx['<EOS>']
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=True,
                      drop_last=True,
                      pin_memory=False,
                      collate_fn=MyCollate(pad_idx, bos_idx, eos_idx, device, max_seq_length))


def create_dataloaders(source_code_texts, summary_texts, source_vocab, target_vocab):
    '''
    Creates a training and validation dataset and dataloaders.

    Args:
        source_code_texts: A list with size of training set containing code snippets.
        summary_texts: A list with the summaries of the training set.
        source_vocab: The vocabulary built from the code snippets in training set.
        target_vocab: The vocabulary built from the summaries in training set.

    Returns:
        The training and validation dataloaders (instances of Dataloader).

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''
    train_dataset = TrainDataset(
        source_code_texts, summary_texts, source_vocab, target_vocab)
    train_dataloader = get_dataloader(train_dataset, source_vocab)

    # Loading the validation set
    val_code_texts, val_summary_texts = load_dataset_file(validation_code_filename,
                                                          validation_summary_filename,
                                                          'validation')

    validation_dataset = ValidationDataset(
        val_code_texts, val_summary_texts, source_vocab, target_vocab)
    val_dataloader = get_dataloader(validation_dataset, source_vocab)
    return train_dataloader, val_dataloader


def load_dataset_file(code_filename, summary_filename, type):
    '''
    Given a file with code snippets and a file with summaries, it loads all 
    examples of them from the respective files.

    Args:
        code_filename (string): The filename of the file with code snippets.
        summary_filename (string): The filename of the file with summaries.
        type (string): Indicates whether we are loading the training set or the
                       validation set from the files.
                       Can be one of the following: "train", "validation"
    
    Returns:
        A list of code snippets examples and a list of summaries examples.
    '''
    if (not os.path.exists(code_filename)):
        raise ValueError("code snippet filename does not exist")

    if (not os.path.exists(summary_filename)):
        raise ValueError("summary filename does not exist")

    with open(code_filename) as code_file:
        num_code_lines = sum(1 for _ in open(code_filename, 'r'))
        if debug_max_lines > 0:
            source_code_texts = [next(code_file) for _ in tqdm(range(debug_max_lines), 
                                                               total=debug_max_lines, 
                                                               desc="Reading " + str(debug_max_lines) + " " + type + " code snippets")]
        else:
            source_code_texts = [line.strip() for line in tqdm(
                code_file, total=num_code_lines, desc="Reading " + type + " code snippets")]

    with open(summary_filename) as summary_file:
        num_summary_lines = sum(1 for _ in open(summary_filename, 'r'))
        if debug_max_lines > 0:
            summary_texts = [next(summary_file) for _ in tqdm(range(debug_max_lines), 
                                                         total=debug_max_lines, 
                                                         desc="Reading " + str(debug_max_lines) + " " + type + " summaries")]
        else:
            summary_texts = [line.strip() for line in tqdm(
                summary_file, total=num_summary_lines, desc="Reading " + type + " summaries")]

    return source_code_texts, summary_texts


def create_vocabulary(source_code_texts, summary_texts):
    '''
    Creates the vocabulary for the code snippets and for the summaries.

    Args:
        source_code_texts: List with the code snippets from the training set.
        summary_texts: List with the summaries from the training set.
    
    Returns:
        The vocabulary of the code snippets and the vocabulary of the summaries.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''
    # Initialize source vocab object and build vocabulary
    source_vocab = Vocabulary(freq_threshold, src_vocab_size)
    source_vocab.build_vocabulary(source_code_texts, "code")
    # Initialize target vocab object and build vocabulary
    target_vocab = Vocabulary(freq_threshold, tgt_vocab_size)
    target_vocab.build_vocabulary(summary_texts, "summary")

    return source_vocab, target_vocab


def greedy_decode(model, src, device, start_symbol_idx, end_symbol_idx):
    '''
    Creates a code comment given a code snippet using the greedy decoding
    strategy (where we generate one token at a time by greedily selecting the
    token with the highest probability at each step)

    Args:
        model: The model (an instance of Transformer). 
        src: The input (a code snippet numericalized). Shape: `(batch_size, src_len)` 
        device: The device where the model and tensors are inserted (GPU or CPU). 
        start_symbol_idx: The vocabulary start symbol index (<BOS> index)
        end_symbol_idx: The vocabulary end symbol index (<EOS> index)

    Returns:
        tgt: The predicted code comment token indexes. Shape: `(1, tgt_length)`

    Source: https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
    '''
    src_mask = model.generate_src_mask(src)
    enc_output = model.encode(src, src_mask)

    # Initializes the target sequence with <BOS> symbol
    # Will be further expland to include the predicted tokens
    tgt = torch.ones(1, 1).fill_(start_symbol_idx).type(torch.long).to(device)

    for _ in range(max_seq_length - 1):
        enc_output = enc_output.to(device)

        # Generate probabilities for the next token
        # Essentially, we're running the decoder phasing of the model
        # to generate the next token
        tgt_mask = model.generate_tgt_mask(tgt)
        dec_output = model.decode(src, src_mask, tgt, tgt_mask, enc_output)
        # prob has shape: `(batch_size, tgt_vocab_size)`
        prob = model.fc(dec_output[-1, :, :])

        # Get the index of of the token with the
        # highest probability as the predicted next word.
        _, next_word = torch.max(prob, dim=1)

        # Get the integer value present in the tensor. It represents
        # the index of the predicted token.
        next_word = next_word.item()

        # Adding the new predicted token to tgt
        tgt = torch.cat([tgt,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

        # If the token predicted is <EOS>, then stop generating new tokens.
        if next_word == end_symbol_idx:
            break
    
    # Adding <EOS> token
    tgt = torch.cat([tgt,
                     torch.ones(1, 1).type_as(src.data).fill_(end_symbol_idx)], dim=0)
    
    # Return the indexes of the predicted tokens
    return tgt


if __name__ == '__main__':
    '''
    Source: https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
            https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer
    '''
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transformer = Transformer(src_vocab_size,
                              tgt_vocab_size,
                              d_model,
                              num_heads,
                              num_layers,
                              d_ff,
                              max_seq_length,
                              dropout,
                              device)

    # Passing the model and all its layers to GPU if available
    transformer = transformer.to(device)

    train_code_texts, train_summary_texts = load_dataset_file(
        train_code_filename, train_summary_filename, 'train')
    source_vocab, target_vocab = create_vocabulary(
        train_code_texts, train_summary_texts)
    train_dataloader, val_dataloader = create_dataloaders(
        train_code_texts, train_summary_texts, source_vocab, target_vocab)

    # ignore_index is the padding token index
    criterion = nn.CrossEntropyLoss(ignore_index=source_vocab.token_to_idx['<PAD>'])
    optimizer = optim.Adam(transformer.parameters(),
                           lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    train_epoch_loss, val_epoch_loss = [], []
    for epoch in range(1, num_epochs + 1):
        start_time = timer()
        train_loss, train_dataset = train_epoch(
            transformer, optimizer, criterion, device, train_dataloader)
        val_loss = evaluate(transformer, criterion, device, val_dataloader, mode, source_vocab, target_vocab)
        end_time = timer()

        print(f"Epoch: {epoch} | Time = {(end_time - start_time):.3f}s")
        print(
            f'Train Loss: {train_loss:.3f} | Train Perplexity: {math.exp(train_loss):7.3f}')
        print(
            f'Validation Loss: {val_loss:.3f} | Validation Perplexity: {math.exp(val_loss):7.3f}')

        train_epoch_loss.append(train_loss)
        val_epoch_loss.append(val_loss)

    create_loss_plot(train_epoch_loss, val_epoch_loss)

    '''
    TODO: For model training, check:
    https://nlp.seas.harvard.edu/annotated-transformer/
    https://medium.com/@hunter-j-phillips/putting-it-all-together-the-implemented-transformer-bfb11ac1ddfe

    TODO: For evaluation, check:
    https://github.com/DeepSoftwareAnalytics/CodeSumEvaluation
    https://arxiv.org/pdf/2107.07112.pdf

    TODO: In inference time, use beam search instead of greedy decoding. Explanation:
    https://datascience.stackexchange.com/a/93146

    TODO: Check tokenization techniques 
    https://towardsdatascience.com/word-subword-and-character-based-tokenization-know-the-difference-ea0976b64e17
    Good paper: https://openreview.net/pdf?id=htL4UZ344nF
                https://openreview.net/forum?id=htL4UZ344nF
    
    TODO: Check Kaiming initialization
    https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

    TODO: Check learning rate scheduler and the following points:
    - Experimentar técnicas diferentes de tokenization (Word, Subword, and Character-Based Tokenization)
    - Aplicar label smoothing
    - Aplicar beam search em vez de greedy decoding no processo de inference
    - Aplicar copy attention e relative positional encoding
    - Aplicar as alterações que proponho no paper
    - Save and load model: https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
    
    General tutorial with Transformer model, training, evaluation and inference:
    https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=transformer

    Basics tutorials: https://pytorch.org/tutorials/beginner/basics/intro.html
    Training Transformer models in multiple GPUs: https://pytorch.org/tutorials/advanced/ddp_pipeline.html?highlight=transformer
    Datasets and Dataloaders: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=transformer
    Deploying Pytorch in python via a REST API with Flask: https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html?highlight=transformer
    Training with Pytorch: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html?highlight=transformer
    '''
