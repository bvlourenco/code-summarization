import os
import pickle
from dataset.domain.vocabulary import Vocabulary


def create_vocabulary(source_code_tokens,
                      summary_tokens, 
                      freq_threshold, 
                      src_vocab_size, 
                      tgt_vocab_size):
    '''
    Creates the vocabulary for the code snippets and for the summaries.

    Args:
        source_code_tokens: List with the code snippet tokens from the training set.
        summary_tokens: List with the summary tokens from the training set.
        freq_threshold (int): the minimum times a word must occur in corpus to be treated in vocab
        src_vocab_size (int): size of the source vocabulary.
        tgt_vocab_size (int): size of the target vocabulary.

    Returns:
        The vocabulary of the code snippets and the vocabulary of the summaries.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''
    # Initialize source vocab object and build vocabulary
    source_vocab = Vocabulary(freq_threshold, src_vocab_size)
    source_vocab.build_vocabulary(source_code_tokens, "code")
    # Initialize target vocab object and build vocabulary
    target_vocab = Vocabulary(freq_threshold, tgt_vocab_size)
    target_vocab.build_vocabulary(summary_tokens, "summary")

    pickle.dump(source_vocab, open('../results/src_vocab.pkl', 'wb'))
    pickle.dump(target_vocab, open('../results/tgt_vocab.pkl', 'wb'))

    return source_vocab, target_vocab


def load_vocab(src_filename, tgt_filename):
    '''
    Loads the source and target vocabulary from the given filename

    Args:
        src_filename: The name of file with the source vocabulary
        tgt_filename: The name of file with the target vocabulary
    '''
    if os.path.isfile(src_filename) and os.path.isfile(tgt_filename):
        return pickle.load(open('../results/src_vocab.pkl', 'rb')), \
            pickle.load(open('../results/tgt_vocab.pkl', 'rb'))
    else:
        raise ValueError("Either " + src_filename + " or " +
                         tgt_filename + " are not files")
