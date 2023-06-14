from dataset.domain.vocabulary import Vocabulary


def create_vocabulary(source_code_texts, summary_texts, freq_threshold, src_vocab_size, tgt_vocab_size):
    '''
    Creates the vocabulary for the code snippets and for the summaries.

    Args:
        source_code_texts: List with the code snippets from the training set.
        summary_texts: List with the summaries from the training set.
        freq_threshold (int): the minimum times a word must occur in corpus to be treated in vocab
        src_vocab_size (int): size of the source vocabulary.
        tgt_vocab_size (int): size of the target vocabulary.

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