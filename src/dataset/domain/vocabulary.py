from tqdm import tqdm


class Vocabulary:
    '''
    Creates the word to index and index to token mapping. Numericalize the 
    code snippets/summaries.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''

    def __init__(self, freq_threshold, max_size):
        '''
        Args:
            freq_threshold (int): the minimum times a word must occur in corpus to be treated in vocab
            max_size (int): max source vocab size. Only considers the top `max_size` most frequent words and discard others
        '''
        # initiate the index to token dict
        # <PAD> -> padding, used for padding the shorter sentences in a batch to match the length of longest sentence in the batch
        # <BOS> -> begin token, added in front of each sentence to signify the begin of sentence
        # <EOS> -> End of sentence token, added to the end of each sentence to signify the end of sentence
        # <UNK> -> words which are not found in the vocab are replace by this token
        self.idx_to_token = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
        # initiate the token to index dict
        self.token_to_idx = {k: j for j, k in self.idx_to_token.items()}

        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def __len__(self):
        '''
        __len__ is used by dataloader later to create batches
        '''
        return len(self.idx_to_token)

    def build_vocabulary(self, tokens_list, type):
        '''
        build the vocab: create a dictionary mapping of index to string (idx_to_token) 
        and string to index (token_to_idx)
        output ex. for token_to_idx -> {'the':5, 'a':6, 'an':7}

        Args:
            tokens_list: List of code snippets tokens/summaries tokens
            type (string): Indicates whether we are building the vocabulary
                           for code snippets or for summaries.
                           Can be one of the following: "code", "summary"
        '''
        # calculate the frequencies of each word first to remove the words with freq < freq_threshold
        frequencies = {}
        idx = 4  # index from which we want our dict to start. We already used 4 indexes for pad, start, end, unk

        # calculate freq of words
        for sentence_tokens in tqdm(tokens_list, total=len(tokens_list),
                                    desc="Building " + type + " word frequency"):
            for word in sentence_tokens:
                if word not in frequencies.keys():
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

        # limit vocab by removing low freq words
        frequencies = {k: v for k, v in frequencies.items() if v >
                       self.freq_threshold}

        # limit vocab to the max_size specified
        # idx = 4 for pad, start, end and unk special tokens
        frequencies = dict(
            sorted(frequencies.items(), key=lambda x: -x[1])[:self.max_size-idx])

        # create vocab
        for word in tqdm(frequencies.keys(), total=len(frequencies),
                         desc="Building " + type + " vocabulary"):
            self.token_to_idx[word] = idx
            self.idx_to_token[idx] = word
            idx += 1

    def numericalize(self, text_tokens):
        '''
        Convert the list of words to a list of corresponding indexes.

        Args:
            text_tokens: The code snippet/summary tokens

        Returns:
            A list with numbers, where each number is the corresponding token index
        '''
        numericalized_text = []
        for token in text_tokens:
            if token in self.token_to_idx.keys():
                numericalized_text.append(self.token_to_idx[token])
            else:  # out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.token_to_idx['<UNK>'])

        return numericalized_text
