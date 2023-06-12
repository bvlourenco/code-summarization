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

    @staticmethod
    def tokenizer(text):
        '''
        TODO: Improve the tokenizer

        a simple tokenizer to split on space and converts the sentence to list of words

        Args:
            text(string): The code snippet/summary to be tokenized
        
        Returns:
            A list of tokens of the code snippet/summary
        '''
        return [tok.lower().strip() for tok in text.split(' ')]

    def build_vocabulary(self, sentence_list):
        '''
        build the vocab: create a dictionary mapping of index to string (itos) and string to index (stoi)
        output ex. for stoi -> {'the':5, 'a':6, 'an':7}

        Args:
            sentence_list: List of code snippets/summaries
        '''
        # calculate the frequencies of each word first to remove the words with freq < freq_threshold
        frequencies = {}
        idx = 4  # index from which we want our dict to start. We already used 4 indexes for pad, start, end, unk

        # calculate freq of words
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
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
        for word in frequencies.keys():
            self.token_to_idx[word] = idx
            self.idx_to_token[idx] = word
            idx += 1
    
    def numericalize(self, text):
        '''
        convert the list of words to a list of corresponding indexes
        
        Args:
            text: The code snippet/summary text
        
        Returns:
            A list with numbers, where each number is the corresponding token index
        '''   
        #tokenize text
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.token_to_idx.keys():
                numericalized_text.append(self.token_to_idx[token])
            else: #out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.token_to_idx['<UNK>'])
                
        return numericalized_text