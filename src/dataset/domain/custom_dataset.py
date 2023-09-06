import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from dataset.domain.sparse_matrix_to_tensor import build_tensor_from_sparse_matrix


class CustomDataset(Dataset):
    '''
    Custom dataset used to load the dataset through a dataloader.

    Source: https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e
    '''

    def __init__(self,
                 source_code_texts,
                 source_code_tokens,
                 summary_texts,
                 summary_tokens,
                 max_src_length,
                 token_matrices,
                 statement_matrices,
                 data_flow_matrices,
                 control_flow_matrices,
                 source_vocab,
                 target_vocab,
                 type):
        '''
        Args:
            source_code_texts: A list containing code snippets.
            source_code_tokens: A list containing the tokens of the code snippets.
            summary_texts: A list containing summaries.
            summary_tokens: A list containing the tokens of the summaries.
            max_src_length (int): Maximum length of the source code.
            token_matrices: A list with the token adjacency matrices for the 
                            code snippets passed as argument.
            statement_matrices: A list with the statement adjacency matrices 
                                for the code snippets passed as argument.
            data_flow_matrices: A list with the data flow adjacency matrices for 
                                the code snippets passed as argument.
            control_flow_matrices: A list with the control flow adjacency matrices 
                                   for the code snippets passed as argument.
            source_vocab: The vocabulary built from the code snippets in training set.
            target_vocab: The vocabulary built from the summaries in training set.
            type (string): Indicates whether we are loading the training set or the
                           validation/testing set.
                           Can be one of the following: "train", "evaluation"
        '''
        super(CustomDataset, self).__init__()

        self.source_code_texts = source_code_texts
        self.summary_texts = summary_texts

        self.source_code_tokens = source_code_tokens
        self.summary_tokens = summary_tokens

        assert len(source_code_texts) == len(summary_texts)
        self.dataset_size = len(source_code_texts)

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.type = type

        self.token_matrices = token_matrices
        self.statement_matrices = statement_matrices
        self.data_flow_matrices = data_flow_matrices
        self.control_flow_matrices = control_flow_matrices

        self.max_src_length = max_src_length

        self.codebert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base", 
                                                                   do_lower_case=False)

    def __len__(self):
        '''
        Returns the number of samples in our dataset.
        '''
        return self.dataset_size

    def __getitem__(self, index):
        '''
        __getitem__ runs on 1 example at a time. Here, we get an example at index 
        and return its numericalize source and target values using the 
        vocabulary objects we created in __init__

        Args:
            index (int): index of the example we want to numericalize

        Returns:
            If this represents the training set:
            - Two tensors with the source code and summary translated to numbers
            Otherwise (if this represents the validation or testing set):
            - The source code, summary and their translation to numbers
        '''
        source_text = self.source_code_texts[index]
        target_text = self.summary_texts[index]

        code_tokens = self.source_code_tokens[index]
        summary_tokens = self.summary_tokens[index]

        token_matrix = self.token_matrices[index]
        statement_matrix = self.statement_matrices[index]
        data_flow_matrix = self.data_flow_matrices[index]
        control_flow_matrix = self.control_flow_matrices[index]

        # numericalize texts ['<BOS>','cat', 'in', 'a', 'bag','<EOS>'] -> [1,12,2,9,24,2]
        numericalized_source = [self.source_vocab.token_to_idx["<BOS>"]]
        numericalized_source += self.source_vocab.numericalize(code_tokens)
        numericalized_source.append(self.source_vocab.token_to_idx["<EOS>"])

        numericalized_target = [self.target_vocab.token_to_idx["<BOS>"]]
        numericalized_target += self.target_vocab.numericalize(summary_tokens)
        numericalized_target.append(self.target_vocab.token_to_idx["<EOS>"])

        df_tensor = build_tensor_from_sparse_matrix(data_flow_matrix, self.max_src_length)
        cf_tensor = build_tensor_from_sparse_matrix(control_flow_matrix, self.max_src_length)

        # Getting the code split into tokens and numericalized for CodeBERT encoder
        source_tokens = self.codebert_tokenizer.tokenize(source_text)[:self.max_src_length - 2]
        source_tokens = [self.codebert_tokenizer.cls_token] + \
                        source_tokens + \
                        [self.codebert_tokenizer.sep_token]
        source_ids =  self.codebert_tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = self.max_src_length - len(source_ids)
        source_ids += [self.codebert_tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        if self.type == 'train':
            # convert the list to tensor and return
            return code_tokens, \
                summary_tokens, \
                torch.tensor(numericalized_source), \
                torch.tensor(numericalized_target), \
                torch.tensor(source_ids), \
                torch.tensor(source_mask), \
                token_matrix.strip().split(), \
                statement_matrix.strip().split(), \
                df_tensor, \
                cf_tensor
        elif self.type == 'evaluation':
            return code_tokens, \
                summary_tokens, \
                source_text, \
                target_text, \
                torch.tensor(numericalized_source), \
                torch.tensor(numericalized_target), \
                torch.tensor(source_ids), \
                torch.tensor(source_mask), \
                token_matrix.strip().split(), \
                statement_matrix.strip().split(), \
                df_tensor, \
                cf_tensor
        else:
            raise ValueError("Invalid type: " + self.type +
                             ". It can only be: train, evaluation.")
