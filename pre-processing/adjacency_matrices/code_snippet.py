import string
import numpy as np
# from ast_matrix.ast_adjacency_matrix import get_AST_adjacency_matrix
from flow.build_flow import build_flow
from scipy import sparse
import code_tokenize as ctok
from tokenization import tokenize_snake_camel_case, tokenize_with_snake_case, tokenize_with_camel_case

MAX_SRC_LEN = 150


class CodeSnippet(object):

    def __init__(self, code_snippet, language, parser, dfg_function, cfg_function, log_file, debug=False):
        self.code_snippet = code_snippet
        self.language = language
        self.parser = parser
        self.dfg_function = dfg_function
        self.cfg_function = cfg_function
        self.in_token_adjacency_matrix = []
        self.in_statement_adjacency_matrix = []
        self.data_flow_adjacency_matrix = np.zeros((MAX_SRC_LEN, MAX_SRC_LEN))
        self.control_flow_adjacency_matrix = np.zeros(
            (MAX_SRC_LEN, MAX_SRC_LEN))
        # self.ast_adjacency_matrix = np.eye(MAX_SRC_LEN)
        self.log_file = log_file
        self.debug = debug
        self.tokens = self.tokenize_code(self.code_snippet)

    def tokenize_code(self, code_snippet, camel_case=True, snake_case=True):
        code_tokens = ctok.tokenize(code_snippet,
                                    lang=self.language,
                                    syntax_error="ignore")

        # Removing some python unwanted tokens representing \n, \t and untabbing.
        code_tokens = [token.__str__() for token in code_tokens
                       if token.__str__() not in ["#NEWLINE#", "#DEDENT#", "#INDENT#"]
                       and token.__str__() not in string.punctuation]

        if snake_case:
            snake_case_tokenized = []
            for token in code_tokens:
                tokens_snake_case = tokenize_with_snake_case(token)
                for token_snake_case in tokens_snake_case:
                    snake_case_tokenized.append(token_snake_case)
            code_tokens = snake_case_tokenized

        if camel_case:
            camel_case_tokenized = []
            for token in code_tokens:
                tokens_camel_case = tokenize_with_camel_case(token)
                for token_camel_case in tokens_camel_case:
                    camel_case_tokenized.append(token_camel_case)
            code_tokens = camel_case_tokenized

        return code_tokens

    def build_token_matrix(self):
        code_tokens = self.tokenize_code(self.code_snippet,
                                         camel_case=False,
                                         snake_case=False)

        camel_snake_case_tokenized = []
        number_snake_camel_case_tokens = 0

        for token in code_tokens:
            subtokens = tokenize_snake_camel_case(token)

            if len(subtokens) == 1:
                val = 0
            else:
                number_snake_camel_case_tokens += 1
                val = number_snake_camel_case_tokens

            camel_snake_case_tokenized.extend(
                (subtoken, val) for subtoken in subtokens)

        self.in_token_adjacency_matrix.extend(
            el[1] for el in camel_snake_case_tokenized)

    def build_statement_matrix(self):
        # CodeSearchNet, TL-CodeSum and code-docstring-corpus datasets splits
        # instructions with '\n'. So we do not need to split code by tokens
        # such as ;, { and } in Java (and other languages)
        instructions = self.code_snippet.split('\n')
        for i in range(len(instructions)):
            if len(instructions[i].strip()) == 0:
                continue

            num_tokens = len(self.tokenize_code(instructions[i]))
            self.in_statement_adjacency_matrix.extend(
                [i for _ in range(num_tokens)])

    def update_flow_matrix(self, type, i, j):
        if type == 'Data flow':
            self.data_flow_adjacency_matrix[i, j] = 1
        elif type == 'Control flow':
            self.control_flow_adjacency_matrix[i, j] = 1
        else:
            raise ValueError("Wrong type for data/control flow matrix: ", type)

    def get_build_flow_function(self, type):
        if type == 'Data flow':
            return self.dfg_function
        elif type == 'Control flow':
            return self.cfg_function
        else:
            raise ValueError("Wrong type for data/control flow matrix: ", type)

    def print_flow_matrix(self, type):
        if type == 'Data flow':
            self.log_file.write(type + " matrix:\n" +
                                self.data_flow_adjacency_matrix.__str__() + "\n\n")
        elif type == 'Control flow':
            self.log_file.write(type + " matrix:\n" +
                                self.control_flow_adjacency_matrix.__str__() + "\n\n")
        else:
            raise ValueError("Wrong type for data/control flow matrix: ", type)

    def build_data_or_control_flow_matrix(self, type):
        '''
        Either builds data flow or control flow 
        '''
        build_flow_graph = self.get_build_flow_function(type)
        _, edges = build_flow(
            self.code_snippet, self.parser, self.language, build_flow_graph, type)
        for edge in edges:
            for dependency in edge[4]:
                if edge[1] < MAX_SRC_LEN and dependency < MAX_SRC_LEN:
                    self.update_flow_matrix(type, edge[1], dependency)

        if self.debug:
            self.log_file.write("\n\n\n" + self.code_snippet + "\n\n")
            self.log_file.write(type + " edges:\n" + edges.__str__() + "\n\n")
            self.print_flow_matrix(type)

    # def build_ast_matrix(self):
    #     tree = self.parser.parse(bytes(self.code_snippet, "utf8"))
    #     self.ast_adjacency_matrix = get_AST_adjacency_matrix(tree)

    #     if self.debug:
    #         self.log_file.write("\n\n\n" + self.code_snippet + "\n\n")
    #         self.log_file.write("AST adjacency matrix:\n" +
    #                             self.ast_adjacency_matrix.__str__() + "\n\n")
    #         self.log_file.write("AST:\n" + tree.root_node.sexp() + "\n\n\n")

    def get_adjacency_matrices(self):
        self.build_token_matrix()
        self.build_statement_matrix()
        self.build_data_or_control_flow_matrix('Data flow')
        self.build_data_or_control_flow_matrix('Control flow')
        # self.build_ast_matrix()

        return {
            "in_token_adjacency_matrix": self.in_token_adjacency_matrix,
            "in_statement_adjacency_matrix": self.in_statement_adjacency_matrix,
            "data_flow_adjacency_matrix": sparse.csr_matrix(self.data_flow_adjacency_matrix),
            "control_flow_adjacency_matrix": sparse.csr_matrix(self.control_flow_adjacency_matrix),
            # "ast_adjacency_matrix": sparse.csr_matrix(self.ast_adjacency_matrix),
        }
