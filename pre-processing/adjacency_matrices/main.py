import argparse
import json
import logging
import os
import pickle
import numpy as np
from tqdm import tqdm

from tree_sitter import Language, Parser
from flow.data_flow.data_flow import DFG_go, DFG_java, DFG_javascript, DFG_php, DFG_python, DFG_ruby
from flow.control_flow.control_flow import CFG_go, CFG_java, CFG_javascript, CFG_php, CFG_python, CFG_ruby
from language_parsers.build.build_language import build_language_library
from code_snippet import CodeSnippet

logger = logging.getLogger(__name__)

# FROM: https://github.com/microsoft/CodeBERT/blob/
# eb1c6e5dc40d996665504173244d274484b2f7e9/GraphCodeBERT/codesearch/run.py#L48
DFG_FUNCTION = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}

CFG_FUNCTION = {
    'python': CFG_python,
    'java': CFG_java,
    'ruby': CFG_ruby,
    'go': CFG_go,
    'php': CFG_php,
    'javascript': CFG_javascript
}


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--filename", default=None, type=str, required=True,
                        help="File with code snippets (json file with key \
                        'original_string' and value code snippet).")
    parser.add_argument("--parsers_path", default="language_parsers/build/languages.so",
                        type=str, required=False, help="Path of file storing parsers \
                            of programming languages.")
    parser.add_argument("--must_process", action='store_true',
                        help="Indicates that code snippets contain special \
                            characters that must be pre-processed (DCNL and DCSP).")
    parser.add_argument("--language", default=None, required=True,
                        choices=['go', 'java', 'javascript',
                                 'php', 'python', 'ruby'],
                        help="Programming language of the code snippets")

    return parser.parse_args()


def setup_logging_print_parsers(parsers_path):
    np.set_printoptions(threshold=np.inf)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    if not os.path.exists(parsers_path):
        build_language_library()


def get_parser(parsers_path, language):
    parser = Parser()
    parser.set_language(Language(parsers_path, language))
    return parser


def pre_process(line):
    line = line.replace('qz', 'd')
    line = line.replace('qq', 'q')
    line = line.replace(' DCNL DCSP ', '\n\t')
    line = line.replace(' DCNL ', '\n')
    line = line.replace(' DCSP ', '\t')
    return line


def read_file(f, must_process, language, parser, log_file, filename):
    filename_no_extension = filename.split('/')[-1].split('.')[0]

    in_token_file = open('files/in_token/' +
                         filename_no_extension + '_' + language + '.txt', 'w')
    in_statement_file = open('files/in_statement/' +
                             filename_no_extension + '_' + language + '.txt', 'w')
    data_flow_file = open('files/data_flow/' +
                          filename_no_extension + '_' + language + '.txt', 'wb')
    control_flow_file = open('files/control_flow/' +
                             filename_no_extension + '_' + language + '.txt', 'wb')
    ast_adjacency_matrix_file = open(
        'files/ast_adjacency_matrix/' + filename_no_extension + '_' + language + '.txt', 'wb')

    num_lines = sum(1 for _ in open(filename, 'r'))

    for line in tqdm(f, total=num_lines, desc="Reading code snippets"):
        if must_process:
            code = pre_process(json.loads(line)['original_string'])
        else:
            code = json.loads(line)['original_string']

        code_snippet = CodeSnippet(code,
                                   language,
                                   parser,
                                   DFG_FUNCTION[language],
                                   CFG_FUNCTION[language],
                                   log_file)

        adjacency_matrices = code_snippet.get_adjacency_matrices()
        store_adjacency_matrices(adjacency_matrices,
                                 in_token_file,
                                 in_statement_file,
                                 data_flow_file,
                                 control_flow_file,
                                 ast_adjacency_matrix_file)

    in_token_file.close()
    in_statement_file.close()
    data_flow_file.close()
    control_flow_file.close()
    ast_adjacency_matrix_file.close()


def read_code_snippets(filename, must_process, language, parser, log_file):
    with open(filename, 'r') as f:
        code_snippets = read_file(
            f, must_process, language, parser, log_file, filename)
    return code_snippets


def write_matrix_to_file(type, matrix, file):
    if type in ['in_token', 'in_statement']:
        for element in matrix:
            file.write(str(element) + ' ')
        file.write('\n')
    elif type in ['data_flow', 'control_flow', 'ast_adjacency_matrix']:
        pickle.dump(matrix, file, pickle.DEFAULT_PROTOCOL)
    else:
        raise ValueError('Unrecognized type to store as adjacency matrix')


def store_adjacency_matrices(adjacency_matrices, in_token_file,
                             in_statement_file, data_flow_file,
                             control_flow_file, ast_adjacency_matrix_file):
    write_matrix_to_file('in_token',
                         adjacency_matrices['in_token_adjacency_matrix'],
                         in_token_file)
    write_matrix_to_file('in_statement',
                         adjacency_matrices['in_statement_adjacency_matrix'],
                         in_statement_file)
    write_matrix_to_file('data_flow',
                         adjacency_matrices['data_flow_adjacency_matrix'],
                         data_flow_file)
    write_matrix_to_file('control_flow',
                         adjacency_matrices['control_flow_adjacency_matrix'],
                         control_flow_file)
    write_matrix_to_file('ast_adjacency_matrix',
                         adjacency_matrices['ast_adjacency_matrix'],
                         ast_adjacency_matrix_file)


def main():
    args = parse_arguments()

    setup_logging_print_parsers(args.parsers_path)

    parser = get_parser(args.parsers_path, args.language)

    filename_no_extension = args.filename.split('/')[-1].split('.')[0]

    with open('files/logs/' + filename_no_extension + '_' + args.language + '_log.txt', 'w') as log_file:
        read_code_snippets(args.filename, args.must_process,
                           args.language, parser, log_file)


if __name__ == '__main__':
    main()
