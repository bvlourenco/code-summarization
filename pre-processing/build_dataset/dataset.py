import argparse
import string
from typing import List
import code_tokenize as ctok
import json
import re
from tqdm import tqdm


DOCSTRING_REGEX_TOKENIZER = re.compile(
    r"[^\s,'\"`.():\[\]=*;>{\}+-/\\]+|\\+|\.+|\(\)|{\}|\[\]|\(+|\)+|:+|\[+|\]+|{+|\}+|=+|\*+|;+|>+|\++|-+|/+")
MAX_SRC_LEN = 150


def str2bool(value):
    '''
    Parses a given value to a boolean. Used to parse booleans in argument parser.

    Args:
        value: The value to be parsed to a boolean.
    '''
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def pre_process(line):
    line = line.replace('qz', 'd')
    line = line.replace('qq', 'q')
    line = line.replace(' DCNL DCSP ', '\n\t')
    line = line.replace(' DCNL ', '\n')
    line = line.replace(' DCSP ', '\t')
    return line


def parse_arguments():
    parser = argparse.ArgumentParser(
        'Converts python/java dataset files into the structure \
         {\'original_string\': <CODE>, \'docstring\': <SUMMARY>}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--language', type=str, required=True,
                        choices=['python', 'java', 'go',
                                 'ruby', 'php', 'javascript'],
                        help="Programming language of the dataset")
    parser.add_argument('--codesearchnet', type=str2bool, default=False,
                        help="Tells whether we're loading a file from \
                              CodeSearchNet dataset or not")

    # Only used with python dataset
    parser.add_argument('--code_snippet_file', type=str, default=None,
                        help="Filename of the code snippets file (for Python dataset)")

    parser.add_argument('--summary_file', type=str, default=None,
                        help="Filename of the summaries file (for Python dataset)")

    # Only used with java dataset
    parser.add_argument('--code_summary_file', type=str, default=None,
                        help="Filename of the file with code snippets and "
                             "summaries (for Java dataset)")

    parser.add_argument('--type', type=str, required=True,
                        choices=['test', 'train', 'validation'],
                        help="Type of the dataset file (if it belongs to the \
                            training set, testing set or validation set)")
    parser.add_argument('--pre_processing', type=str2bool, default=False,
                        help='If true, tells to pre-process the code snippets by \
                            replacing the DCNL and DCSP tokens by newline and tab \
                            characters')

    args = parser.parse_args()

    if not args.codesearchnet:
        if args.language == 'java' and args.code_summary_file is None:
            parser.error("--language java requires --code_summary_file")
        elif args.language == 'python' and (args.code_snippet_file is None or
                                            args.summary_file is None):
            parser.error("--language python requires --code_snippet_file and"
                         " --summary_file")
        elif args.language == 'java' and (args.code_snippet_file is not None or
                                          args.summary_file is not None):
            parser.error("--language java does not require --code_snippet_file and"
                         " --summary_file")
        elif args.language == 'python' and args.code_summary_file is not None:
            parser.error(
                "--language python does not require --code_summary_file")
    else:
        if args.code_summary_file is None:
            parser.error("--codesearchnet (with any language) requires "
                         "--code_summary_file")
        elif args.code_snippet_file is not None or args.summary_file is not None:
            parser.error("--codesearchnet does not require --code_snippet_file "
                         "and --summary_file")

    return args


def tokenize_with_camel_case(token):
    matches = re.finditer(
        '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', token)
    return [m.group(0) for m in matches]


def tokenize_with_snake_case(token):
    return token.split('_')


def tokenize_code(code_snippet, language, camel_case=True, snake_case=True):
    code_tokens = ctok.tokenize(
        code_snippet, lang=language, syntax_error="ignore")

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

    # Lowercasing here to split camel case words
    code_tokens = [token.__str__().lower() for token in code_tokens]
    return code_tokens


def tokenize_docstring(docstring: str) -> List[str]:
    '''
    Source: https://github.com/github/CodeSearchNet/blob/106e827405c968597da938f6b373d30183918869/function_parser/function_parser/parsers/language_parser.py
    '''
    return [t for t in DOCSTRING_REGEX_TOKENIZER.findall(docstring) if t is not None and len(t) > 0]


def main():
    args = parse_arguments()

    if args.codesearchnet:
        path = 'CodeSearchNet/dataset/' + args.language
    else:
        path = args.language

    # Cleaning file
    open('../../data/' + path + "/" + args.type + '_processed.json', 'w').close()

    dataset = open('../../data/' + path + "/" +
                   args.type + '_processed.json', 'w')

    if path == 'python':
        code_snippet_file = open(args.code_snippet_file, 'r')
        summary_file = open(args.summary_file, 'r')

        num_lines = sum(1 for _ in open(args.code_snippet_file, 'r'))

        # Reset the file pointer to the beggining
        code_snippet_file.seek(0)

        for _ in tqdm(range(num_lines), desc="Reading python code snippets and summaries"):
            code_snippet = code_snippet_file.readline()
            summary = summary_file.readline()

            if code_snippet is None or summary is None:
                raise ValueError(
                    "Code snippet and summary files with different size")

            if args.pre_processing:
                code_snippet = pre_process(code_snippet)

            code_comment = {
                "original_string": code_snippet.strip(),
                "docstring": summary.strip(),
                "code_tokens": tokenize_code(code_snippet.strip(), args.language),
                "docstring_tokens": tokenize_docstring(summary.strip())
            }

            json.dump(code_comment, dataset)
            dataset.write("\n")

        code_snippet_file.close()
        summary_file.close()
    elif path == 'java' or path == 'CodeSearchNet/dataset/' + args.language:

        if path == 'java':
            code_key = 'code'
            summary_key = 'comment'
        elif path == 'CodeSearchNet/dataset/' + args.language:
            code_key = 'original_string'
            summary_key = 'docstring'

        code_summary_file = open(args.code_summary_file, 'r')

        num_lines = sum(1 for _ in open(args.code_summary_file, 'r'))

        for _ in tqdm(range(num_lines), desc=f"Reading CodeSearchNet " +
                      f"{args.language} code snippets and summaries"):
            line = code_summary_file.readline()
            code_summary = json.loads(line)

            code_comment = {
                "original_string": code_summary[code_key].strip(),
                "docstring": code_summary[summary_key].strip(),
                "code_tokens": tokenize_code(code_summary[code_key].strip(), args.language),
                "docstring_tokens": tokenize_docstring(code_summary[summary_key].strip())
            }

            json.dump(code_comment, dataset)
            dataset.write("\n")

        code_summary_file.close()

    dataset.close()


if __name__ == '__main__':
    main()
