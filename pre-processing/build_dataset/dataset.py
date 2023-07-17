import argparse
import string
from typing import List
import code_tokenize as ctok
import json
import re
import requests
from tqdm import tqdm
from requests.exceptions import HTTPError


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

    parser.add_argument('--chatgpt_comments', type=str2bool, required=True,
                        help="If true, it gets comments for the code snippets \
                              using chatGPT")

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
                   if token.__str__() not in ["#NEWLINE#", "#DEDENT#", "#INDENT#"]]

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


def tokenize_docstring(docstring: str) -> List[str]:
    '''
    Source: https://github.com/github/CodeSearchNet/blob/106e827405c968597da938f6b373d30183918869/function_parser/function_parser/parsers/language_parser.py
    '''
    return [t for t in DOCSTRING_REGEX_TOKENIZER.findall(docstring) if t is not None and len(t) > 0]


def perform_request(input):
    headers = {'Cookie': '_ga_MWL7THFH58=GS1.1.1689235868.2.1.1689236089.0.0.0; _ga=GA1.1.1559918855.1689156808; __stripe_mid=08fe76e1-1a30-4881-b263-a3cdf3ebfc027e45ab; __Host-next-auth.csrf-token=b99a0f66c5957d8e04508a04bafc04c0fc5f701f0cd4b6514964479fc1426011%7Cd1b629481cc19d321616b66a7f23ea4e1327d8a46640e98dc52f39dd239d30d0; __Secure-next-auth.callback-url=https%3A%2F%2Fora.ai%2F; __cf_bm=VnBduHvz3qSx8BCbrkoZfCQPw7XmEPHeKirERYbk.nE-1689235868-0-AQm8y9jG4samxymw1rVNHta7dOBnF1djk2su4zFjH138AbB9JhLb4amLpKashfMpNA==; __stripe_sid=419d4f5e-ccd7-4c38-bce9-bce8803fe89ea73ccf; __Secure-next-auth.session-token=eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..U04dYn_PRGB4Zmcx._VbhQtHt9j6wmIqf-uPPqXPGvGkmbGAL6Gp0HqISbpRj-hajMvSzFpcvhF9Z0Jb1lVgNpfqCd1Aqo65XgHheNDtyZa50djIsfqQ9QpXxEtXDjln3Bd4u20reXBctxw8yI8Q_NZHowI30dFOwWcd4nAlTsyFWcssmiTrm4J1oNaF8MY0qtCD_FDVyEg2XBb-d4Cu_q2TkAwvwNByTL5QVKNVtHyvhHdZGn5HZzqKBOXwb57xwGwe8OLWeO2x7EdS4JpQ3OHH-rXpLsPBnDy81R4UqqhyNbL3-Jo23rTFZ4x9KT8CTR4o89eIkR_dwhlWbyrKJu4K9lZ3Z4oMFAZ4sEbZTH34c4S8fE26fwPwPBqXm3ZQ0mEfFhd8Ju6Zdp-tcQFp5FJJb5WX3kYerP55J2qXTia0BbZcuJ_CQzdWycLb-rqagpPDA0X8S1Wype7QBkBASopmWzyvY78_q7pom0DbjS4vmPhiQN0Ow5JzrK0aSSpSQH05jZVW2VKrJp3FwDs4Is8OpoDvCZPqVP-psCKFzagNctDYOXdpolWhfpkvIbhfSKpntU6rtHKNlqeqLhuGi8jxeV7fE9tNWwouPporbsEU-z0lFtS7bc5gfr1mmp-0nD3f-_n1x91dH7-3JlwZlKtU2LexlkZGmeFpUgdabAkJ71ismKkM3SVqLTp0z0jLfEw.E8cxPmQeKOqcmARyUVDTtQ'}
    payload = {
        "chatbotId": "66381037-ea3c-4f7d-b87c-ba9b759eae3a",
        "conversationId": "979802f1-6bc8-44b9-b91e-1089a812bd54",
        "userId": "08b85e39-f118-48bf-8c59-39429f18cf68",
        "provider": "OPEN_AI",
        "config": False,
        "includeHistory": False
    }
    payload["input"] = input
    response = requests.post("https://ora.ai/api/conversation",
                             headers=headers,
                             json=payload)
    response.raise_for_status()
    comment = response.json()["response"]

    if comment.startswith("This code snippet "):
        comment = comment.replace("This code snippet ", "")
    elif comment.startswith("This code "):
        comment = comment.replace("This code ", "")
    elif comment.startswith("The code "):
        comment = comment.replace("The code ", "")

    return comment


def get_comment(code_snippet):
    try:
        comment = ""
        goodComment = False
        while not goodComment:
            comment = perform_request(code_snippet)
            tokens = comment.split()
            if len(tokens) > 48:
                comment = perform_request(
                    "Summarize the code comment below in less than 48 words:\n" + comment)

                tokens = comment.split()
                if len(tokens) <= 48:
                    goodComment = True
            else:
                goodComment = True
        return comment
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')


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

            code_tokens = tokenize_code(code_snippet.strip(), args.language)
            if len(code_tokens) > MAX_SRC_LEN:
                code_tokens = [token for token in code_tokens if token not in string.punctuation]

            code_comment = {
                "original_string": code_snippet.strip(),
                "docstring": summary.strip(),
                "code_tokens": code_tokens,
                "docstring_tokens": tokenize_docstring(summary.strip())
            }

            if args.chatgpt_comments:
                comment = get_comment(code_snippet.strip())

                code_comment["docstring_chatGPT"] = comment.strip()
                code_comment["docstring_chatGPT_tokens"] = tokenize_docstring(
                    comment.strip())

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

            if len(code_tokens) > MAX_SRC_LEN:
                code_tokens = [token for token in code_tokens if token not in string.punctuation]
            code_comment = {
                "original_string": code_summary[code_key].strip(),
                "docstring": code_summary[summary_key].strip(),
                "code_tokens": code_tokens,
                "docstring_tokens": tokenize_docstring(code_summary[summary_key].strip())
            }

            if args.chatgpt_comments:
                comment = get_comment(code_summary[code_key].strip())

                code_comment["docstring_chatGPT"] = comment.strip()
                code_comment["docstring_chatGPT_tokens"] = tokenize_docstring(
                    comment.strip())

            json.dump(code_comment, dataset)
            dataset.write("\n")

        code_summary_file.close()

    dataset.close()


if __name__ == '__main__':
    main()
