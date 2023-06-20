import argparse
import json
from tqdm import tqdm


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
                        choices=['python', 'java'],
                        help="Programming language of the dataset")

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
        parser.error("--language python does not require --code_summary_file")

    return args


def main():
    args = parse_arguments()

    # Cleaning file
    open('../data/' + args.language + "/" +
         args.type + '_processed.json', 'w').close()

    dataset = open('../data/' + args.language + "/" +
                   args.type + '_processed.json', 'w')

    if args.language == 'python':
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
                "original_string": code_snippet.strip(), "docstring": summary.strip()}

            json.dump(code_comment, dataset)
            dataset.write("\n")

        code_snippet_file.close()
        summary_file.close()
    elif args.language == 'java':
        code_summary_file = open(args.code_summary_file, 'r')

        num_lines = sum(1 for _ in open(args.code_summary_file, 'r'))

        for _ in tqdm(range(num_lines), desc="Reading java code snippets and summaries"):
            line = code_summary_file.readline()
            code_summary = json.loads(line)

            code_comment = {
                "original_string": code_summary['code'].strip(),
                "docstring": code_summary['comment'].strip()
            }

            json.dump(code_comment, dataset)
            dataset.write("\n")

        code_summary_file.close()

    dataset.close()


if __name__ == '__main__':
    main()
