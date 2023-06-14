import os
from tqdm import tqdm


def load_dataset_file(code_filename, summary_filename, type, debug_max_lines):
    '''
    Given a file with code snippets and a file with summaries, it loads all 
    examples of them from the respective files.

    Args:
        code_filename (string): The filename of the file with code snippets.
        summary_filename (string): The filename of the file with summaries.
        type (string): Indicates whether we are loading the training set or the
                       validation set from the files.
                       Can be one of the following: "train", "validation"
        debug_max_lines (int): Represents the number of examples we want to read
                               from the dataset. If we pass a non-positive value, 
                               the whole dataset will be read.

    Returns:
        A list of code snippets examples and a list of summaries examples.
    '''
    if (not os.path.exists(code_filename)):
        raise ValueError("code snippet filename does not exist")

    if (not os.path.exists(summary_filename)):
        raise ValueError("summary filename does not exist")

    with open(code_filename) as code_file:
        num_code_lines = sum(1 for _ in open(code_filename, 'r'))
        if debug_max_lines > 0:
            source_code_texts = [next(code_file) for _ in tqdm(range(debug_max_lines),
                                                               total=debug_max_lines,
                                                               desc="Reading " + str(debug_max_lines) + " " + type + " code snippets")]
        else:
            source_code_texts = [line.strip() for line in tqdm(
                code_file, total=num_code_lines, desc="Reading " + type + " code snippets")]

    with open(summary_filename) as summary_file:
        num_summary_lines = sum(1 for _ in open(summary_filename, 'r'))
        if debug_max_lines > 0:
            summary_texts = [next(summary_file) for _ in tqdm(range(debug_max_lines),
                                                              total=debug_max_lines,
                                                              desc="Reading " + str(debug_max_lines) + " " + type + " summaries")]
        else:
            summary_texts = [line.strip() for line in tqdm(
                summary_file, total=num_summary_lines, desc="Reading " + type + " summaries")]

    return source_code_texts, summary_texts
