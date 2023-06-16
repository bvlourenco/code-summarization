import json
import os
from tqdm import tqdm


def load_dataset_file(dataset_filename, type, debug_max_lines):
    '''
    Given a file with code snippets and a file with summaries, it loads all 
    examples of them from the respective files.

    Args:
        dataset_filename (string): The filename of the dataset.
        summary_filename (string): The filename of the file with summaries.
        type (string): Indicates whether we are loading the training set, the
                       validation set or the testing set from the files.
                       Can be one of the following: "train", "validation", "test"
        debug_max_lines (int): Represents the number of examples we want to read
                               from the dataset. If we pass a non-positive value, 
                               the whole dataset will be read.

    Returns:
        A list of code snippets examples and a list of summaries examples.
    '''
    if (not os.path.exists(dataset_filename)):
        raise ValueError("dataset filename does not exist")

    source_code_texts, summary_texts = [], []
    with open(dataset_filename) as dataset_file:
        if debug_max_lines > 0:
            num_lines = debug_max_lines
            description = "Reading {} {} entries of the dataset".format(debug_max_lines, type)
            lines_to_read = range(debug_max_lines)
        else:
            num_lines = sum(1 for _ in open(dataset_filename, 'r'))
            description = "Reading {} dataset".format(type)
            lines_to_read = dataset_file
        
        for line in tqdm(lines_to_read, total=num_lines, desc=description):
            if debug_max_lines > 0:
                line = json.loads(next(dataset_file))
            else:
                line = json.loads(line)
            source_code_texts.append(line['original_string'])
            summary_texts.append(line['docstring'])

    return source_code_texts, summary_texts
