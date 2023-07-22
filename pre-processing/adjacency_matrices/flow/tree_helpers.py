import string
from tokenization import tokenize_with_camel_case, tokenize_with_snake_case


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment' and root_node.type != '\n' and root_node.type not in string.punctuation:
        node_text_camel_case = get_node_text_snake_camel(root_node)
        if len(node_text_camel_case) > 1:
            token_idxs = []
            begin = root_node.start_point
            # To keep track of the total length of tokens that are already in token_idxs
            tokens_len = begin[1]
            for token in node_text_camel_case:
                if token == '_':
                    tokens_len += 1
                    continue

                token_idxs.append(
                    ((begin[0], tokens_len), (begin[0], tokens_len + len(token))))
                tokens_len += len(token)
            return token_idxs
        else:
            return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def get_node_text_snake_camel(root_node):
    node_text = root_node.text.decode('utf-8')
    node_text_snake_case = tokenize_with_snake_case(
        node_text, keep_delimiter=True)
    node_text_camel_case = []
    for token in node_text_snake_case:
        node_text_camel_case.extend(tokenize_with_camel_case(token))
    return node_text_camel_case


def get_var_index_leaf_node(root_node, index_to_code, start, end):
    index = (start, end)
    if index not in index_to_code:
        return []

    _, code = index_to_code[index]
    if root_node.type != code:
        return [(start, end)]
    else:
        return []


def tree_to_variable_index(root_node, index_to_code):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment' and root_node.type != '\n' and root_node.type not in string.punctuation:
        node_text_camel_case = get_node_text_snake_camel(root_node)
        if len(node_text_camel_case) > 1:
            var_idxs = []
            begin = root_node.start_point
            # To keep track of the total length of tokens that are already in token_idxs
            tokens_len = begin[1]
            for token in node_text_camel_case:
                if token == '_':
                    tokens_len += 1
                    continue

                start = (begin[0], tokens_len)
                end = (begin[0], tokens_len + len(token))
                var_idxs += get_var_index_leaf_node(root_node, index_to_code,
                                                    start, end)
                tokens_len += len(token)
            return var_idxs
        else:
            return get_var_index_leaf_node(root_node, index_to_code,
                                           root_node.start_point,
                                           root_node.end_point)
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_variable_index(child, index_to_code)
        return code_tokens
