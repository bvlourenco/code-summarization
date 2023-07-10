'''
FROM: https://github.com/kkcookies99/UAST/blob/
      fff81885aa07901786141a71e5600a08d7cb4868/gen_graph.py#LL1C1-L1C1
'''
import re
import numpy as np
from flow.tree_helpers import get_node_text_snake_camel


def get_node_representation(root):
    if root.type != '(' and root.type != ')':
        return root.type
    elif root.type == '(':
        return '<PARENTHESIS_LEFT>'
    elif root.type == ')':
        return '<PARENTHESIS_RIGHT>'


def structure_based_traversal(root):
    '''
    ADAPTED from Deep Code Comment Generation (algorithm 1)
    '''
    seq = ''
    if len(root.children) == 0:
        node_text_camel_case = get_node_text_snake_camel(root)
        if len(node_text_camel_case) > 1:
            for token in node_text_camel_case:
                if token == '_':
                    continue

                seq += '(' + get_node_representation(root) + ')'
        else:
            seq = '(' + get_node_representation(root) + ')'
    else:
        seq = '(' + get_node_representation(root)
        for child in root.children:
            seq += structure_based_traversal(child)
        seq += ')'
    return seq


def get_nodes_index(ast_path):
    ast_path_list = []
    ast_path_split = re.split('[()]', ast_path)
    for i in ast_path_split:
        i = i.strip()
        if i != "":
            ast_path_list.append(i)
    nodes_num = {}
    for i in range(len(ast_path_list)):
        if ast_path_list[i] not in nodes_num:
            nodes_num[ast_path_list[i]] = [i]
        else:
            nodes_num[ast_path_list[i]].append(i)
    return nodes_num


def update_AST_entries(root, child, nodes_num, max_len, adj_matrix):
    if get_node_representation(root) in nodes_num and get_node_representation(child) in nodes_num:
        i = nodes_num[get_node_representation(root)][0]
        j = nodes_num[get_node_representation(child)][0]
        if i >= max_len or j >= max_len:
            return adj_matrix
        adj_matrix[i][j] = 1
        if len(nodes_num[get_node_representation(child)]) > 1:
            nodes_num[get_node_representation(
                child)] = nodes_num[get_node_representation(child)][1:]
    return adj_matrix

def build_AST_adjacency_matrix(root, nodes_num, max_len=150):
    adj_matrix = np.eye(max_len)
    if len(root.children) == 0 and root.type != '\n':
        return adj_matrix
    else:
        for child in root.children:
            # The AST does not split tokens in snake_case and camelCase and since it
            # is impossible to change the AST, we need to split those tokens here to
            # make sure that a snake_case/camelCase token get split into multiple AST
            # "nodes" (1 per sub-token)
            if len(child.children) == 0:
                node_text_camel_case = get_node_text_snake_camel(child)
                if len(node_text_camel_case) > 1:
                    for token in node_text_camel_case:
                        if token == '_':
                            continue
                        adj_matrix = update_AST_entries(root, child, nodes_num, max_len, adj_matrix)
                else:
                    adj_matrix = update_AST_entries(root, child, nodes_num, max_len, adj_matrix)
            else:
                adj_matrix = update_AST_entries(root, child, nodes_num, max_len, adj_matrix)
            adj_matrix += build_AST_adjacency_matrix(child, nodes_num)
    return adj_matrix


def get_AST_adjacency_matrix(tree):
    ast_path = structure_based_traversal(tree.root_node)
    nodes_num = get_nodes_index(ast_path)
    adj_matrix = build_AST_adjacency_matrix(tree.root_node, nodes_num)
    # Putting diagonal entries with value 1
    np.fill_diagonal(adj_matrix, 1)
    return adj_matrix
