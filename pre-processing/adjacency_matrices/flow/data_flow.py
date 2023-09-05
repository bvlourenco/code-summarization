'''
FROM: https://github.com/microsoft/CodeBERT/blob/
      eb1c6e5dc40d996665504173244d274484b2f7e9/GraphCodeBERT/translation/parser/
      DFG.py#LL1C1-L1C1
'''

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import string
from flow.tree_helpers import tree_to_variable_index, get_node_text_snake_camel

def add_dfg_edges(root_node, index_to_code, states, start, end, root_node_text):
    idx, code = index_to_code[(start, end)]
    if root_node.type == code:
        return [], states
    elif root_node_text in states and idx not in states[root_node_text]:
        return [(code, idx, 'comesFrom', [code], states[root_node_text].copy())], states
    else:
        if root_node.type == 'identifier':
            node_text_camel_case = get_node_text_snake_camel(root_node)
            idxs = []
            for i in range(len(node_text_camel_case)):
                idxs.append(idx + i)
            states[root_node_text] = idxs
        return [(code, idx, 'comesFrom', [], [])], states

def process_leaf_node(root_node, index_to_code, states):
    if (root_node.start_point, root_node.end_point) not in index_to_code:
        # The AST does not split tokens in snake_case and camelCase and since it
        # is impossible to change the AST, we need to split those tokens here to
        # make sure that a snake_case/camelCase token get split into multiple
        # "nodes" (1 per sub-token) 
        node_text_camel_case = get_node_text_snake_camel(root_node)
        if len(node_text_camel_case) > 1:
            begin = root_node.start_point
            # To keep track of the total length of tokens that are already in token_idxs
            tokens_len = begin[1]
            edges = []
            for token in node_text_camel_case:
                if token == '_':
                    tokens_len += 1
                    continue
                
                start = (begin[0], tokens_len)
                end = (begin[0], tokens_len + len(token))
                new_edges, states = add_dfg_edges(root_node, index_to_code, states, start, end, root_node.text.decode())
                edges.extend(new_edges)

                tokens_len += len(token)            
            return edges, states
        else:
            return [], states
    else:
        return add_dfg_edges(root_node, index_to_code, states, root_node.start_point, root_node.end_point, root_node.text.decode())

def DFG_python(root_node, index_to_code, states):
    assignment = ['assignment', 'augmented_assignment', 'for_in_clause']
    if_statement = ['if_statement']
    for_statement = ['for_statement']
    while_statement = ['while_statement']
    do_first_statement = ['for_in_clause']
    def_statement = ['default_parameter']
    states = states.copy()
    if root_node.type in string.punctuation:
        return [], states
    elif (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment' and root_node.type != '\n':
        return process_leaf_node(root_node, index_to_code, states)
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        DFG = []
        if value is None:
            # method name should not have dependencies
            if root_node.start_point[0] != 0:
                indexs = tree_to_variable_index(name, index_to_code)
                all_idxs = []
                for index in indexs:
                    idx, code = index_to_code[index]
                    DFG.append((code, idx, 'comesFrom', [], []))
                    all_idxs.append(idx)
                states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_python(value, index_to_code, states)
            DFG += temp
            all_idxs = []
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                all_idxs.append(idx1)
            states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        if root_node.type == 'for_in_clause':
            right_nodes = [root_node.children[-1]]
            left_nodes = [root_node.child_by_field_name('left')]
        else:
            if root_node.child_by_field_name('right') is None:
                return [], states
            left_nodes = [x for x in root_node.child_by_field_name(
                'left').children if x.type != ',']
            right_nodes = [x for x in root_node.child_by_field_name(
                'right').children if x.type != ',']
            if len(right_nodes) != len(left_nodes):
                left_nodes = [root_node.child_by_field_name('left')]
                right_nodes = [root_node.child_by_field_name('right')]
            if len(left_nodes) == 0:
                left_nodes = [root_node.child_by_field_name('left')]
            if len(right_nodes) == 0:
                right_nodes = [root_node.child_by_field_name('right')]
            
            # Avoiding splitting string nodes into " string_content "
            if root_node.child_by_field_name('left').type == 'string':
                left_nodes = [root_node.child_by_field_name('left')]
            if root_node.child_by_field_name('right').type == 'string':
                right_nodes = [root_node.child_by_field_name('right')]

        DFG = []
        for node in right_nodes:
            temp, states = DFG_python(node, index_to_code, states)
            DFG += temp

        for left_node, right_node in zip(left_nodes, right_nodes):
            left_tokens_index = tree_to_variable_index(
                left_node, index_to_code)
            right_tokens_index = tree_to_variable_index(
                right_node, index_to_code)
            temp = []
            all_idxs = []
            for token1_index in left_tokens_index:
                idx1, code1 = index_to_code[token1_index]
                temp.append((code1, idx1, 'computedFrom', [index_to_code[x][1] for x in right_tokens_index],
                             [index_to_code[x][0] for x in right_tokens_index]))
                all_idxs.append(idx1)
            states[right_node.text.decode()] = all_idxs
            DFG += temp
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in ['elif_clause', 'else_clause']:
                temp, current_states = DFG_python(
                    child, index_to_code, current_states)
                DFG += temp
            else:
                temp, new_states = DFG_python(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for i in range(2):
            right_nodes = [x for x in root_node.child_by_field_name(
                'right').children if x.type != ',']
            left_nodes = [x for x in root_node.child_by_field_name(
                'left').children if x.type != ',']
            if len(right_nodes) != len(left_nodes):
                left_nodes = [root_node.child_by_field_name('left')]
                right_nodes = [root_node.child_by_field_name('right')]
            if len(left_nodes) == 0:
                left_nodes = [root_node.child_by_field_name('left')]
            if len(right_nodes) == 0:
                right_nodes = [root_node.child_by_field_name('right')]
            
            # Avoiding splitting string nodes into " string_content "
            if root_node.child_by_field_name('left').type == 'string':
                left_nodes = [root_node.child_by_field_name('left')]
            if root_node.child_by_field_name('right').type == 'string':
                right_nodes = [root_node.child_by_field_name('right')]

            for node in right_nodes:
                temp, states = DFG_python(node, index_to_code, states)
                DFG += temp
            for left_node, right_node in zip(left_nodes, right_nodes):
                left_tokens_index = tree_to_variable_index(
                    left_node, index_to_code)
                right_tokens_index = tree_to_variable_index(
                    right_node, index_to_code)
                temp = []
                all_idxs = []
                for token1_index in left_tokens_index:
                    idx1, code1 = index_to_code[token1_index]
                    temp.append((code1, idx1, 'computedFrom', [index_to_code[x][1] for x in right_tokens_index],
                                 [index_to_code[x][0] for x in right_tokens_index]))
                    all_idxs.append(idx1)
                states[left_node.text.decode()] = all_idxs
                DFG += temp
            if root_node.children[-1].type == "block":
                temp, states = DFG_python(
                    root_node.children[-1], index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in while_statement:
        DFG = []
        for i in range(2):
            for child in root_node.children:
                temp, states = DFG_python(child, index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_python(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_python(child, index_to_code, states)
                DFG += temp

        return sorted(DFG, key=lambda x: x[1]), states


def DFG_java(root_node, index_to_code, states):
    assignment = ['assignment_expression']
    def_statement = ['variable_declarator']
    increment_statement = ['update_expression']
    if_statement = ['if_statement', 'else']
    for_statement = ['for_statement']
    enhanced_for_statement = ['enhanced_for_statement']
    while_statement = ['while_statement']
    do_first_statement = []
    states = states.copy()
    if root_node.type in string.punctuation:
        return [], states
    elif (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment' and root_node.type != '\n':
        return process_leaf_node(root_node, index_to_code, states)
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        DFG = []
        if value is None:
            # method name should not have dependencies
            if root_node.start_point[0] != 0:
                indexs = tree_to_variable_index(name, index_to_code)
                all_idxs = []
                for index in indexs:
                    idx, code = index_to_code[index]
                    DFG.append((code, idx, 'comesFrom', [], []))
                    all_idxs.append(idx)
                states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_java(value, index_to_code, states)
            DFG += temp
            all_idxs = []
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                all_idxs.append(idx1)
            states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        DFG = []
        temp, states = DFG_java(right_nodes, index_to_code, states)
        DFG += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        all_idxs = []
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            all_idxs.append(idx1)
        states[right_nodes.text.decode()] = all_idxs
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in increment_statement:
        DFG = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        all_idxs = []
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            all_idxs.append(idx1)
        states[root_node.text.decode()] = all_idxs
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        flag = False
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and flag is False:
                temp, current_states = DFG_java(
                    child, index_to_code, current_states)
                DFG += temp
            else:
                flag = True
                temp, new_states = DFG_java(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for child in root_node.children:
            temp, states = DFG_java(child, index_to_code, states)
            DFG += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
            elif child.type == "local_variable_declaration":
                flag = True
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in enhanced_for_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        body = root_node.child_by_field_name('body')
        DFG = []
        for i in range(2):
            temp, states = DFG_java(value, index_to_code, states)
            DFG += temp
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            all_idxs = []
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                all_idxs.append(idx1)
            states[name.text.decode()] = all_idxs
            temp, states = DFG_java(body, index_to_code, states)
            DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in while_statement:
        DFG = []
        for i in range(2):
            for child in root_node.children:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_java(child, index_to_code, states)
                DFG += temp

        return sorted(DFG, key=lambda x: x[1]), states


def DFG_ruby(root_node, index_to_code, states):
    assignment = ['assignment', 'operator_assignment']
    if_statement = ['if', 'elsif', 'else', 'unless', 'when']
    for_statement = ['for']
    while_statement = ['while_modifier', 'until']
    do_first_statement = []
    def_statement = ['keyword_parameter']
    states = states.copy()
    if root_node.type in string.punctuation:
        return [], states
    elif (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment' and root_node.type != '\n':
        return process_leaf_node(root_node, index_to_code, states)
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        DFG = []
        if value is None:
            # method name should not have dependencies
            if root_node.start_point[0] != 0:
                indexs = tree_to_variable_index(name, index_to_code)
                all_idxs = []
                for index in indexs:
                    idx, code = index_to_code[index]
                    DFG.append((code, idx, 'comesFrom', [], []))
                    all_idxs.append(idx)
                states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_ruby(value, index_to_code, states)
            DFG += temp
            all_idxs = []
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                all_idxs.append(idx1)
            states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        left_nodes = [x for x in root_node.child_by_field_name(
            'left').children if x.type != ',']
        right_nodes = [x for x in root_node.child_by_field_name(
            'right').children if x.type != ',']
        if len(right_nodes) != len(left_nodes):
            left_nodes = [root_node.child_by_field_name('left')]
            right_nodes = [root_node.child_by_field_name('right')]
        if len(left_nodes) == 0:
            left_nodes = [root_node.child_by_field_name('left')]
        if len(right_nodes) == 0:
            right_nodes = [root_node.child_by_field_name('right')]
        
        # Avoiding splitting string nodes into " string_content "
        if root_node.child_by_field_name('left').type == 'string':
            left_nodes = [root_node.child_by_field_name('left')]
        if root_node.child_by_field_name('right').type == 'string':
            right_nodes = [root_node.child_by_field_name('right')]

        if root_node.type == "operator_assignment":
            left_nodes = [root_node.children[0]]
            right_nodes = [root_node.children[-1]]

        DFG = []
        for node in right_nodes:
            temp, states = DFG_ruby(node, index_to_code, states)
            DFG += temp

        for left_node, right_node in zip(left_nodes, right_nodes):
            left_tokens_index = tree_to_variable_index(
                left_node, index_to_code)
            right_tokens_index = tree_to_variable_index(
                right_node, index_to_code)
            temp = []
            all_idxs = []
            for token1_index in left_tokens_index:
                idx1, code1 = index_to_code[token1_index]
                temp.append((code1, idx1, 'computedFrom', [index_to_code[x][1] for x in right_tokens_index],
                             [index_to_code[x][0] for x in right_tokens_index]))
                all_idxs.append(idx1)
            states[left_node.text.decode()] = all_idxs
            DFG += temp
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement:
                temp, current_states = DFG_ruby(
                    child, index_to_code, current_states)
                DFG += temp
            else:
                temp, new_states = DFG_ruby(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for i in range(2):
            left_nodes = [root_node.child_by_field_name('pattern')]
            right_nodes = [root_node.child_by_field_name('value')]
            assert len(right_nodes) == len(left_nodes)
            for node in right_nodes:
                temp, states = DFG_ruby(node, index_to_code, states)
                DFG += temp
            for left_node, right_node in zip(left_nodes, right_nodes):
                left_tokens_index = tree_to_variable_index(
                    left_node, index_to_code)
                right_tokens_index = tree_to_variable_index(
                    right_node, index_to_code)
                temp = []
                all_idxs = []
                for token1_index in left_tokens_index:
                    idx1, code1 = index_to_code[token1_index]
                    temp.append((code1, idx1, 'computedFrom', [index_to_code[x][1] for x in right_tokens_index],
                                 [index_to_code[x][0] for x in right_tokens_index]))
                    all_idxs.append(idx1)
                states[left_node.text.decode()] = all_idxs  
                DFG += temp
            temp, states = DFG_ruby(root_node.child_by_field_name(
                'body'), index_to_code, states)
            DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in while_statement:
        DFG = []
        for i in range(2):
            for child in root_node.children:
                temp, states = DFG_ruby(child, index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_ruby(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_ruby(child, index_to_code, states)
                DFG += temp

        return sorted(DFG, key=lambda x: x[1]), states


def DFG_go(root_node, index_to_code, states):
    assignment = ['assignment_statement',]
    def_statement = ['var_spec']
    increment_statement = ['inc_statement']
    if_statement = ['if_statement', 'else']
    for_statement = ['for_statement']
    enhanced_for_statement = []
    while_statement = []
    do_first_statement = []
    states = states.copy()
    if root_node.type in string.punctuation:
        return [], states
    elif (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment' and root_node.type != '\n':
        return process_leaf_node(root_node, index_to_code, states)
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        DFG = []
        if value is None:
            # method name should not have dependencies
            if root_node.start_point[0] != 0:
                indexs = tree_to_variable_index(name, index_to_code)
                all_idxs = []
                for index in indexs:
                    idx, code = index_to_code[index]
                    DFG.append((code, idx, 'comesFrom', [], []))
                    all_idxs.append(idx)
                states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_go(value, index_to_code, states)
            DFG += temp
            all_idxs = []
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                all_idxs.append(idx1)
            states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        DFG = []
        temp, states = DFG_go(right_nodes, index_to_code, states)
        DFG += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        all_idxs = []
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            all_idxs.append(idx1)
        states[name.text.decode()] = all_idxs
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in increment_statement:
        DFG = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            all_idxs.append(idx1)
        states[root_node.text.decode()] = all_idxs
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        flag = False
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and flag is False:
                temp, current_states = DFG_go(
                    child, index_to_code, current_states)
                DFG += temp
            else:
                flag = True
                temp, new_states = DFG_go(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in states:
            if key not in new_states:
                new_states[key] = states[key]
            else:
                new_states[key] += states[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for child in root_node.children:
            temp, states = DFG_go(child, index_to_code, states)
            DFG += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = DFG_go(child, index_to_code, states)
                DFG += temp
            elif child.type == "for_clause":
                if child.child_by_field_name('update') is not None:
                    temp, states = DFG_go(child.child_by_field_name(
                        'update'), index_to_code, states)
                    DFG += temp
                flag = True
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_go(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_go(child, index_to_code, states)
                DFG += temp

        return sorted(DFG, key=lambda x: x[1]), states


def DFG_php(root_node, index_to_code, states):
    assignment = ['assignment_expression', 'augmented_assignment_expression']
    def_statement = ['simple_parameter']
    increment_statement = ['update_expression']
    if_statement = ['if_statement', 'else_clause']
    for_statement = ['for_statement']
    enhanced_for_statement = ['foreach_statement']
    while_statement = ['while_statement']
    do_first_statement = []
    states = states.copy()
    if root_node.type in string.punctuation:
        return [], states
    elif (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment' and root_node.type != '\n':
        return process_leaf_node(root_node, index_to_code, states)
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('default_value')
        DFG = []
        if value is None:
            # method name should not have dependencies
            if root_node.start_point[0] != 0:
                indexs = tree_to_variable_index(name, index_to_code)
                all_idxs = []
                for index in indexs:
                    idx, code = index_to_code[index]
                    DFG.append((code, idx, 'comesFrom', [], []))
                    all_idxs.append(idx)
                states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_php(value, index_to_code, states)
            DFG += temp
            all_idxs = []
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                all_idxs.append(idx1)
            states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        DFG = []
        temp, states = DFG_php(right_nodes, index_to_code, states)
        DFG += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        all_idxs = []
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            all_idxs.append(idx1)
        states[name.text.decode()] = all_idxs
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in increment_statement:
        DFG = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        all_idxs = []
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            all_idxs.append(idx1)
        states[root_node.text.decode()] = all_idxs
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        flag = False
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and flag is False:
                temp, current_states = DFG_php(
                    child, index_to_code, current_states)
                DFG += temp
            else:
                flag = True
                temp, new_states = DFG_php(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in states:
            if key not in new_states:
                new_states[key] = states[key]
            else:
                new_states[key] += states[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for child in root_node.children:
            temp, states = DFG_php(child, index_to_code, states)
            DFG += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = DFG_php(child, index_to_code, states)
                DFG += temp
            elif child.type == "assignment_expression":
                flag = True
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in enhanced_for_statement:
        name = None
        value = None
        for child in root_node.children:
            if child.type == 'variable_name' and value is None:
                value = child
            elif child.type == 'variable_name' and name is None:
                name = child
                break
        body = root_node.child_by_field_name('body')
        DFG = []
        for i in range(2):
            temp, states = DFG_php(value, index_to_code, states)
            DFG += temp
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            all_idxs = []
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
                all_idxs.append(idx1)
            states[name.text.decode()] = all_idxs
            temp, states = DFG_php(body, index_to_code, states)
            DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in while_statement:
        DFG = []
        for i in range(2):
            for child in root_node.children:
                temp, states = DFG_php(child, index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_php(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_php(child, index_to_code, states)
                DFG += temp

        return sorted(DFG, key=lambda x: x[1]), states


def DFG_javascript(root_node, index_to_code, states):
    assignment = ['assignment_pattern', 'augmented_assignment_expression']
    def_statement = ['variable_declarator']
    increment_statement = ['update_expression']
    if_statement = ['if_statement', 'else']
    for_statement = ['for_statement']
    enhanced_for_statement = []
    while_statement = ['while_statement']
    do_first_statement = []
    states = states.copy()
    if root_node.type in string.punctuation:
        return [], states
    elif (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment' and root_node.type != '\n':
        return process_leaf_node(root_node, index_to_code, states)
    elif root_node.type in def_statement:
        name = root_node.child_by_field_name('name')
        value = root_node.child_by_field_name('value')
        DFG = []
        if value is None:
            # method name should not have dependencies
            if root_node.start_point[0] != 0:
                indexs = tree_to_variable_index(name, index_to_code)
                all_idxs = []
                for index in indexs:
                    idx, code = index_to_code[index]
                    DFG.append((code, idx, 'comesFrom', [], []))
                    all_idxs.append(idx)
                states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
        else:
            name_indexs = tree_to_variable_index(name, index_to_code)
            value_indexs = tree_to_variable_index(value, index_to_code)
            temp, states = DFG_javascript(value, index_to_code, states)
            DFG += temp
            all_idxs = []
            for index1 in name_indexs:
                idx1, code1 = index_to_code[index1]
                for index2 in value_indexs:
                    idx2, code2 = index_to_code[index2]
                    DFG.append((code1, idx1, 'comesFrom', [code2], [idx2]))
                all_idxs.append(idx1)
            states[name.text.decode()] = all_idxs
            return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in assignment:
        left_nodes = root_node.child_by_field_name('left')
        right_nodes = root_node.child_by_field_name('right')
        DFG = []
        temp, states = DFG_javascript(right_nodes, index_to_code, states)
        DFG += temp
        name_indexs = tree_to_variable_index(left_nodes, index_to_code)
        value_indexs = tree_to_variable_index(right_nodes, index_to_code)
        all_idxs = []
        for index1 in name_indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in value_indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            all_idxs.append(idx1)
        states[left_nodes.text.decode()] = all_idxs
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in increment_statement:
        DFG = []
        indexs = tree_to_variable_index(root_node, index_to_code)
        all_idxs = []
        for index1 in indexs:
            idx1, code1 = index_to_code[index1]
            for index2 in indexs:
                idx2, code2 = index_to_code[index2]
                DFG.append((code1, idx1, 'computedFrom', [code2], [idx2]))
            states[code1] = [idx1]
            all_idxs.append(idx1)
        states[root_node.text.decode()] = all_idxs
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in if_statement:
        DFG = []
        current_states = states.copy()
        others_states = []
        flag = False
        tag = False
        if 'else' in root_node.type:
            tag = True
        for child in root_node.children:
            if 'else' in child.type:
                tag = True
            if child.type not in if_statement and flag is False:
                temp, current_states = DFG_javascript(
                    child, index_to_code, current_states)
                DFG += temp
            else:
                flag = True
                temp, new_states = DFG_javascript(child, index_to_code, states)
                DFG += temp
                others_states.append(new_states)
        others_states.append(current_states)
        if tag is False:
            others_states.append(states)
        new_states = {}
        for dic in others_states:
            for key in dic:
                if key not in new_states:
                    new_states[key] = dic[key].copy()
                else:
                    new_states[key] += dic[key]
        for key in states:
            if key not in new_states:
                new_states[key] = states[key]
            else:
                new_states[key] += states[key]
        for key in new_states:
            new_states[key] = sorted(list(set(new_states[key])))
        return sorted(DFG, key=lambda x: x[1]), new_states
    elif root_node.type in for_statement:
        DFG = []
        for child in root_node.children:
            temp, states = DFG_javascript(child, index_to_code, states)
            DFG += temp
        flag = False
        for child in root_node.children:
            if flag:
                temp, states = DFG_javascript(child, index_to_code, states)
                DFG += temp
            elif child.type == "variable_declaration":
                flag = True
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    elif root_node.type in while_statement:
        DFG = []
        for i in range(2):
            for child in root_node.children:
                temp, states = DFG_javascript(child, index_to_code, states)
                DFG += temp
        dic = {}
        for x in DFG:
            if (x[0], x[1], x[2]) not in dic:
                dic[(x[0], x[1], x[2])] = [x[3], x[4]]
            else:
                dic[(x[0], x[1], x[2])][0] = list(
                    set(dic[(x[0], x[1], x[2])][0]+x[3]))
                dic[(x[0], x[1], x[2])][1] = sorted(
                    list(set(dic[(x[0], x[1], x[2])][1]+x[4])))
        DFG = [(x[0], x[1], x[2], y[0], y[1])
               for x, y in sorted(dic.items(), key=lambda t:t[0][1])]
        return sorted(DFG, key=lambda x: x[1]), states
    else:
        DFG = []
        for child in root_node.children:
            if child.type in do_first_statement:
                temp, states = DFG_javascript(child, index_to_code, states)
                DFG += temp
        for child in root_node.children:
            if child.type not in do_first_statement:
                temp, states = DFG_javascript(child, index_to_code, states)
                DFG += temp

        return sorted(DFG, key=lambda x: x[1]), states
