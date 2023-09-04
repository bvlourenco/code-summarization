import string
from flow.tree_helpers import get_node_text_snake_camel


def add_cfg_edges(index_to_code, instructions_code, dependencies, start, end):
    curr_token = index_to_code[(start, end)][1]
    next_tokens, next_indexes = [], []

    next_instructions = list(filter(lambda x: x > start[0], instructions_code.keys()))
    if len(next_instructions) > 0:
        next_instruction = min(next_instructions)
        next_tokens = [token_pos[1] for token_pos in instructions_code[next_instruction]]
        next_indexes = [token_pos[0] for token_pos in instructions_code[next_instruction]]

    # putting control flow edges
    if start[0] in dependencies: 
        dependency_next_instrs = list(filter(lambda x: x >= dependencies[start[0]], instructions_code.keys()))
        if len(dependency_next_instrs) > 0:
            dependency_next_instr = min(dependency_next_instrs)

            next_indexes += [token_pos[0] for token_pos in instructions_code[dependency_next_instr]]
            next_tokens += [token_pos[1] for token_pos in instructions_code[dependency_next_instr]]

    return [(curr_token, index_to_code[(start, end)][0], "Next Instruction", next_tokens, next_indexes)]

def process_leaf_node(root, index_to_code, instructions_code, dependencies):
    if (root.start_point, root.end_point) in index_to_code:
        return add_cfg_edges(index_to_code, instructions_code, dependencies, root.start_point, root.end_point)
    else:
        # The AST does not split tokens in snake_case and camelCase and since it
        # is impossible to change the AST, we need to split those tokens here to
        # make sure that a snake_case/camelCase token get split into multiple
        # "nodes" (1 per sub-token) 
        node_text_camel_case = get_node_text_snake_camel(root)
        if len(node_text_camel_case) > 1:
            begin = root.start_point
            # To keep track of the total length of tokens that are already in token_idxs
            tokens_len = begin[1]
            edges = []
            for token in node_text_camel_case:
                if token == '_':
                    tokens_len += 1
                    continue
                
                start = (begin[0], tokens_len)
                end = (begin[0], tokens_len + len(token))
                edges += add_cfg_edges(index_to_code, instructions_code, dependencies, start, end)

                tokens_len += len(token)            
            return edges
        else:
            # This case should not happen
            print("Returning empty list...")
            return []


def CFG_python(root, index_to_code, instructions_code, dependencies):
    if root.type in string.punctuation:
        return []
    elif (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        return process_leaf_node(root, index_to_code, instructions_code, dependencies)
    else:
        edges = []
        if root.type == 'while_statement' or root.type == 'if_statement':
            dependencies[root.start_point[0]] = root.end_point[0] + 1

        if root.type == 'while_statement':
            dependencies[root.children[-1].end_point[0]] = root.start_point[0]

        for child in root.children:
            edges += CFG_python(child, index_to_code,
                                instructions_code, dependencies)
        return edges


def CFG_java(root, index_to_code, instructions_code, dependencies):
    if root.type in string.punctuation:
        return []
    elif (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        return process_leaf_node(root, index_to_code, instructions_code, dependencies)
    else:
        edges = []
        if root.type in ['while_statement', 'for_statement', 'enhanced_for_statement']:
            dependencies[root.start_point[0]] = root.end_point[0] + 1
        elif root.type == 'if_statement':
            hasElse = False
            for child in root.children:
                if child.type == 'else':
                    hasElse = True
                    dependencies[root.start_point[0]] = child.start_point[0]
            if not hasElse:
                dependencies[root.start_point[0]] = root.end_point[0] + 1

        if root.type in ['while_statement', 'for_statement', 'enhanced_for_statement']:
            dependency_next_instrs = list(filter(lambda x: x <= root.children[-1].end_point[0], instructions_code.keys()))
            if len(dependency_next_instrs) > 0:
                dependency_next_instr = max(dependency_next_instrs)
                dependencies[dependency_next_instr] = root.start_point[0]

        for child in root.children:
            edges += CFG_java(child, index_to_code,
                              instructions_code, dependencies)
        return edges


def CFG_ruby(root, index_to_code, instructions_code, dependencies):
    if root.type in string.punctuation:
        return []
    elif (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        return process_leaf_node(root, index_to_code, instructions_code, dependencies)
    else:
        edges = []
        if root.type == 'while':
            dependencies[root.start_point[0]] = root.end_point[0] + 1
        elif root.type == 'if':
            for child in root.children:
                if child.type == 'else':
                    dependencies[root.start_point[0]] = child.start_point[0]

        if root.type == 'while':
            dependencies[root.children[-1].end_point[0]] = root.start_point[0]

        for child in root.children:
            edges += CFG_ruby(child, index_to_code,
                              instructions_code, dependencies)
        return edges


def CFG_go(root, index_to_code, instructions_code, dependencies):
    if root.type in string.punctuation:
        return []
    elif (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        return process_leaf_node(root, index_to_code, instructions_code, dependencies)
    else:
        edges = []
        if root.type == 'for_statement':
            dependencies[root.start_point[0]] = root.end_point[0] + 1
        elif root.type == 'if_statement':
            for child in root.children:
                if child.type == 'else':
                    dependencies[root.start_point[0]] = child.start_point[0]

        if root.type == 'for_statement':
            dependencies[root.children[-1].end_point[0]] = root.start_point[0]

        for child in root.children:
            if child.type == '\n':
                continue
            edges += CFG_go(child, index_to_code,
                              instructions_code, dependencies)
        return edges


def CFG_php(root, index_to_code, instructions_code, dependencies):
    if root.type in string.punctuation:
        return []
    elif (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        return process_leaf_node(root, index_to_code, instructions_code, dependencies)
    else:
        edges = []
        if root.type == 'while_statement':
            dependencies[root.start_point[0]] = root.end_point[0] + 1
        elif root.type == 'if_statement':
            for child in root.children:
                if child.type == 'else_clause':
                    dependencies[root.start_point[0]] = child.start_point[0]

        if root.type == 'while_statement':
            dependencies[root.children[-1].end_point[0]] = root.start_point[0]

        for child in root.children:
            edges += CFG_php(child, index_to_code,
                              instructions_code, dependencies)
        return edges


def CFG_javascript(root, index_to_code, instructions_code, dependencies):
    if root.type in string.punctuation:
        return []
    elif (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        return process_leaf_node(root, index_to_code, instructions_code, dependencies)
    else:
        edges = []
        if root.type == 'while_statement':
            dependencies[root.start_point[0]] = root.end_point[0] + 1
        elif root.type == 'if_statement':
            for child in root.children:
                if child.type == 'else_clause':
                    dependencies[root.start_point[0]] = child.start_point[0]

        if root.type == 'while_statement':
            dependencies[root.children[-1].end_point[0]] = root.start_point[0]

        for child in root.children:
            edges += CFG_javascript(child, index_to_code,
                              instructions_code, dependencies)
        return edges
