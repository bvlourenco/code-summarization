def CFG_python(root, index_to_code, instructions_code, dependencies):
    if (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        curr_token = index_to_code[(root.start_point, root.end_point)][1]

        if root.start_point[0] + 1 in instructions_code:
            next_tokens = [instructions_code[root.start_point[0] + 1][0][1]]
            next_indexes = [
                instructions_code[root.start_point[0] + 1][0][0] + 1]
        else:
            next_tokens = ['END']
            # END token index is last token index + 1 (all indexes of tokens are shifted by 1)
            next_indexes = [instructions_code[root.start_point[0]][-1][0] + 2]

        # putting control flow edges
        if root.start_point[0] in dependencies and dependencies[root.start_point[0]] in instructions_code:
            next_indexes += [instructions_code[dependencies[root.start_point[0]]][0][0] + 1]
            next_tokens += [instructions_code[dependencies[root.start_point[0]]][0][1]]
        elif root.start_point[0] in dependencies:
            next_tokens += ['END']
            next_indexes += [instructions_code[dependencies[root.start_point[0]] - 1][-1][0] + 2]

        return [(curr_token, index_to_code[(root.start_point, root.end_point)][0] + 1, "Next Instruction", next_tokens, next_indexes)]
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
    if (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        curr_token = index_to_code[(root.start_point, root.end_point)][1]

        if root.start_point[0] + 1 in instructions_code:
            next_tokens = [instructions_code[root.start_point[0] + 1][0][1]]
            next_indexes = [
                instructions_code[root.start_point[0] + 1][0][0] + 1]
        else:
            next_tokens = ['END']
            # END token index is last token index + 1 (all indexes of tokens are shifted by 1)
            next_indexes = [instructions_code[root.start_point[0]][-1][0] + 2]

        # putting control flow edges
        if root.start_point[0] in dependencies and dependencies[root.start_point[0]] in instructions_code:
            next_indexes += [instructions_code[dependencies[root.start_point[0]]][0][0] + 1]
            next_tokens += [instructions_code[dependencies[root.start_point[0]]][0][1]]
        elif root.start_point[0] in dependencies:
            next_tokens += ['END']
            next_indexes += [instructions_code[dependencies[root.start_point[0]] - 1][-1][0] + 2]

        return [(curr_token, index_to_code[(root.start_point, root.end_point)][0] + 1, "Next Instruction", next_tokens, next_indexes)]
    else:
        edges = []
        if root.type == 'while_statement':
            dependencies[root.start_point[0]] = root.end_point[0] + 1
        elif root.type == 'if_statement':
            for child in root.children:
                if child.type == 'else':
                    dependencies[root.start_point[0]] = child.start_point[0]

        if root.type == 'while_statement':
            dependencies[root.children[-1].end_point[0]] = root.start_point[0]

        for child in root.children:
            edges += CFG_java(child, index_to_code,
                              instructions_code, dependencies)
        return edges


def CFG_ruby(root, index_to_code, instructions_code, dependencies):
    if (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        curr_token = index_to_code[(root.start_point, root.end_point)][1]

        if root.start_point[0] + 1 in instructions_code:
            next_tokens = [instructions_code[root.start_point[0] + 1][0][1]]
            next_indexes = [
                instructions_code[root.start_point[0] + 1][0][0] + 1]
        else:
            next_tokens = ['END']
            # END token index is last token index + 1 (all indexes of tokens are shifted by 1)
            next_indexes = [instructions_code[root.start_point[0]][-1][0] + 2]

        # putting control flow edges
        if root.start_point[0] in dependencies and dependencies[root.start_point[0]] in instructions_code:
            next_indexes += [instructions_code[dependencies[root.start_point[0]]][0][0] + 1]
            next_tokens += [instructions_code[dependencies[root.start_point[0]]][0][1]]
        elif root.start_point[0] in dependencies:
            next_tokens += ['END']
            next_indexes += [instructions_code[dependencies[root.start_point[0]] - 1][-1][0] + 2]

        return [(curr_token, index_to_code[(root.start_point, root.end_point)][0] + 1, "Next Instruction", next_tokens, next_indexes)]
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
    if (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        curr_token = index_to_code[(root.start_point, root.end_point)][1]

        if root.start_point[0] + 1 in instructions_code:
            next_tokens = [instructions_code[root.start_point[0] + 1][0][1]]
            next_indexes = [
                instructions_code[root.start_point[0] + 1][0][0] + 1]
        else:
            next_tokens = ['END']
            # END token index is last token index + 1 (all indexes of tokens are shifted by 1)
            next_indexes = [instructions_code[root.start_point[0]][-1][0] + 2]

        # putting control flow edges
        if root.start_point[0] in dependencies and dependencies[root.start_point[0]] in instructions_code:
            next_indexes += [instructions_code[dependencies[root.start_point[0]]][0][0] + 1]
            next_tokens += [instructions_code[dependencies[root.start_point[0]]][0][1]]
        elif root.start_point[0] in dependencies:
            next_tokens += ['END']
            next_indexes += [instructions_code[dependencies[root.start_point[0]] - 1][-1][0] + 2]

        return [(curr_token, index_to_code[(root.start_point, root.end_point)][0] + 1, "Next Instruction", next_tokens, next_indexes)]
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
    if (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        curr_token = index_to_code[(root.start_point, root.end_point)][1]

        if root.start_point[0] + 1 in instructions_code:
            next_tokens = [instructions_code[root.start_point[0] + 1][0][1]]
            next_indexes = [
                instructions_code[root.start_point[0] + 1][0][0] + 1]
        else:
            next_tokens = ['END']
            # END token index is last token index + 1 (all indexes of tokens are shifted by 1)
            next_indexes = [instructions_code[root.start_point[0]][-1][0] + 2]

        # putting control flow edges
        if root.start_point[0] in dependencies and dependencies[root.start_point[0]] in instructions_code:
            next_indexes += [instructions_code[dependencies[root.start_point[0]]][0][0] + 1]
            next_tokens += [instructions_code[dependencies[root.start_point[0]]][0][1]]
        elif root.start_point[0] in dependencies:
            next_tokens += ['END']
            next_indexes += [instructions_code[dependencies[root.start_point[0]] - 1][-1][0] + 2]

        return [(curr_token, index_to_code[(root.start_point, root.end_point)][0] + 1, "Next Instruction", next_tokens, next_indexes)]
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
    if (len(root.children) == 0 or root.type == 'string') and root.type != 'comment' and root.type != '\n':
        curr_token = index_to_code[(root.start_point, root.end_point)][1]

        if root.start_point[0] + 1 in instructions_code:
            next_tokens = [instructions_code[root.start_point[0] + 1][0][1]]
            next_indexes = [
                instructions_code[root.start_point[0] + 1][0][0] + 1]
        else:
            next_tokens = ['END']
            # END token index is last token index + 1 (all indexes of tokens are shifted by 1)
            next_indexes = [instructions_code[root.start_point[0]][-1][0] + 2]

        # putting control flow edges
        if root.start_point[0] in dependencies and dependencies[root.start_point[0]] in instructions_code:
            next_indexes += [instructions_code[dependencies[root.start_point[0]]][0][0] + 1]
            next_tokens += [instructions_code[dependencies[root.start_point[0]]][0][1]]
        elif root.start_point[0] in dependencies:
            next_tokens += ['END']
            next_indexes += [instructions_code[dependencies[root.start_point[0]] - 1][-1][0] + 2]

        return [(curr_token, index_to_code[(root.start_point, root.end_point)][0] + 1, "Next Instruction", next_tokens, next_indexes)]
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
