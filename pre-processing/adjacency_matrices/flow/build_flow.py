import sys
import traceback
from flow.tree_helpers import tree_to_token_index
from flow.utils import index_to_code_token, print_ast, remove_comments_and_docstrings

def build_flow(codeT, parser, lang, build_flow_graph, type):
    '''
    FROM: https://github.com/microsoft/CodeBERT/blob/
          eb1c6e5dc40d996665504173244d274484b2f7e9/GraphCodeBERT/translation/run.py#L74
    '''
    # remove comments
    try:
        code = remove_comments_and_docstrings(codeT, lang)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
    # obtain dataflow
    if lang == "php":
        code = "<?php"+code+"?>"
    try:
        tree = parser.parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        # print(root_node.text)
        # print_ast(root_node)
        tokens_index = tree_to_token_index(root_node)

        # print("TOKENS INDEX:", tokens_index)
        # for el in tokens_index:
        #     print(el)

        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]

        # print("\nCODE TOKENS:", code_tokens)
        # for el in code_tokens:
        #     print(el)
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)

        # print("\nINDEX TO CODE:", index_to_code)
        # for key, value in index_to_code.items():
        #     print(key, ':', value)
        # print('')

        instructions_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            if index[0][0] not in instructions_code:
                instructions_code[index[0][0]] = [(idx, code)]
            else:
                instructions_code[index[0][0]] += [(idx, code)]

        # print("\nINSTRUCTIONS CODE:", instructions_code)
        # for key, value in instructions_code.items():
        #     print(key, ':', value)
        # print('')

        try:
            if type == 'Data flow':
                graph, _ = build_flow_graph(root_node, index_to_code, {})
            elif type == 'Control flow':
                graph = [(('START', 0, 'Next Instruction', [
                          instructions_code[0][0][1]], [1]))]
                graph += build_flow_graph(root_node,
                                          index_to_code, instructions_code, {})
                graph += [(('END', instructions_code[max(k for k, _ in instructions_code.items())]
                           [-1][0] + 2, 'Next Instruction', [], []))]
                # for edge in graph:
                #     print(edge)
            else:
                raise ValueError(
                    "Wrong type for data/control flow matrix: ", type)
        except Exception as e:
            print(codeT, "\n")
            traceback.print_exc()
            graph = []
            sys.exit(1)

        graph = sorted(graph, key=lambda x: x[1])
        indexs = set()
        for edge in graph:
            if len(edge[-1]) != 0:
                indexs.add(edge[1])
            for x in edge[-1]:
                indexs.add(x)
        new_graph = []
        for edge in graph:
            if edge[1] in indexs:
                new_graph.append(edge)
        graph = new_graph
    except Exception as e:
        traceback.print_exc()
        graph = []
        code_tokens = []
        sys.exit(1)

    return code_tokens, graph
