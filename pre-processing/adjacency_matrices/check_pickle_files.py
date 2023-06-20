import pickle
import sys


def check_file(f):
    count = 0
    while 1:
        try:
            pickle.load(f)
            count += 1
        except EOFError:
            break
    print("NUM EXAMPLES:", count)


if __name__ == '__main__':
    data_flow_file = open('files/data_flow/' +
                          sys.argv[1].split('/')[-1], 'rb')
    control_flow_file = open('files/control_flow/' +
                             sys.argv[1].split('/')[-1], 'rb')
    ast_adjacency_matrix_file = open(
        'files/ast_adjacency_matrix/' + sys.argv[1].split('/')[-1], 'rb')

    check_file(data_flow_file)
    check_file(control_flow_file)
    check_file(ast_adjacency_matrix_file)

    data_flow_file.close()
    control_flow_file.close()
    ast_adjacency_matrix_file.close()
