import argparse
import random
import os


def main(args: argparse.Namespace):
    m = args.size_m
    n = args.size_n
    out_file = args.out_file
    seed = args.seed
    # print('size: {}'.format(size))
    # print('out_file: {}'.format(out_file))
    # print('seed: {}'.format(seed))
    if os.path.exists(out_file):
        print('{} exists, exiting'.format(out_file))
        exit(1)

    if seed:
        random.seed(seed)

    with open(out_file, 'w') as file:
        file.write('{},{}\n'.format(m,n))
        for i in range(0, m):
            out = ''
            for j in range(0, n):
                out += '{:1.4f}'.format(random.random())
                if j != n - 1:
                    out += ', '
            out += '\n'
            file.write(out)


parser = argparse.ArgumentParser(description="Creates a random mx matrix with a specified size and random range.")
parser.add_argument('size_m', type=int, help='The row size of the matrix.')
parser.add_argument('size_n', type=int, help='The column size of the matrix.')
parser.add_argument('out_file', type=str, help="The location of the output file.")
# parser.add_argument('--min', dest='min', default=0.0,
#                     help="The minimum value that can be generated.")
# parser.add_argument('--max', dest='max', default=1.0,
#                     help="The maximum value that can be generated.")
parser.add_argument('-s', '--seed', help="The seed to use in matrix generator")

parser_args = parser.parse_args()
# print(parser_args)
main(parser_args)
