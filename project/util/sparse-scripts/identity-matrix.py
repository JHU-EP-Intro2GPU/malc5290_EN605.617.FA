import argparse
import os


def main(args: argparse.Namespace):
    size = args.size
    out_file = args.out_file

    if os.path.exists(out_file):
        print('{} exists, exiting'.format(out_file))
        exit(1)

    with open(out_file, 'w') as file:
        file.write('{}\n'.format(size))
        for i in range(0, size):
            out = ''
            for j in range(0, size):
                out += '{:1.4f}'.format(1 if i is j else 0)
                if j != size - 1:
                    out += ', '
            out += '\n'
            file.write(out)


parser = argparse.ArgumentParser(description="Creates a random nxn matrix with a specified size and random range.")
parser.add_argument('size', type=int, help='The size of the square matrix.')
parser.add_argument('out_file', type=str, help="The location of the output file.")
# parser.add_argument('--min', dest='min', default=0.0,
#                     help="The minimum value that can be generated.")
# parser.add_argument('--max', dest='max', default=1.0,
#                     help="The maximum value that can be generated.")
parser.add_argument('-s', '--seed', help="The seed to use in matrix generator")

parser_args = parser.parse_args()
print(parser_args)
main(parser_args)
