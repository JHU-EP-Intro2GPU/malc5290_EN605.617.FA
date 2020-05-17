import argparse
import random
import os


def main(args: argparse.Namespace):
    out_file = args.out_file
    if os.path.exists(out_file):
        print('{} exists, exiting'.format(out_file))
        exit(1)

    maxval = None if not args.max else int(args.max)
    m = args.size_m
    n = args.size_n
    density = args.density
    densityn = density/100
    values = [0,1]
    weights = [1 - densityn, densityn]
    
    index = 0
    dist = random.choices(values, weights=weights,k=m * n)

    with open(out_file, 'w') as file:
        file.write('{},{}\n'.format(m,n))
        for i in range(0, m):
            out = ''
            for j in range(0, n):
                if maxval:
                    ival = dist[index] if dist[index] is 0 else random.randrange(maxval + 1)
                    out += str(ival)
                    # print('IVAL: {}'.format(ival))
                else:
                    val = dist[index] if dist[index] is 0 else random.random()
                    out += '{:1.4f}'.format(val)
                    # print('VAL: {}'.format(val))
                index += 1
                if j != n - 1:
                    out += ', '
            out += '\n'
            file.write(out)


parser = argparse.ArgumentParser(description="Creates a random mx matrix with a specified size and random range.")
parser.add_argument('density', type=int, help='Distribution (0 - 100) of zeroes in the matrix.')
parser.add_argument('size_m', type=int, help='The row size of the matrix.')
parser.add_argument('size_n', type=int, help='The column size of the matrix.')
parser.add_argument('out_file', type=str, help="The location of the output file.")
# parser.add_argument('--min', dest='min', default=0.0,
#                     help="The minimum value that can be generated.")
parser.add_argument('--int-max', dest='max',
                     help="The maximum value that can be generated.")

parser_args = parser.parse_args()
# print(parser_args)
main(parser_args)
