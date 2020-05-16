import sys
import random
import os


def main(data: {}):
    datatype = data['datatype']
    m = data['m']
    n = data['n']
    p = data['p']
    time_load = float(data['time_load'])
    time_exec = float(data['time_exec'])
    time_read = float(data['time_read'])
    
    mult_lib = data['mult_lib']
    out_file = 'mat_{}_{}_{}_data-{}.csv'.format(m,n,p,datatype)

    if not os.path.exists(out_file):
        print('creating {}'.format(out_file))
        with open(out_file, 'w') as file:
            file.write('mat_lib,time_load,time_exec,time_read,time_total\n')

    with open(out_file, 'a') as file:
        file.write('{},{},{},{},{}\n'.format(mult_lib,time_load,time_exec,time_read, (time_load + time_exec + time_read)))

rawinput = sys.stdin.read()
rawinput = rawinput.split(' ')

data = {
    'mult_lib': rawinput[0],
    'datatype': rawinput[1],
    'm': rawinput[2],
    'n': rawinput[3],
    'p': rawinput[4],
    'time_load': rawinput[5],
    'time_exec': rawinput[6],
    'time_read': rawinput[7],
}

main(data)
