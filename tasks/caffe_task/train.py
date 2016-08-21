#!/usr/bin/env python
"""

train.py is an example of training script with Caffe

"""

import argparse

import caffe


def main(argv):
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
            'solve_file',
            help='Solver prototxt file'
    )
    parser.add_argument(
            '--gpu',
            action='store_true',
            help="Switch for gpu computation."
    )
    args = parser.parse_args()

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")
    
    # define solver
    solver = None
    solver = caffe.get_solver(solve_file)

    # train net
    solver.solve()


if __name__ == '__main__':
    main(sys.argv)
