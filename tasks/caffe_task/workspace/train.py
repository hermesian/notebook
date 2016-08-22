#!/usr/bin/env python
"""

train.py is an example of training script with Caffe

"""

import argparse

import caffe


def main():
        
    caffe.set_mode_cpu()
    print("CPU mode")
    
    # define solver
    solver = None
    solver = caffe.get_solver("./solver.prototxt")

    # train net
    solver.solve()


if __name__ == '__main__':
    main()
