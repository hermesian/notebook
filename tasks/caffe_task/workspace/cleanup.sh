#!/usr/bin/env sh
set -e

rm lenet_iter_10000.caffemodel
rm lenet_iter_10000.solverstate
rm lenet_iter_5000.caffemodel
rm lenet_iter_5000.solverstate
rm -rf  mnist_test_lmdb
rm -rf  mnist_train_lmdb
rm t10k-images-idx3-ubyte
rm t10k-labels-idx1-ubyte
rm train-images-idx3-ubyte
rm train-labels-idx1-ubyte
