#!/usr/bin/env sh
set -e

# Download the data
./get_mnist.sh

# Create the LMDB database
./create_mnist.sh

# Train net
caffe train --solver=solver.prototxt $*
