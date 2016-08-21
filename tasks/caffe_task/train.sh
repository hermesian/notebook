#!/usr/bin/env sh
set -e

DOCKER_CMD=docker
IMAGE=caffe:cpu
DOCKER_OPTIONS="--rm -ti --volume=$(pwd):/workspace --workdir=/workspace"
DOCKER_RUN="$DOCKER_CMD run $DOCKER_OPTIONS $IMAGE"

# Download the data
$DOCKER_RUN bash -c "cd workspace;
                     ./get_mnist.sh"

# Create the LMDB database
$DOCKER_RUN bash -c "cd workspace;
                     ./create_mnist.sh"

# Train net
#$DOCKER_RUN bash -c "cp \$CAFFE_ROOT/examples/mnist/lenet_solver.prototxt ./;
#                     cp \$CAFFE_ROOT/examples/mnist/lenet_train_test.prototxt ./"
$DOCKER_RUN caffe train --solver=workspace/solver.prototxt $*
