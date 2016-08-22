#!/usr/bin/env sh
set -e

DOCKER_CMD=docker
IMAGE=caffe:cpu
DOCKER_OPTIONS="--rm -ti --volume=$(pwd)/workspace:/workspace"
DOCKER_RUN="$DOCKER_CMD run $DOCKER_OPTIONS $IMAGE"

# Train net
$DOCKER_RUN bash -c "cd /workspace; ./train.sh"
