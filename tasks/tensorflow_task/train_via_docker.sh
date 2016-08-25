#!/usr/bin/env sh
set -e

DOCKER_CMD=docker
IMAGE=tf:cpu
DOCKER_OPTIONS="--rm -ti --volume=$(pwd)/workspace:/workspace --workdir=/workspace"
DOCKER_RUN="$DOCKER_CMD run $DOCKER_OPTIONS $IMAGE"

# Train net
$DOCKER_RUN python /workspace/train.py
