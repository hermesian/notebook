#!/usr/bin/env sh
set -e

# Train net
curl -L -H 'Content-Type: application/json' -X POST -d@chronos_tf_train.json http:///scheduler/iso8601
