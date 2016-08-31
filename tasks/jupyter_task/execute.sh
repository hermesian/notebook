!#/usr/bin/bash

set -x

curl -X POST -H "Content-Type: application/json" http://marathon/v2/apps -d@marathon_jupyter_task.json | jq .
