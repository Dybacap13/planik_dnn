#!/bin/bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

docker build --network host --no-cache -t tiago_dnn_ik -f docker/Dockerfile .