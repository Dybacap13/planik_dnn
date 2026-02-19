#!/bin/bash
set -e
set -u

docker run -it --network=host --rm --volume ${PWD}:/docker-ros/ws/src/tiago-inference --gpus all --name rosml_container tiago_inference /bin/bash
