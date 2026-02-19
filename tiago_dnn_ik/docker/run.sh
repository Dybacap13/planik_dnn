#!/bin/bash
set -e
set -u

if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --rm -w /home/tiago_dnn_ik --volume ${PWD}:/home/tiago_dnn_ik --gpus all --name tiago_dnn_ik_container tiago_dnn_ik /bin/bash
else
    export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	xhost +
	docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e NVIDIA_VISIBLE_DEVICES=all -w /home/tiago_dnn_ik --volume ${PWD}:/home/tiago_dnn_ik --runtime=nvidia --gpus=all --name=tiago_dnn_ik_container tiago_dnn_ik /bin/bash
	xhost -
fi
