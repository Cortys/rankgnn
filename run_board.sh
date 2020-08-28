#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

LOGDIR="/${1:-logs}"

if [ -z "$GRASPE_CONTAINER_NAME" ]; then
	GRASPE_CONTAINER_NAME="graspe"
fi

docker exec -it $USER $(docker ps -aqf "name=^$GRASPE_CONTAINER_NAME\$") tensorboard --logdir $LOGDIR --bind_all
