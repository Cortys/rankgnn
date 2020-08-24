#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

if [ "$1" == "root" ]; then
	USER="-u 0:0"
fi

if [ -z "$GRASPE_CONTAINER_NAME" ]; then
	GRASPE_CONTAINER_NAME="graspe"
fi

docker exec -it $USER $(docker ps -aqf "name=^$GRASPE_CONTAINER_NAME\$") bash
