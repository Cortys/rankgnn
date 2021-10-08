#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

if [ "$1" == "root" ]; then
	USER="-u 0:0"
fi

if [ -z "$RGNN_CONTAINER_NAME" ]; then
	RGNN_CONTAINER_NAME="rgnn"
fi

docker exec -it $USER $(docker ps -aqf "name=^$RGNN_CONTAINER_NAME\$") bash
