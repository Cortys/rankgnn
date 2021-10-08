#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"
CUDA_ENV=""

if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
	CUDA_ENV="-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if [ -z "$RGNN_CONTAINER_NAME" ]; then
	RGNN_CONTAINER_NAME="rgnn"
fi

docker exec -it $USER $CUDA_ENV --workdir /app/src $(docker ps -aqf "name=^$RGNN_CONTAINER_NAME\$") python3 ./rgnn/run_evaluations.py $@ \
	| grep --line-buffered -vE \
	"BaseCollectiveExecutor::StartAbort|IteratorGetNext|Shape/|Shape_[0-9]+/"
