#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"
CUDA_ENV=""

if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
	CUDA_ENV="-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if [ -z "$GRASPE_CONTAINER_NAME" ]; then
	GRASPE_CONTAINER_NAME="graspe"
fi

docker exec -it $USER $CUDA_ENV --workdir /app/src $(docker ps -aqf "name=^$GRASPE_CONTAINER_NAME\$") python3 ./graspe/run_evaluations.py $@ \
	| grep --line-buffered -vE \
	"BaseCollectiveExecutor::StartAbort|IteratorGetNext|Shape/|Shape_[0-9]+/"
