#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

if [ -z "$GRASPE_CONTAINER_NAME" ]; then
	GRASPE_CONTAINER_NAME="graspe"
fi

if [ -z "$REPL_PORT" ]; then
	REPL_PORT=7888
fi

if [ ! -z "$(docker ps -aqf "name=^$GRASPE_CONTAINER_NAME\$")" ]; then
	echo "GRASPE is already started in container ${GRASPE_CONTAINER_NAME}." >&2
	exit 1
fi

TF_FORCE_GPU_ALLOW_GROWTH=true
REBUILD=""
ARGS=""

if [ "$1" == "rebuild" ]; then
	REBUILD=1
fi

if [ "$REBUILD" == "1" ]; then
	echo "Building container..."
	docker build . -t graspe/graspe -f Dockerfile
fi

if [ -f "$HOME/.lein/profiles.clj" ]; then
	ARGS="$ARGS -v $HOME/.lein/profiles.clj:/home/.lein/profiles.clj"
fi

echo "Starting REPL on port $REPL_PORT..."

docker run --gpus all --rm --name $GRASPE_CONTAINER_NAME \
	-p $REPL_PORT:7888 \
	-v $(pwd):/app \
	-v $HOME/.m2:/home/.m2 \
 	$ARGS \
	-u $(id -u):$(id -g) \
	-e TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH \
	-it graspe/graspe
