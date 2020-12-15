#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

if [ -z "$GRASPE_CONTAINER_NAME" ]; then
	GRASPE_CONTAINER_NAME="graspe"
fi

if [ -z "$JUPYTER_PORT" ]; then
	JUPYTER_PORT=8888
fi

JUPYTER_TOKEN=${JUPYTER_TOKEN:-$(cat JUPYTER_TOKEN)}
JUPYTER_URL="http://localhost:$JUPYTER_PORT/?token=$JUPYTER_TOKEN"

if [ ! -z "$(docker ps -aqf "name=^$GRASPE_CONTAINER_NAME\$")" ]; then
	echo "GRASPE is already started in container ${GRASPE_CONTAINER_NAME}." >&2
	exit 1
fi

TF_FORCE_GPU_ALLOW_GROWTH=true
REBUILD="1"
ARGS="--rm -it"

if [ "$1" == "detached" ]; then
	ARGS="-d"
fi


if [ "$REBUILD" == "1" ]; then
	echo "Building container..."
	docker build . -t graspe/graspe -f Dockerfile
fi

echo "Starting Jupyter at ${JUPYTER_URL} ..."
echo ""

mkdir -p logs
mkdir -p evaluations
mkdir -p data

docker run --gpus all --name $GRASPE_CONTAINER_NAME \
	-p $JUPYTER_PORT:8888 \
	-p 6006:6006 -p 10666:10666 \
	-v $(pwd):/app \
 	$ARGS \
	-u $(id -u):$(id -g) \
	-e "JUPYTER_TOKEN=$JUPYTER_TOKEN" \
	-e TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH \
	-e HOST_PWD=$(pwd) \
	graspe/graspe
