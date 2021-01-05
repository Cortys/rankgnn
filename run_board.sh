#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

LOGDIR="/app/${1:-logs}"

if [ -z "$GRASPE_CONTAINER_NAME" ]; then
	GRASPE_CONTAINER_NAME="graspe"
fi

TB_PORT=6006
TB_URL="http://localhost:$TB_PORT"
echo "Starting TensorBoard at ${TB_URL} ..."
docker exec -it $USER $(docker ps -aqf "name=^$GRASPE_CONTAINER_NAME\$") tensorboard --logdir $LOGDIR --bind_all
