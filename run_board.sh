#!/usr/bin/env bash
cd "${BASH_SOURCE%/*}" || exit

USER="-u $(id -u):$(id -g)"

LOGDIR="/app/${1:-logs}"

if [ -z "$GRASPE_CONTAINER_NAME" ]; then
	GRASPE_CONTAINER_NAME="graspe"
fi

# trap 'kill %1; exit 0' SIGINT
# trap 'kill -TERM %1; exit 0' SIGTERM

TB_PORT=6006
TB_URL="http://localhost:$TB_PORT"
echo "Starting TensorBoard at ${TB_URL} ..."
docker exec -it $USER $(docker ps -aqf "name=^$GRASPE_CONTAINER_NAME\$") tensorboard --logdir $LOGDIR --bind_all #&

# while true; do
# 	read in
# 	if [ "$in" == "rm" ]; then
# 		rm -r ./logs/[^.]* 2> /dev/null && echo "Removed logs." || echo "No logs to remove."
# 	fi
# done
