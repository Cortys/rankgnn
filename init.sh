#!/usr/bin/env bash

source /etc/bash.bashrc

cd /app

if [ ! -z "$HOST_PWD" ]; then
	mkdir -p $(dirname $HOST_PWD)
	ln -s /app $HOST_PWD
fi

mkdir -p /tmp/matplotlib_cache
export MPLCONFIGDIR=/tmp/matplotlib_cache

jupyter lab --notebook-dir=/app/src --ip 0.0.0.0 --no-browser --allow-root
