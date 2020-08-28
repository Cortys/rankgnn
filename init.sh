#!/usr/bin/env bash

source /etc/bash.bashrc

cd /app

if [ ! -z "$HOST_PWD" ]; then
	mkdir -p $(dirname $HOST_PWD)
	ln -s /app $HOST_PWD
fi

trap 'kill %1; exit 0' SIGINT
trap 'kill -TERM %1; exit 0' SIGTERM

jupyter lab --notebook-dir=/app/py_src --ip 0.0.0.0 --no-browser --allow-root &

if [ "$MODE" == "jack" ]; then
	lein update-in :dependencies conj '[nrepl"0.6.0"]' -- update-in :dependencies conj '[cider/piggieback"0.4.1"]' -- update-in :plugins conj '[cider/cider-nrepl"0.22.4"]' -- update-in '[:repl-options :nrepl-middleware]' conj '["cider.nrepl/cider-middleware"]' -- with-profile +dev repl :headless :host 0.0.0.0 :port 7888
elif [ "$MODE" == "socket" ]; then
	JVM_OPTS='-Dclojure.server.myrepl={:port,5555,:address,"0.0.0.0",:accept,clojure.core.server/repl}' lein repl :headless
elif [ "$MODE" == "nrepl" ]; then
	lein repl :headless :host 0.0.0.0 :port 7888
else
	wait $!
fi
