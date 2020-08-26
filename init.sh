#!/usr/bin/env bash

cd /app

if [ "$MODE" == "jack" ]; then
	lein update-in :dependencies conj '[nrepl"0.6.0"]' -- update-in :dependencies conj '[cider/piggieback"0.4.1"]' -- update-in :plugins conj '[cider/cider-nrepl"0.22.4"]' -- update-in '[:repl-options :nrepl-middleware]' conj '["cider.nrepl/cider-middleware"]' -- with-profile +dev repl :headless :host 0.0.0.0 :port 7888
elif [ "$MODE" == "socket" ]; then
	JVM_OPTS='-Dclojure.server.myrepl={:port,5555,:address,"0.0.0.0",:accept,clojure.core.server/repl}' lein repl
else
	lein repl :start :host 0.0.0.0 :port 7888
fi
