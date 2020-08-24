(defproject graspe "0.1.0-SNAPSHOT"
  :description "Graph Synthesis WIP"

  :dependencies [[org.clojure/clojure "1.10.1"]
                 [org.clojure/tools.logging "1.1.0"]
                 [clj-python/libpython-clj "1.45"]
                 [aysylu/loom "1.0.2"]]

  :main graspe.core

  :profiles {:dev {:source-paths ["dev" "src" "test"]
                   :dependencies [[org.clojure/tools.namespace "1.0.0"]]
                   :repl-options {:init-ns user
                                  :init (start)}}})
