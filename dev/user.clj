(ns user
  (:require [clojure.tools.namespace.repl :as ctnr]
            [clojure.tools.logging :as log]
            [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :refer [py.]]))

(require-python '[tensorflow :as tf]
                '[tensorflow.keras :as keras]
                '[tensorflow.keras.optimizers :as opt]
                '[tensorflow.keras.layers :as layers])

(defn start
  []
  (ctnr/set-refresh-dirs "dev" "src" "test")
  (log/info "REPL started."))

(defn refresh
  []
  (ctnr/refresh :after 'user/start))

(defn refresh-all
  []
  (ctnr/refresh-all :after 'user/start))

(defn xor-test
  []
  (let [model (keras/Sequential [(keras/Input [2] :name "in")
                                 (layers/Dense 3 :activation "sigmoid" :name "hidden1")
                                 (layers/Dense 1 :activation "sigmoid" :name "out")])
        opt (opt/Adam 0.1)
        x (tf/constant [[0 0] [0 1] [1 0] [1 1]])
        y (tf/constant [0 1 1 0])]
    (py. model compile :optimizer opt :loss "binary_crossentropy" :metrics ["accuracy"])
    (py. model fit x y :epochs 100)
    (py. model predict x)))
