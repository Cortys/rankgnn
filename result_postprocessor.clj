#!/usr/bin/env bb

(require '[clojure.string :as str]
         '[cheshire.core :as json]
         '[clojure.java.shell :refer [sh]])

(def include-lta true)

(def default-radii (constantly 1))
(def single-depth-radii default-radii)
(def default-iterations {"WL_st_SVM" 5
                         "WL_sp_SVM" 5})
(def ds-rename {"triangle_count_dataset" "TRIANGLES"
                "ogbg-molesol_ogb" "molesol"
                "ogbg-mollipo_ogb" "mollipo"
                "ogbg-molfreesolv_ogb" "molfreesolv"
                "ZINC_full_tu" "ZINC"})
(def ds-colnames {"triangle_count_dataset" "triangle"
                  "ogbg-molesol_ogb" "molesol"
                  "ogbg-mollipo_ogb" "mollipo"
                  "ogbg-molfreesolv_ogb" "molfreesolv"
                  "ZINC_full_tu" "zinc"})
(def datasets ["triangle_count_dataset"
               "ogbg-molesol_ogb"
               "ogbg-mollipo_ogb"
               "ogbg-molfreesolv_ogb"
               "ZINC_full_tu"])
(def wlst-model "WL\\textsubscript{ST}")
(def mean-pool "\\mean")
(def sam-pool (if include-lta "\\mathrm{SAM}" "\\wmean"))
(def models-with-potential-oom #{wlst-model})
(def use-incomplete #{["ZINC_full_tu", "DirectRankWL2GNN"]})

(defn round
  ([num] (round num 0))
  ([num prec]
   (if (int? num)
     num
     (if (zero? prec)
       (int (Math/round num))
       (let [p (Math/pow 10 prec)
             rounded (/ (Math/round (* num p)) p)]
         (if (pos? prec)
           rounded
           (int rounded)))))))

(defn dim-str
  [feat lab]
  (if (zero? (+ feat lab))
    "0 + 1"
    (str feat " + " lab)))

(defn stats-dict->csv-line
  [[name {:strs [node_counts
                 edge_counts
                 dim_node_features
                 dim_edge_features
                 num_node_labels
                 num_edge_labels
                 node_degrees
                 radii]}]]
  (str/join ","
            [(ds-rename name name)
             (get node_counts "count") ; graph count
             (round (get node_counts "min"))
             (round (get node_counts "mean") 1)
             (round (get node_counts "max"))
             (round (get edge_counts "min"))
             (round (get edge_counts "mean") 1)
             (round (get edge_counts "max"))
             (dim-str dim_node_features num_node_labels)
             (dim-str dim_edge_features num_edge_labels)
             (round (get node_degrees "min"))
             (round (get node_degrees "mean") 1)
             (round (get node_degrees "std") 1)
             (round (get node_degrees "max"))
             (round (get radii "mean") 1)
             (round (get radii "std") 1)]))

(defn ds-stats->csv
  []
  (let [stats (json/parse-string (slurp "data/stats.json"))
        head (str "name,"
                  "graph_count,"
                  "node_count_min,node_count_mean,node_count_max,"
                  "edge_count_min,edge_count_mean,edge_count_max,"
                  "dim_node_features,dim_edge_features,",
                  "node_degree_min,node_degree_mean,node_degree_std,node_degree_max,"
                  "radius_mean,radius_std")
        stats (str head "\n" (str/join "\n" (map stats-dict->csv-line stats)) "\n")]
    (spit "./results/ds_stats.csv" stats)
    (println stats)))

(defn extract-pool
  [name]
  (let [pool (if (str/includes? name "Avg") mean-pool sam-pool)]
    pool
    #_(if (str/ends-with? name "FC")
        (str pool " + \\mathrm{MLP}")
        pool)))

(defn eval-name->params
  [dataset name]
  (let [[_ r n _ T time-eval] (re-find #"n?(\d?)_?(.+?)(_FC)?(_\d)?(_time_eval)?$" name)
        ;_ (println "y" name dataset r n T)
        r (if (seq r) (Integer/parseInt r) (default-radii dataset))
        T (when (seq T) (Integer/parseInt (subs T 1)))
        ; pool (extract-pool name)
        time-eval (some? time-eval)]
    (condp #(str/includes? %2 %1) n
      ; pair:
      "DirectRankGCN"
      {:name "DirectRankGCN" :order [1 0 0]
       :pool ""
       :time-eval time-eval
       :is-default true}
      "DirectRankGIN"
      {:name "DirectRankGIN" :order [1 1 0]
       :pool ""
       :time-eval time-eval
       :is-default true}
      "DirectRankWL2GNN"
      {:name "2-WL-DirectRankGNN" :order [1 2 r]
       :pool ""
       :time-eval time-eval
       :is-default true}
      "CmpGCN"
      {:name "CmpGCN" :order [1 3 0]
       :pool ""
       :time-eval time-eval
       :is-default true}
      "CmpGIN"
      {:name "CmpGIN" :order [1 4 0]
       :pool ""
       :time-eval time-eval
       :is-default true}
      "CmpWL2GNN"
      {:name "2-WL-CmpGNN" :order [1 5 r]
       :pool ""
       :time-eval time-eval
       :is-default true}
      ; point:
      ; "WL_st"
      ; {:name wlst-model
      ;  :order [0 0 (or T 5)]
      ;  :it (str "T=" (or T 5))
      ;  :T (or T 5)
      ;  :is-default true ; (or (= T 1) (= T 3))
      ;  :hide-diff (= T 1)}
      "GCN"
      {:name "GCN" :order [0 1 0]
       :pool ""
       :time-eval time-eval
       :is-default true}
      "GIN"
      {:name "GIN" :order [0 2 0]
       :pool ""
       :time-eval time-eval
       :is-default true}
      "WL2GNN"
      {:name "2-WL-GNN"
       :order [0 3 r]
       :r r
       :is-default (= r (single-depth-radii dataset))
       :time-eval time-eval
       :pool ""}
      nil)))

(defn to-proc
  [x]
  (or x 0))
  ;(* 100 (or x 0)))

(defn ls-dir
  [dir]
  (let [{out :out} (sh "ls" dir)]
    (str/split out #"\n+")))

(defn dataset-results
  [dataset & {:keys [only-default] :or {only-default true}}]
  (let [evals (ls-dir "./evaluations/")
        summaries (into []
                        (comp (filter #(and (str/starts-with? % dataset) (not (str/ends-with? % "quick"))))
                              (map (juxt identity #(try
                                                     [(slurp (str "./evaluations/" % "/summary/results.json"))
                                                      (slurp (str "./evaluations/" % "/config.json"))]
                                                     (catch Exception _))))
                              (keep (fn [[name [s c]]]
                                      (when (and c s)
                                        (assoc (json/parse-string s true)
                                               :name (subs name (inc (count dataset)))
                                               :config (json/parse-string c true)))))
                              (keep (fn [{name :name
                                          config :config
                                          folds :folds
                                          {test :tau} :combined_test
                                          {train :tau} :combined_train
                                          done :done}]
                                      (when (or done (use-incomplete [dataset name]))
                                        {:name name
                                         :config config
                                         :test-mean (to-proc (:mean test))
                                         :test-std (to-proc (:std test))
                                         :train-mean (to-proc (:mean train))
                                         :train-std (to-proc (:std train))
                                         :folds (map (fn [{{test :tau} :test}]
                                                       {:test-mean (to-proc (:mean test))})
                                                     folds)}))))
                        evals)
        results (keep (fn [{name :name :as sum}]
                        (println dataset name (eval-name->params dataset name))
                        (when-let [params (eval-name->params dataset name)]
                          (when (or (not only-default) (:is-default params))
                            (merge sum {:model (params :name)
                                        :order (params :order)
                                        :dataset dataset
                                        :is-default (:is-default params)
                                        :hide-diff (:hide-diff params)
                                        :pool (or (:pool params) "")
                                        :it (or (:it params) "")
                                        :T (:T params)
                                        :r (:r params)
                                        :params "" #_(str/join ", " (keep params [:pool :it]))}))))
                      summaries)
        typed-max (fn ([] {}) ([x] x) ([max-dict [t v]] (update max-dict t (fnil max 0) v)))
        max-grouper (juxt (comp first :order) :is-default)
        max-test (transduce (map (juxt max-grouper :test-mean)) typed-max {} results)
        max-train (transduce (map (juxt max-grouper :train-mean)) typed-max {} results)
        results (map (fn [res] (assoc res
                                      :is-best-test (= (max-test (max-grouper res)) (:test-mean res))
                                      :is-best-train (= (max-train (max-grouper res)) (:train-mean res))))
                     results)]
      results))

(defn dataset-result-head
  [dataset & {with-best :with-best :or {with-best true}}]
  (let [pre (ds-colnames dataset)]
    (if with-best
      (str pre "TestMean;" pre "TestStd;" pre "TrainMean;" pre "TrainStd;" pre "BestTest;" pre "BestTrain")
      (str pre "TestMean;" pre "TestStd;" pre "TrainMean;" pre "TrainStd"))))

(defn dataset-result-row
  [datasets results idx [model params is-default]]
  (let [results (into {}
                      (comp (filter #(and (= (:model %) model)
                                          (= (:params %) params)
                                          (= (:is-default %) is-default)))
                            (map (juxt :dataset identity)))
                      results)]
    (str idx ";" model ";" (when (seq params) (str params)) ";"
         (if is-default "1" "0")  ";"
         (str/join ";" (map (fn [ds]
                              (if-let [res (results ds)]
                                (str (str/join ";" (->> [:test-mean :test-std :train-mean :train-std]
                                                        (map res)
                                                        (map #(format "%.3f" (double %)))))
                                     ";" (if (:is-best-test res) "1" "0")
                                     ";" (if (:is-best-train res) "1" "0"))
                                (if (models-with-potential-oom model) "m;m;m;m;0;0" "t;t;t;t;0;0")))
                            datasets)))))

(defn eval-results->csv
  [{:keys [only-default]} file & args]
  (let [datasets (if (empty? args) datasets args)
        results (sort-by :order (mapcat #(dataset-results % :only-default only-default) datasets))
        models-with-params (distinct (map (juxt :model :params :is-default) results))
        _ (println (map :model results))
        head (str "id;model;params;isDefault;" (str/join ";" (map dataset-result-head datasets)))
        rows (into [] (map-indexed (partial dataset-result-row datasets results)) models-with-params)
        csv (str head "\n" (str/join "\n" rows) "\n")]
    (spit (str "./results/" file ".csv") csv)
    (println csv)))

(defn mean
  [vals]
  (/ (apply + vals) (count vals)))

(defn std
  [vals]
  (let [m (mean vals)]
    [m (Math/sqrt
         (/ (apply + (map (comp #(* % %) #(- % m))
                          vals))
            (count vals)))]))

(defn default-action
  []
  (ds-stats->csv)
  (eval-results->csv {:only-default false} "results"))

(def actions {"ds_stats" ds-stats->csv
              "eval_res" (partial eval-results->csv {:only-default false} "results")
              nil default-action})

(println "GRASPE Results Postprocessor.")
(if-let [action (actions (first *command-line-args*))]
  (apply action (rest *command-line-args*))
  (println "Unknown action:" (first *command-line-args*)))
(println "Done.")
