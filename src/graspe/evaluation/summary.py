from __future__ import absolute_import, division, print_function,\
  unicode_literals

import os
import json
from collections import defaultdict
import funcy as fy
import numpy as np

from graspe.utils import cache_write
from graspe.stats.utils import statistics

selection_metrics = {
  "binary_accuracy": "max",
  "categorical_accuracy": "max",
  "accuracy": "max",
  "r2": "max",
  "tau": "max",
  "binary_crossentropy": "min",
  "categorical_crossentropy": "min",
  "loss": "min",
  "mse": "min",
  "mae": "min",
  "mean_squared_error": "min",
  "mean_absolute_error": "min",
}

def dict_map(f, d):
  return {k: f(v) for k, v in d.items()}

def summarize_evaluation(
  eval_dir, selection_metric="tau", ignore_worst=0):
  if not eval_dir.exists():
    print(f"No evalutation '{eval_dir}' found.")
    return

  with open(eval_dir / "config.json") as f:
    config = json.load(f)

  with open(eval_dir / "hyperparams.json") as f:
    hps = json.load(f)

  results_dir = eval_dir / "results"
  assert results_dir.exists(), f"No results found for '{eval_dir}'."
  summary_dir = eval_dir / "summary"

  if not summary_dir.exists():
    os.makedirs(summary_dir)

  result_files = [
    (list(fy.map(int, f[:-5].split("-"))), results_dir / f)
    for f in os.listdir(results_dir)]

  fold_files = fy.group_by(lambda f: f[0][0], result_files)
  fold_param_files = {
    fold: fy.group_by(lambda f: f[0][1], files)
    for fold, files in fold_files.items()}
  folds = list(fold_param_files.items())
  folds.sort(key=fy.first)

  best_goal = selection_metrics[selection_metric]

  results = []
  all_hps = True

  for fold_i, param_files in folds:
    best_res = None
    param_file_items = list(param_files.items())

    all_hps = all_hps and len(param_files) == len(hps)

    for hp_i, files in param_file_items:
      hp_train_results = defaultdict(list)
      hp_val_results = defaultdict(list)
      hp_test_results = defaultdict(list)
      selection_vals = []
      all_selection_vals = []
      for (_, _, i), file in files:
        with open(file, "r") as f:
          result = json.load(f)

        for metric, val in result["train"].items():
          hp_train_results[metric].append(val)
        for metric, val in result["val"].items():
          hp_val_results[metric].append(val)
        for metric, val in result["test"].items():
          hp_test_results[metric].append(val)

        selection_val = result["val"][selection_metric]
        all_selection_vals.append(selection_val)
        if i < config["repeat"]:
          selection_vals.append(selection_val)

      top_idxs = np.argsort(np.array(all_selection_vals))

      if len(all_selection_vals) > ignore_worst:
        if best_goal == "max":
          top_idxs = top_idxs[ignore_worst:]
        elif best_goal == "min":
          top_idxs = top_idxs[:-ignore_worst]

      top_statistics = fy.compose(
        statistics,
        lambda l: np.array(l)[top_idxs])

      hp_res = dict(
        fold_idx=fold_i,
        train=dict_map(top_statistics, hp_train_results),
        val=dict_map(top_statistics, hp_val_results),
        test=dict_map(top_statistics, hp_test_results),
        select=np.mean(selection_vals),
        hp_i=hp_i,
        hp=hps[hp_i],
        select_repeats=len(selection_vals),
        eval_repeats=len(files))

      if (
        best_res is None
        or (best_goal == "max" and best_res["select"] < hp_res["select"])
        or (best_goal == "min" and best_res["select"] > hp_res["select"])
        or (
          best_res["select"] == hp_res["select"]
          and best_res["eval_repeats"] < hp_res["eval_repeats"])):
        best_res = hp_res

    if best_res is not None:
      results.append(best_res)
    else:
      print(f"No results for {fold_i}.")

  if len(results) == 1:
    combined_train = results[0]["train"]
    combined_test = results[0]["test"]
  else:
    combined_train = dict_map(
      statistics,
      fy.merge_with(np.array, *map(
        lambda res: dict_map(lambda t: t["mean"], res["train"]), results)))
    combined_test = dict_map(
      statistics,
      fy.merge_with(np.array, *map(
        lambda res: dict_map(lambda t: t["mean"], res["test"]), results)))

  results_summary = {
    "folds": results,
    "combined_train": combined_train,
    "combined_test": combined_test,
    "args": {
      "ignore_worst": ignore_worst
    },
    "done": all_hps and len(folds) == config["outer_k"]
  }

  cache_write(summary_dir / "results.json", results_summary, "pretty_json")

  return results_summary
