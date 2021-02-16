from __future__ import absolute_import, division, print_function,\
  unicode_literals

import os
import gc
import json
from timeit import default_timer as timer
from pathlib import Path
from datetime import datetime
import numpy as np
from tensorflow import keras
import funcy as fy
import tensorflow as tf

from graspe.stats.utils import statistics
from graspe.utils import tolerant, NumpyEncoder
import graspe.models.sort as model_sort
import graspe.evaluation.summary as summary

eval_dir_base = Path("../evaluations")
log_dir_base = Path("../logs")

def time_str():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class EpochTimeCallback(keras.callbacks.Callback):
  def __init__(self):
    self.times = []
    self.currTime = timer()

  def on_epoch_end(self, epoch, logs={}):
    currTime = timer()
    self.times.append(currTime - self.currTime)
    self.currTime = currTime

  def on_train_end(self, logs={}):
    pass

def train(
  model, train_ds, val_ds=None, label=None,
  log_dir_base=log_dir_base,
  epochs=1000, patience=100, restore_best=False,
  stopping_min_delta=0.0001,
  verbose=2, measure_epoch_times=False):
  label = "_" + label if label is not None else ""

  t = time_str()
  tb = keras.callbacks.TensorBoard(
    log_dir=log_dir_base / f"{t}{label}/",
    histogram_freq=10,
    write_images=False)
  es = keras.callbacks.EarlyStopping(
    monitor="loss" if val_ds is None else "val_loss",
    patience=patience,
    min_delta=stopping_min_delta,
    restore_best_weights=restore_best)

  if measure_epoch_times:
    tc = EpochTimeCallback()
    callbacks = [tb, es, tc]
  else:
    callbacks = [tb, es]

  hist = model.fit(
    train_ds, validation_data=val_ds,
    epochs=epochs, callbacks=callbacks,
    verbose=verbose)

  if measure_epoch_times:
    return hist.history, tc.times
  else:
    return hist.history

@tolerant
def evaluation_step(
  model_ctr, train_ds, val_ds, test_ds, k, hp_i, i, hp,
  res_dir, fold_str, hp_str, verbose, log_dir_base,
  epochs, patience, stopping_min_delta, restore_best,
  repeat, winner_repeat, pos_hp_i=None, custom_evaluator=None):
  if i >= repeat:
    repeat = winner_repeat

  rep_str = f"{i+1}/{repeat}"
  label = f"{k}-{hp_i}-{i}"
  res_file = res_dir / f"{label}.json"

  if res_file.exists():
    print(
      time_str(),
      f"- Iteration {rep_str}, fold {fold_str}, hps {hp_str} already done.")
    return False

  print(
    time_str(),
    f"- Iteration {rep_str}, fold {fold_str}, hps {hp_str}...")

  t_start = timer()
  model = model_ctr(**hp)
  history = train(
    model, train_ds, val_ds, label,
    epochs=epochs, patience=patience,
    stopping_min_delta=stopping_min_delta,
    restore_best=restore_best,
    log_dir_base=log_dir_base,
    verbose=verbose)
  t_end = timer()
  train_dur = t_end - t_start
  t_start = t_end
  train_res = model.evaluate(train_ds)
  val_res = model.evaluate(val_ds)
  test_res = model.evaluate(test_ds)
  t_end = timer()
  eval_dur = t_end - t_start
  train_res = dict(zip(model.metrics_names, train_res))
  val_res = dict(zip(model.metrics_names, val_res))
  test_res = dict(zip(model.metrics_names, test_res))

  if custom_evaluator is not None:
    train_cust, val_cust, test_cust = custom_evaluator(model)
    train_res = fy.merge(train_res, train_cust)
    val_res = fy.merge(val_res, val_cust)
    test_res = fy.merge(test_res, test_cust)

  with open(res_file, "w") as f:
    json.dump({
      "history": history,
      "train": train_res,
      "val": val_res,
      "test": test_res,
      "train_duration": train_dur,
      "eval_duration": eval_dur
    }, f, cls=NumpyEncoder)

  tf.keras.backend.clear_session()
  gc.collect()

  print(
    f"\nTest results in {train_dur}s/{eval_dur}s for",
    f"it {rep_str}, fold {fold_str}, hps {hp_str}:",
    test_res)
  return True

def find_eval_dir(model_factory, ds_provider, label=None, split=None):
  label = "_" + label if label is not None else ""
  split = "_" + split if isinstance(split, str) else ""
  mf_name = model_factory.name
  ds_name = ds_provider.full_name
  return eval_dir_base / f"{ds_name}{split}_{mf_name}{label}"

def sort_evaluator(ds_provider, enc, outer_idx, **config):
  train_idxs, val_idxs, test_idxs = ds_provider.get_split_indices(
    outer_idx=outer_idx, relative=True)
  train_get = fy.partial(ds_provider.get_train_split, outer_idx=outer_idx)
  val_get = fy.partial(ds_provider.get_validation_split, outer_idx=outer_idx)
  test_get = fy.partial(ds_provider.get_test_split, outer_idx=outer_idx)
  train_eval = model_sort.evaluate_model_sort(
    train_idxs, train_get, enc=enc, **config)
  val_eval = model_sort.evaluate_model_sort(
    val_idxs, val_get, enc=enc, **config)
  test_eval = model_sort.evaluate_model_sort(
    test_idxs, test_get, enc=enc, **config)

  def evaluator(model):
    return train_eval(model), val_eval(model), test_eval(model)

  return evaluator

def evaluate(
  model_factory, ds_provider,
  split=None, repeat=1, winner_repeat=3, epochs=2000,
  patience=100, stopping_min_delta=0.0001,
  restore_best=False, hp_args=None, label=None,
  selection_metric="accuracy",
  eval_dir=None, verbose=2, dry=False, ignore_worst=0, single_hp=None):
  if ds_provider.default_split != 0 and split is None:
    split = ds_provider.default_split

  outer_k = ds_provider.outer_k if split is None else 1
  inner_k = None

  model = model_factory.get_model()

  mf_name = model_factory.name
  ds_config = model_factory.config
  ds_name = ds_provider.full_name

  prefer_in_enc = model_factory.prefer_in_enc
  prefer_out_enc = model_factory.prefer_out_enc
  encs = list(ds_provider.find_compatible_encodings(model))
  if len(encs) > 1:
    filtered_encs = list(fy.filter(
      lambda enc: (
        (prefer_in_enc is None or enc[0] == prefer_in_enc)
        and (prefer_out_enc is None or enc[1] == prefer_out_enc)
      ), encs))
    if len(filtered_encs) > 0:
      encs = filtered_encs

  enc = encs[0]
  model_ctr = fy.func_partial(
    model, enc=enc, in_meta=ds_provider.in_meta, out_meta=ds_provider.out_meta)

  t = time_str()
  if eval_dir is None:
    eval_dir = find_eval_dir(model_factory, ds_provider, label, split)
    if not eval_dir.exists():
      os.makedirs(eval_dir)

    with open(eval_dir / "config.json", "w") as f:
      config = {
        "outer_k": outer_k,
        "inner_k": inner_k,
        "repeat": repeat,
        "winner_repeat": winner_repeat,
        "epochs": epochs,
        "patience": patience,
        "stopping_min_delta": stopping_min_delta,
        "restore_best": restore_best,
        "hp_args": hp_args,
        "enc": enc,
        "ds_name": ds_name,
        "mf_name": mf_name,
        "split": split,
        "start_time": t,
        "end_time": None,
        "duration": 0
      }
      json.dump(config, f, indent="\t", sort_keys=True, cls=NumpyEncoder)
    resume = False
  else:
    assert eval_dir.exists(), "Invalid resume directory."
    print()
    with open(eval_dir / "config.json", "r") as f:
      config = json.load(f)
      assert (
        config["outer_k"] == outer_k
        and config["inner_k"] == inner_k
        and config["repeat"] == repeat
        and config["winner_repeat"] == winner_repeat
        and config["epochs"] == epochs
        and config["patience"] == patience
        and config["stopping_min_delta"] == stopping_min_delta
        and config["restore_best"] == restore_best
        and config["hp_args"] == hp_args
        and tuple(config["enc"]) == tuple(enc)
        and config["ds_name"] == ds_name
        and config["mf_name"] == mf_name
        and config["split"] == split), "Incompatible config."
    resume = True

  log_dir_base = eval_dir / "logs"
  res_dir = eval_dir / "results"
  pos_file = eval_dir / "state.txt"

  if not res_dir.exists():
    os.makedirs(res_dir)

  if not log_dir_base.exists():
    os.makedirs(log_dir_base)

  k_start = 0
  hp_start = 0
  i_start = -1
  print(t, f"- Evaluating {ds_name} using {mf_name}...")

  if single_hp is not None:
    print(f"!!! USING SINGLE HP MODE FOR HYPERPARAM {single_hp} !!!")
    print("The resulting mean accuracy might be negatively affected.")

  if resume and pos_file.exists():
    pos = pos_file.read_text().split(",")
    if len(pos) == 3:
      k_start, hp_start, i_start = fy.map(int, pos)
      print(f"Continuing at {k_start}, {hp_start}, {i_start}.")

  hp_args = hp_args or dict()
  hps = model_factory.get_hyperparams(
    enc, in_meta=ds_provider.in_meta, out_meta=ds_provider.out_meta, **hp_args)
  hpc = len(hps)

  hp_file = eval_dir / "hyperparams.json"
  if hp_file.exists():
    hps_json = json.dumps(hps, indent="\t", sort_keys=True, cls=NumpyEncoder)
    old_hps_json = hp_file.read_text()

    assert hps_json == old_hps_json, \
        f"Existing hyperparam list incompatible to:\n{hps_json}"

    print("Existing hyperparam list is compatible.")
  else:
    with open(hp_file, "w") as f:
      json.dump(hps, f, indent="\t", sort_keys=True, cls=NumpyEncoder)

  if dry:
    print(f"Completed dry evaluation of {ds_name} using {mf_name}.")
    return

  t_start_eval = timer()
  completed_evaluation_step = False
  try:
    for k in range(k_start, outer_k):
      print()
      fold_str = f"{k+1}/{outer_k}"
      print(time_str(), f"- Evaluating fold {fold_str}...")
      t_start_fold = timer()
      if split is None:
        outer_idx = k
      elif split is True:
        outer_idx = None
      else:
        outer_idx = split
      train_ds, val_ds, test_ds = ds_provider.get_split(
        enc, outer_idx=outer_idx, config=ds_config)
      custom_evaluator = sort_evaluator(
        ds_provider, enc, outer_idx, **ds_config)

      if train_ds is None or test_ds is None:
        print(time_str(), f"- Data of fold {fold_str} could not be loaded.")
        continue

      for hp_i, hp in enumerate(hps):
        hp_str = f"{hp_i+1}/{hpc}"
        curr_i_start = 0

        if single_hp is not None and single_hp != hp_i:
          print(f"Skipping {fold_str} with hp {hp_str} due to single hp mode.")
          continue

        if k == k_start:
          if hp_i < hp_start:
            print(f"Already evaluated {fold_str} with hyperparams {hp_str}.")
            continue
          elif hp_i == hp_start and i_start >= 0:
            print(
              f"Already evaluated {fold_str} with hyperparams {hp_str}",
              f"{i_start + 1}/{repeat} times.")
            curr_i_start = i_start + 1

        print(f"\nFold {fold_str} with hyperparams {hp_str}.")

        for i in range(curr_i_start, repeat):
          completed_evaluation_step |= evaluation_step(
            model_ctr, train_ds, val_ds, test_ds, k, hp_i, i, hp,
            res_dir, fold_str, hp_str, verbose, log_dir_base,
            custom_evaluator=custom_evaluator,
            **config)
          if single_hp is None:
            pos_file.write_text(f"{k},{hp_i},{i}")

      t_end_fold = timer()
      dur_fold = t_end_fold - t_start_fold
      summ = summary.summarize_evaluation(
        eval_dir, selection_metric=selection_metric, ignore_worst=ignore_worst)
      print(time_str(), f"- Evaluated hps of fold {fold_str} in {dur_fold}s.")

      if winner_repeat > repeat:
        if single_hp is not None:
          print(f"No repeats because of single hp mode for hp={single_hp}.")
        elif hp_start == hpc and i_start + 1 == winner_repeat:
          print(f"Already did winner evaluations of fold {fold_str}.")
        else:
          best_hp_i = summ["folds"][k]["hp_i"]
          best_hp = hps[best_hp_i]
          hp_str = f"{best_hp_i+1}/{hpc}"
          add_rep = winner_repeat - repeat
          print(
            time_str(),
            f"- Additional {add_rep} evals of fold {fold_str}",
            f"and winning hp {hp_str}.")

          for i in range(repeat, winner_repeat):
            completed_evaluation_step = evaluation_step(
              model_ctr, train_ds, val_ds, test_ds, k, best_hp_i, i, best_hp,
              res_dir, fold_str, hp_str, verbose, log_dir_base,
              custom_evaluator=custom_evaluator,
              **config)
            if single_hp is None:
              pos_file.write_text(f"{k},{hpc},{i}")

          summary.summarize_evaluation(
            eval_dir,  selection_metric=selection_metric,
            ignore_worst=ignore_worst)
          print(
            time_str(),
            f"- Completed additional {add_rep} evals of fold {fold_str}",
            f"and winning hp {hp_str}.")
  finally:
    t_end_eval = timer()
    dur_eval = t_end_eval - t_start_eval

    if completed_evaluation_step:
      with open(eval_dir / "config.json", "w") as f:
        config["duration"] += dur_eval
        config["end_time"] = time_str()
        json.dump(config, f, indent="\t", sort_keys=True, cls=NumpyEncoder)

  summary.summarize_evaluation(
    eval_dir, selection_metric=selection_metric, ignore_worst=ignore_worst)
  print(
    time_str(),
    f"- Evaluation of {ds_name} using {mf_name} completed in {dur_eval}s.",
    "No steps were executed." if not completed_evaluation_step else "")

def resume_evaluation(model_factory, ds_provider, eval_dir=None, **kwargs):
  if eval_dir is None:
    eval_dir = find_eval_dir(model_factory, ds_provider)

  if not (eval_dir / "config.json").exists():
    print(f"Starting new evaluation at {eval_dir}...")
    return evaluate(model_factory, ds_provider, **kwargs)

  print(f"Resuming evaluation at {eval_dir}...")

  with open(eval_dir / "config.json", "r") as f:
    config = json.load(f)

  return evaluate(
    model_factory, ds_provider,
    split=config["split"],
    repeat=config["repeat"],
    winner_repeat=config["winner_repeat"],
    epochs=config["epochs"],
    patience=config["patience"],
    stopping_min_delta=config["stopping_min_delta"],
    restore_best=config["restore_best"],
    hp_args=config["hp_args"],
    eval_dir=eval_dir,
    **fy.omit(kwargs, config.keys()))

def quick_evaluate(model_factory, ds_provider, **kwargs):
  return evaluate(
    model_factory, ds_provider,
    epochs=1, repeat=1, winner_repeat=1, label="quick",
    **fy.omit(kwargs, ["epochs", "repeat", "winner_repeat", "label"]))

def evaluate_epoch_time(model_factory, ds_provider, hp_args=None):
  eval_dir = find_eval_dir(model_factory, ds_provider, "time_eval")
  if not eval_dir.exists():
    os.makedirs(eval_dir)

  log_dir_base = eval_dir / "logs"
  times_file = eval_dir / "times.json"

  if not log_dir_base.exists():
    os.makedirs(log_dir_base)

  if times_file.exists():
    print(f"Already evaluated times of {model_factory.name} on the dataset.")
    return

  hp_args = hp_args or dict()
  model_ctr, hps = model_factory(ds_provider, **hp_args)
  model = model_ctr(**list(hps)[0])
  train_ds = ds_provider.get_all(output_type=model_factory.input_type)

  wc = int(np.sum([
    keras.backend.count_params(w)
    for w in model.trainable_weights]))
  print(f"\nModel {model_factory.name} trainable weights = {wc}\n")

  _, times = train(
    model, train_ds,
    log_dir_base=log_dir_base,
    epochs=110, patience=110,
    measure_epoch_times=True)

  with open(times_file, "w") as f:
    json.dump(
      {
        "summary": statistics(times[10:]),
        "weight_count": wc,
        "epoch_times": times
      },
      f, cls=NumpyEncoder, indent="\t")
