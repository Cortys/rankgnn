from collections import Sized
import numpy as np

def statistics(vals, mask_invalid=False):
  if mask_invalid:
    vals_masked = np.ma.masked_invalid(vals)
    return {
      "mean": np.mean(vals_masked),
      "std": np.std(vals_masked),
      "median": np.median(vals_masked),
      "min": np.min(vals),
      "max": np.max(vals),
      "max_masked": np.max(vals_masked),
      "min_masked": np.min(vals_masked),
      "count": len(vals),
      "count_masked": vals_masked.count()
    }
  else:
    return {
      "mean": np.mean(vals),
      "std": np.std(vals),
      "median": np.median(vals),
      "min": np.min(vals),
      "max": np.max(vals),
      "count": len(vals) if isinstance(vals, Sized) else 1
    }


def general_computer(vals, named_splits=None):
  all_stats = statistics(vals)

  if named_splits is None:
    return all_stats

  stats = dict(all=all_stats)

  for split_name, split in named_splits.items():
    stats[split_name] = dict(
      model_selection=[dict(
        train=statistics(vals[m["train"]]),
        val=statistics(vals[m["validation"]])
      ) for m in split["model_selection"]],
      test=statistics(vals[split["test"]]))

  return stats


stat_computer = dict()

def register_stat_computer(type, f=None):
  if f is None:
    return lambda f: register_stat_computer(type, f)

  stat_computer[type] = f

  return f

def find_stat_computer(type):
  if isinstance(type, (tuple, list)):
    return lambda vals, named_splits=None: tuple(
      find_stat_computer(t)(v, named_splits)
      for t, v in zip(type, vals))

  return stat_computer.get(type, general_computer)

def normalize(a, ref=None):
  if ref is None:
    ref = a
  return (a.astype('float64') - np.min(ref)) / np.ptp(ref)
