import json
import itertools
import contextlib
import numpy as np
import funcy as fy
from collections import Sized
import matplotlib.pyplot as plt
import networkx as nx
import inspect

def tolerant(f=None, only_named=True):
  if f is None:
    return lambda f: tolerant(f, only_named)

  if hasattr(f, "__tolerant__"):
    return f

  spec = inspect.getfullargspec(f.__init__ if inspect.isclass(f) else f)
  f_varargs = spec.varargs is not None
  f_varkws = spec.varkw is not None

  if (only_named or f_varargs) and f_varkws:
    return f

  f_args = spec.args
  f_kwonlyargs = spec.kwonlyargs

  @fy.wraps(f)
  def wrapper(*args, **kwargs):
    if not (only_named or f_varargs):
      args = args[:len(f_args)]
    if not f_varkws:
      kwargs = fy.project(kwargs, f_args[len(args):] + f_kwonlyargs)

    return f(*args, **kwargs)

  wrapper.__tolerant__ = True

  return wrapper


fully_tolerant = tolerant(only_named=False)

def select_prefixed_keys(map, prefix, include_others=False, target=dict()):
  for k, v in map.items():
    if k.startswith(prefix):
      target[k[len(prefix):]] = v
    elif include_others:
      target[k] = v

  return target

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if (
      isinstance(obj, np.float32)
      or isinstance(obj, np.float64)
      or isinstance(obj, np.int32)
      or isinstance(obj, np.int64)):
      return np.asscalar(obj)
    return json.JSONEncoder.default(self, obj)

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

@contextlib.contextmanager
def local_seed(seed):
  state = np.random.get_state()
  np.random.seed(seed)
  try:
    yield
  finally:
    np.random.set_state(state)

def cart(*pos_params, **params):
  "Lazily computes the cartesian product of the given lists or dicts."
  if len(pos_params) > 0:
    return itertools.product(*pos_params)

  return (dict(zip(params, x)) for x in itertools.product(*params.values()))

def cart_merge(*dicts):
  "Lazily computes all possible merge combinations of the given dicts."
  return (fy.merge(*c) for c in itertools.product(*dicts))

def entry_duplicator(duplicates):
  def f(d):
    for source, targets in duplicates.items():
      d_source = d[source]
      for target in targets:
        d[target] = d_source

    return d

  return f

def unzip(tuples):
  return list(zip(*tuples))

def vec_to_unit(feat):
  u = 0
  for i, s in enumerate(np.clip(feat, 0, 1), 1):
    u += (2 ** -i) * s

  return u

def draw_graph(
  g, y=None, with_features=False, with_colors=True, label_colors=True):
  plt.figure()

  if y is not None:
    plt.title('Label: {}'.format(y))

  cmap = plt.get_cmap("hsv")
  node_color = [
    vec_to_unit([d.get("label", 0)] if label_colors else d.get("features", []))
    for n, d in g.nodes(data=True)] if with_colors else "#1f78b4"

  if with_features:
    labels = {
      n: f"{n}:" + str(data.get("features"))
      for n, data in g.nodes(data=True)
    }
    nx.draw_spring(
      g, labels=labels,
      node_color=node_color, vmin=0, vmax=1, cmap=cmap)
  else:
    nx.draw_spring(
      g, with_labels=True,
      node_color=node_color, vmin=0, vmax=1, cmap=cmap)

  plt.show()
