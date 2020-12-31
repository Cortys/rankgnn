import os
import json
import numbers
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
    return super().default(obj)

class NumpyDecoder(json.JSONDecoder):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, object_hook=self.object_hook, **kwargs)

  def object_hook(self, obj):
    if isinstance(obj, list):
      if fy.all(lambda o: isinstance(o, numbers.Number), obj):
        return np.array(obj)
      else:
        return [self.object_hook(o) for o in obj]
    elif isinstance(obj, dict):
      for key in obj.keys():
        obj[key] = self.object_hook(obj[key])

    return obj

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

def graphs_stats(graphs):
  node_counts = []
  edge_counts = []
  radii = []
  degrees = []

  for g in graphs:
    node_counts.append(g.order())
    edge_counts.append(g.size())
    degrees += [d for n, d in g.degree()]
    try:
      radii.append(nx.algorithms.distance_measures.radius(g))
    except Exception:
      radii.append(np.inf)

  return dict(
    node_counts=statistics(node_counts),
    edge_counts=statistics(edge_counts),
    node_degrees=statistics(degrees),
    radii=statistics(radii, mask_invalid=True))

def graph_feature_dims(g):
  dim_node_features = 0
  dim_edge_features = 0

  for _, data in g.nodes(data=True):
    f = data.get("features")
    if f is not None:
      dim_node_features = len(f)
    break

  for _, _, data in g.edges(data=True):
    f = data.get("features")
    if f is not None:
      dim_edge_features = len(f)
    break

  return dim_node_features, dim_edge_features

def graphs_meta(graphs):
  assert len(graphs) > 0
  n_labels = set()
  e_labels = set()

  for g in graphs:
    for n, d in g.nodes(data=True):
      if "label" in d:
        n_labels.add(d["label"])

    for u, v, d in g.edges(data=True):
      if "label" in d:
        e_labels.add(d["label"])

  n_nl = (
    max(n_labels) if n_labels != set() else 0)
  if n_nl != 0 and min(n_labels) == 0:
    n_nl += 1

  n_el = (
    max(e_labels) if e_labels != set() else 0)
  if n_el != 0 and min(e_labels) == 0:
    n_el += 1

  d_nf, d_ef = graph_feature_dims(graphs[0])

  return dict(
    node_feature_dim=d_nf,
    edge_feature_dim=d_ef,
    node_label_count=n_nl,
    edge_label_count=n_el
  )

def make_dir(dir):
  if not dir.exists():
    os.makedirs(dir)

  return dir
