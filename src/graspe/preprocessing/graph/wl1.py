import numpy as np

from graspe.utils import tolerant
import graspe.preprocessing.utils as enc_utils
import graspe.preprocessing.batcher as batcher
import graspe.preprocessing.encoder as encoder
import graspe.preprocessing.preference.utility as pref_util

@tolerant
def feature_dim(node_feature_dim=0, node_label_count=0):
  return max(node_feature_dim + node_label_count, 1)

def encode_graph(
  g, node_ordering=None,
  node_feature_dim=None, node_label_count=None):
  if node_feature_dim is None:
    node_feature_dim = 0

  if node_label_count is None:
    node_label_count = 0

  if node_ordering is None:
    node_ordering = g.nodes

  n_count = g.order()
  e_count = g.size()
  node_dim = feature_dim(node_feature_dim, node_label_count)
  I_n = np.eye(node_label_count)
  if node_feature_dim > 0 or node_label_count > 0:
    x_init = np.zeros
  else:
    x_init = np.ones
  X = x_init((n_count, node_dim), dtype=np.float32)
  ref_a = np.zeros(e_count, dtype=np.int32)
  ref_b = np.zeros(e_count, dtype=np.int32)

  n_ids = {}
  i = 0
  for node in node_ordering:
    data = g.nodes[node]
    if node_label_count > 0:
      X[i, 0:node_label_count] = I_n[data["label"] - 1]
    if node_feature_dim > 0:
      X[i, node_label_count:node_dim] = data["features"]

    n_ids[node] = i
    i += 1

  i = 0
  for a, b in g.edges():
    ref_a[i] = n_ids[a]
    ref_b[i] = n_ids[b]
    i += 1

  return dict(
    X=X,
    ref_a=ref_a,
    ref_b=ref_b)

def vertex_count(e):
  return len(e["X"])

def total_count(e):
  return len(e["X"]) + len(e["ref_a"])


space_metrics = dict(
  embeddings_count=vertex_count,
  total_count=total_count
)

class WL1Encoder(encoder.ObjectEncoder):
  name = "wl1"

  def __init__(
    self, node_feature_dim=None, node_label_count=None, ordered=False):
    self.node_feature_dim = node_feature_dim
    self.node_label_count = node_label_count
    self.ordered = ordered

  def encode_element(self, graph):
    if self.ordered:
      graph, node_ordering = graph
    else:
      node_ordering = None

    return encode_graph(
      graph,
      node_feature_dim=self.node_feature_dim,
      node_label_count=self.node_label_count,
      node_ordering=node_ordering)

def make_wl1_batch(graphs, masking=False):
  return enc_utils.make_graph_batch(
    graphs,
    ref_keys=["ref_a", "ref_b"],
    masking_fns=dict(
      ref_a_idx=lambda e: e["ref_a"],
      ref_b_idx=lambda e: e["ref_b"]) if masking else None,
    meta_fns=dict(n=vertex_count))

class WL1Batcher(batcher.Batcher):
  name = "wl1"

  def __init__(self, masking=False, space_metric="embeddings_count", **kwargs):
    super().__init__(**kwargs)
    assert space_metric in space_metrics, "Unknown WL1 space metric."
    self.masking = masking
    self.space_metric = space_metric

    suffix = ""
    if self.batch_space_limit is not None:
      suffix += f"_{space_metric}_metric"
    if masking:
      suffix += "_masked"

    self.name += suffix
    self.basename += suffix

  def finalize(self, graphs):
    return make_wl1_batch(graphs, self.masking)

  def compute_space(self, graph, batch):
    return space_metrics[self.space_metric](graph)

class WL1UtilityPreferenceBatcher(pref_util.UtilityPreferenceBatcher):
  name = "wl1_util_pref"

  def __init__(
    self, in_meta, out_meta, **kwargs):
    super().__init__(in_meta, out_meta, **kwargs)
    self.space_metric = in_meta.get("space_metric", "embeddings_count")
    assert self.space_metric in space_metrics, "Unknown WL1 space metric."
    self.masking = in_meta.get("masking", False)

    suffix = ""
    if self.batch_space_limit is not None:
      suffix += f"_{self.space_metric}_metric"
    if self.masking:
      suffix += "_masked"

    self.name += suffix
    self.basename += suffix

  def finalize_objects(self, graphs):
    return make_wl1_batch(graphs, self.masking)

  def compute_object_space(self, graph):
    return space_metrics[self.space_metric](graph)
