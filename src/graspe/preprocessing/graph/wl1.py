import numpy as np

from graspe.utils import tolerant
import graspe.preprocessing.utils as enc_utils
import graspe.preprocessing.batcher as batcher
import graspe.preprocessing.encoder as encoder

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

  node_dim = node_feature_dim + node_label_count

  n_zero_f = np.zeros(node_feature_dim)
  I_n = np.eye(node_label_count)

  X = []
  ref_a = []
  ref_b = []

  n_ids = {}
  i = 0

  for node in node_ordering:
    data = g.nodes[node]

    if node_dim == 0:
      f = [1]
      lab = []
    else:
      f = data.get("features", n_zero_f)
      if node_label_count > 0:
        lab = I_n[data["label"] - 1]
      else:
        lab = []

    n_ids[node] = i
    i += 1
    X.append(np.concatenate((lab, f)))

  for (a, b) in g.edges():
    ref_a.append(n_ids[a])
    ref_b.append(n_ids[b])

  return dict(
    X=np.array(X),
    ref_a=np.array(ref_a),
    ref_b=np.array(ref_b))

def vertex_count(e):
  return len(e["X"])

def total_count(e):
  return len(e["X"]) + len(e["ref_a"])


space_metrics = dict(
  vertex_count=vertex_count,
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

class WL1Batcher(batcher.Batcher):
  def __init__(self, masking=False, space_metric="vertex_count", **kwargs):
    super().__init__(**kwargs)
    assert space_metric in space_metrics, "Unknown WL1 space metric."
    self.masking = masking
    self.space_metric = space_metric

    name = "wl1"

    if self.batch_space_limit is not None:
      name += f"_{space_metric}_metric"
    if masking:
      name += "_masked"

    self.name = name

  def finalize(self, graphs):
    return enc_utils.make_graph_batch(
      graphs,
      ref_keys=["ref_a", "ref_b"],
      masking_fns=dict(
        ref_a_idx=lambda e: e["ref_a"],
        ref_b_idx=lambda e: e["ref_b"]) if self.masking else None,
      meta_fns=dict(
        n=vertex_count))

  def compute_space(self, graph):
    return space_metrics[self.space_metric](graph)
