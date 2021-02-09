import numpy as np
import networkx as nx

from graspe.utils import tolerant
import graspe.preprocessing.utils as enc_utils
import graspe.preprocessing.batcher as batcher
import graspe.preprocessing.encoder as encoder
import graspe.preprocessing.preference.utility as pref_util

@tolerant
def feature_dim(
  node_feature_dim=0, node_label_count=0,
  edge_feature_dim=0, edge_label_count=0,
  ignore_node_features=False, ignore_node_labels=False,
  ignore_edge_features=False, ignore_edge_labels=False):
  if ignore_node_features:
    node_feature_dim = 0
  if ignore_node_labels:
    node_label_count = 0
  if ignore_edge_features:
    edge_feature_dim = 0
  if ignore_edge_labels:
    edge_label_count = 0

  return 3 + node_feature_dim + node_label_count \
      + edge_feature_dim + edge_label_count

def eid_lookup(e_ids, i, j):
  if i > j:
    i, j = j, i

  return e_ids[(i, j)]

def encode_graph(
  g, radius=1,
  node_feature_dim=0, node_label_count=0,
  edge_feature_dim=0, edge_label_count=0,
  ignore_node_features=False, ignore_node_labels=False,
  ignore_edge_features=False, ignore_edge_labels=False):
  assert radius >= 1

  if ignore_node_features:
    node_feature_dim = 0
  if ignore_node_labels:
    node_label_count = 0
  if ignore_edge_features:
    edge_feature_dim = 0
  if ignore_edge_labels:
    edge_label_count = 0

  g_p = nx.power(g, radius)
  for node in g.nodes:
    g_p.add_edge(node, node)

  e_count = g_p.size()
  X_dim = feature_dim(
    node_feature_dim, node_label_count,
    edge_feature_dim, edge_label_count)
  X = np.zeros((e_count, X_dim), dtype=np.float32)
  ref_a = []
  ref_b = []
  backref = []
  I_n = np.eye(node_label_count)
  I_e = np.eye(edge_label_count)
  node_label_offset = 3
  node_feature_offset = node_label_offset + node_label_count
  edge_label_offset = node_feature_offset + node_feature_dim
  edge_feature_offset = edge_label_offset + edge_label_count
  e_ids = {e: i for i, e in enumerate(g_p.edges)}
  i = 0
  for edge in g_p.edges:
    a, b = edge
    neighbors = list(nx.common_neighbors(g_p, a, b))
    n_count = len(neighbors)
    n_a = [eid_lookup(e_ids, a, k) for k in neighbors]
    n_b = [eid_lookup(e_ids, b, k) for k in neighbors]

    if a == b:
      X[i, 0] = 1
      d = g.nodes[a]
      if node_label_count > 0:
        label = I_n[d.get("label", 0) - 1]
        X[i, node_label_offset:node_feature_offset] = label
      if node_feature_dim > 0:
        X[i, node_feature_offset:edge_label_offset] = d["features"]

      ref_a.append(i)
      ref_b.append(i)
      n_count += 1
    else:
      if g.has_edge(a, b):
        X[i, 1] = 1
        d = g.edges[edge]
        if edge_label_count > 0:
          label = I_e[d.get("label", 0) - 1]
          X[i, edge_label_offset:edge_feature_offset] = label
        if edge_feature_dim > 0:
          X[i, edge_feature_offset:X_dim] = d["features"]
      else:
        X[i, 2] = 1
      ref_a += [i, eid_lookup(e_ids, a, a)]
      ref_b += [eid_lookup(e_ids, b, b), i]
      n_count += 2

    ref_a += n_a
    ref_b += n_b
    backref += [i] * n_count
    i += 1

  return dict(
    X=X,
    ref_a=np.array(ref_a),
    ref_b=np.array(ref_b),
    backref=np.array(backref),
    n=g.order())

def vertex_count(e):
  return e["n"]

def embeddings_count(e):
  return len(e["X"])

def total_count(e):
  return len(e["X"]) + len(e["ref_a"])


space_metrics = dict(
  vertex_count=vertex_count,
  embeddings_count=embeddings_count,
  total_count=total_count
)

class WL2Encoder(encoder.ObjectEncoder):
  name = "wl2"

  def __init__(
    self, radius=1,
    node_feature_dim=0, node_label_count=0,
    edge_feature_dim=0, edge_label_count=0,
    ignore_node_features=False, ignore_node_labels=False,
    ignore_edge_features=False, ignore_edge_labels=False):
    self.radius = radius
    self.node_feature_dim = node_feature_dim
    self.node_label_count = node_label_count
    self.edge_feature_dim = edge_feature_dim
    self.edge_label_count = edge_label_count
    self.ignore_node_features = ignore_node_features
    self.ignore_node_labels = ignore_node_labels
    self.ignore_edge_features = ignore_edge_features
    self.ignore_edge_labels = ignore_edge_labels
    self.name = f"wl2_r{radius}"
    if ignore_node_features and node_feature_dim > 0:
      self.name += "_inf"
    if ignore_node_labels and node_label_count > 0:
      self.name += "_inl"
    if ignore_edge_features and edge_feature_dim > 0:
      self.name += "_ief"
    if ignore_edge_labels and edge_label_count > 0:
      self.name += "_iel"

  def encode_element(self, graph):
    return encode_graph(
      graph, self.radius,
      self.node_feature_dim, self.node_label_count,
      self.edge_feature_dim, self.edge_label_count,
      self.ignore_node_features, self.ignore_node_labels,
      self.ignore_edge_features, self.ignore_edge_labels)

def make_wl2_batch(graphs):
  return enc_utils.make_graph_batch(
    graphs,
    ref_keys=["ref_a", "ref_b", "backref"],
    meta_fns=dict(n=vertex_count))

class WL2Batcher(batcher.Batcher):
  name = "wl2"

  def __init__(self, space_metric="embeddings_count", **kwargs):
    super().__init__(**kwargs)
    assert space_metric in space_metrics, "Unknown WL2 space metric."
    self.space_metric = space_metric
    if self.batch_space_limit is not None:
      suffix = f"_{space_metric}_metric"
      self.name += suffix
      self.basename += suffix

  def finalize(self, graphs):
    return make_wl2_batch(graphs)

  def compute_space(self, graph, batch):
    return space_metrics[self.space_metric](graph)

class WL2UtilityPreferenceBatcher(pref_util.UtilityPreferenceBatcher):
  name = "wl2_util_pref"

  def __init__(self, in_meta, out_meta, **kwargs):
    super().__init__(in_meta, out_meta, **kwargs)
    self.space_metric = in_meta.get("space_metric", "embeddings_count")
    assert self.space_metric in space_metrics, "Unknown WL2 space metric."

    if self.batch_space_limit is not None:
      suffix = f"_{self.space_metric}_metric"
      self.name += suffix
      self.basename += suffix

  def finalize_objects(self, graphs):
    return make_wl2_batch(graphs)

  def compute_object_space(self, graph):
    return space_metrics[self.space_metric](graph)
