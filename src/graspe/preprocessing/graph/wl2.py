import numpy as np
import networkx as nx

from graspe.utils import tolerant
import graspe.preprocessing.utils as enc_utils
import graspe.preprocessing.batcher as batcher
import graspe.preprocessing.encoder as encoder

@tolerant
def feature_dim(
  node_feature_dim=0, node_label_count=0,
  edge_feature_dim=0, edge_label_count=0):
  return 3 + node_feature_dim + node_label_count \
    + edge_feature_dim + edge_label_count

def eid_lookup(e_ids, i, j):
  if i > j:
    i, j = j, i

  return e_ids[(i, j)]

def encode_graph(
  g, radius=1,
  node_feature_dim=0, node_label_count=0,
  edge_feature_dim=0, edge_label_count=0):
  assert radius >= 1
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
    print(i, edge)
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
    elif g.has_edge(a, b):
      X[i, 1] = 1
      d = g.edges[edge]
      if edge_label_count > 0:
        label = I_e[d.get("label", 0) - 1]
        X[i, edge_label_offset:edge_feature_offset] = label
      if edge_feature_dim > 0:
        X[i, edge_feature_offset:X_dim] = d["features"]
    else:
      X[i, 2] = 1

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
  def __init__(
    self, radius=1,
    node_feature_dim=0, node_label_count=0,
    edge_feature_dim=0, edge_label_count=0):
    self.radius = radius
    self.node_feature_dim = node_feature_dim
    self.node_label_count = node_label_count
    self.edge_feature_dim = edge_feature_dim
    self.edge_label_count = edge_label_count
    self.name = f"wl2_r{radius}"

  def encode_element(self, graph):
    return encode_graph(
      graph, self.radius,
      self.node_feature_dim, self.node_label_count,
      self.eedge_feature_dim, self.edge_label_count)

class WL2Batcher(batcher.Batcher):
  def __init__(self, space_metric="embeddings_count", **kwargs):
    super().__init__(**kwargs)
    assert space_metric in space_metrics, "Unknown WL2 space metric."
    self.space_metric = space_metric
    if self.batch_space_limit is not None:
      self.name = f"wl2_{space_metric}_metric"
    else:
      self.name = "wl2"

  def finalize(self, graphs):
    return enc_utils.make_graph_batch(
      graphs,
      ref_keys=["ref_a", "ref_b", "backref"],
      meta_fns=dict(n=vertex_count))

  def compute_space(self, graph):
    return space_metrics[self.space_metric](graph)
