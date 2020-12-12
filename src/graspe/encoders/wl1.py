import numpy as np

import graspe.encoders.utils as enc_utils

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

space_fns = dict(
  vertex_count=vertex_count,
  total_count=total_count
)

def make_batcher(masking=False, space_metric="vertex_count"):
  @enc_utils.with_space_fn(space_fns[space_metric])
  def batcher(encoded_graphs):
    return enc_utils.make_graph_batch(
      encoded_graphs,
      ref_keys=["ref_a", "ref_b"],
      masking_fns=dict(
        ref_a_idx=lambda e: e["ref_a"],
        ref_b_idx=lambda e: e["ref_b"]) if masking else None,
      meta_fns=dict(
        n=vertex_count))

  return batcher
