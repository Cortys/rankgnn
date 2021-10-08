import networkx as nx
import numpy as np

def create_graph_from_ogb(encoding):
  edge_index = encoding["edge_index"]
  edge_feat = encoding["edge_feat"]
  node_feat = encoding["node_feat"]
  edges = np.transpose(edge_index)

  g = nx.Graph()

  for i, v in enumerate(node_feat):
    g.add_node(i, features=v)

  for edge, feat in zip(edges, edge_feat):
    n1, n2 = edge

    g.add_edge(n1, n2, features=feat)

  return g
