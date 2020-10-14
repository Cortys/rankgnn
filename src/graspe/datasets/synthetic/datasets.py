from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import networkx as nx
import funcy as fy

from graspe.utils import cart, local_seed, unzip

def noisy_triangle_graph(sl, sr, d, y):
  g = nx.Graph()
  y_not = 1 - y
  t_nodes = np.arange(3)
  l_nodes = np.arange(sl)
  r_nodes = np.arange(sr) + sl

  g.add_nodes_from(l_nodes, label=y)
  g.add_nodes_from(r_nodes, label=y_not)
  nx.add_cycle(g, t_nodes)

  def try_adding_non_triangle_edge(g, nodes):
    u = np.random.choice(nodes)
    v = np.random.choice(nodes)

    if len(list(nx.common_neighbors(g, u, v))) > 0:
      return

    g.add_edge(u, v)

  for _ in range(int(0.5 * sl * (sl + 1) * d)):
    try_adding_non_triangle_edge(g, l_nodes)

  for _ in range(int(0.5 * sr * (sr + 1) * d)):
    try_adding_non_triangle_edge(g, r_nodes)

  for _ in range(int(sr * sl * d)):
    ln = np.random.choice(l_nodes)
    rn = np.random.choice(r_nodes)
    g.add_edge(ln, rn)

  return g, y

def triangle_classification_dataset(seed=1337):
  with local_seed(seed):
    sizes = range(3, 10)
    balances = [(1, 1), (1, 3), (3, 1)]
    dens = [0.25, 0.5]
    classes = [0, 1]

    max_verts = 33
    repeat = 3
    configs = [
      (s * l, s * r, d, y)
      for s, (l, r), d, y in cart(sizes, balances, dens, classes)
      if s * (l + r) <= max_verts]

    graphs, ys = unzip([
      entry
      for config in configs
      for entry in fy.repeatedly(
        fy.partial(noisy_triangle_graph, *config), repeat)])

    return graphs, np.array(ys)
