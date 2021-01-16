from __future__ import absolute_import, division, print_function,\
  unicode_literals

import numpy as np
import networkx as nx
import funcy as fy

from graspe.utils import cart, local_seed, unzip
from graspe.datasets.synthetic.provider import synthetic_graph_embed_dataset

@synthetic_graph_embed_dataset(type="binary")
def twothree_dataset():
  g2 = nx.Graph()
  g2.add_edge(0, 1)

  g3 = nx.Graph()
  nx.add_cycle(g3, range(3))

  return [g2, g3], np.array([-1, 1])

@synthetic_graph_embed_dataset(type="binary")
def threesix_dataset():
  g3 = nx.Graph()
  nx.add_cycle(g3, range(3))
  nx.add_cycle(g3, range(3, 6))

  g6 = nx.Graph()
  nx.add_cycle(g6, range(6))

  return [g3, g6], np.array([0, 1])

@synthetic_graph_embed_dataset(type="binary")
def small_grid_dataset():
  g10 = nx.grid_graph(dim=(10, 10))
  g20 = nx.grid_graph(dim=(20, 20))
  g50 = nx.grid_graph(dim=(50, 50))

  h10 = nx.grid_graph(dim=(5, 20))
  h20 = nx.grid_graph(dim=(10, 40))
  h50 = nx.grid_graph(dim=(25, 100))

  return [h10, h20, h50, g10, g20, g50], [0, 0, 0, 1, 1, 1]

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

@synthetic_graph_embed_dataset(type="binary")
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

def geo(r, n):
  if r == 1:
    return n + 1

  return (r ** (n + 1) - 1) // (r - 1)

def triangle_count_graph(height, arity, triangles, cycles):
  g = nx.balanced_tree(arity, height)
  max_idx = geo(arity, height - 1)
  n = geo(arity, height)
  off = 0
  if arity == 1:
    max_idx -= 1
    off = 1

  if max_idx < 1:
    idxs = [] if off == 0 or triangles == 0 or height < 2 else [1]
  else:
    idxs = np.random.choice(max_idx, triangles, replace=False) + off

  for i in idxs:
    ns = np.array(list(nx.neighbors(g, i)))
    np.random.shuffle(ns)
    n1, n2 = fy.first(
      (n1, n2)
      for n1 in ns for n2 in ns
      if n1 != n2
      and not g.has_edge(n1, n2)
      and len(list(nx.common_neighbors(g, n1, n2))) == 1)
    g.add_edge(n1, n2)

  if cycles > 0:
    cycles *= max(1, n / 10)
    for _ in range(int(cycles)):
      n1, n2 = np.random.choice(n, 2, replace=False)
      if not g.has_edge(n1, n2) \
          and len(list(nx.common_neighbors(g, n1, n2))) == 0:
        g.add_edge(n1, n2)

  # count = sum(nx.triangles(g).values()) // 3
  # assert triangles == count

  return g, triangles

@synthetic_graph_embed_dataset(type="vector")
def triangle_count_dataset(seed=1337):
  with local_seed(seed):
    heights = range(2, 8)
    arities = range(1, 5)
    triangles = range(0, 10)
    cycles = [0, 1, 3, 6, 9]
    max_verts = 100
    repeat = 3
    configs = [
      (h, a, t, c)
      for h, a, t, c in cart(heights, arities, triangles, cycles)
      if c + t + a ** h + (1 if a == 1 else 0) <= geo(a, h) <= max_verts
    ]

    bins = fy.group_by(lambda e: (e[0].size(), e[0].order(), e[1]), [
      entry
      for config in configs
      for entry in fy.repeatedly(
        fy.partial(triangle_count_graph, *config),
        repeat if config[-2] + config[-1] > 0 else 1)
    ]).values()

    graphs = []
    ys = []

    for bin in bins:
      bin_graphs = []
      bin_ys = []
      for g, y in bin:
        is_new = True
        for g2 in bin_graphs:
          if nx.could_be_isomorphic(g, g2):
            is_new = False
            break
        if not is_new:
          continue
        bin_graphs.append(g)
        bin_ys.append(y)
      graphs += bin_graphs
      ys += bin_ys

    return graphs, np.array(ys)
