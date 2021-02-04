import numpy as np
import networkx as nx

from graspe.stats.utils import statistics, register_stat_computer

def summarize_graph_stats(stats):
  return dict(
    node_counts=statistics(stats["node_counts"]),
    edge_counts=statistics(stats["edge_counts"]),
    node_degrees=statistics(stats["node_degrees"]),
    radii=statistics(stats["radii"], mask_invalid=True),
    triangles=statistics(stats["triangles"]))

def filter_graph_stats(stats, idxs):
  return dict(
    node_counts=stats["node_counts"][idxs],
    edge_counts=stats["edge_counts"][idxs],
    node_degrees=stats["node_degrees"][idxs],
    radii=stats["radii"][idxs],
    triangles=stats["triangles"][idxs])

def graphs_stats(graphs, summarize=True):
  gl = len(graphs)
  log = gl > 1000
  node_counts = []
  edge_counts = []
  radii = []
  degrees = []
  triangles = []
  i = 0

  for g in graphs:
    if log:
      if i % 500 == 0:
        print(f"[dbg] Graph {i}/{gl}...")
      i += 1
    node_counts.append(g.order())
    edge_counts.append(g.size())
    degrees += [d for n, d in g.degree()]
    try:
      radii.append(nx.algorithms.distance_measures.radius(g))
    except Exception:
      radii.append(np.inf)

    triangles.append(sum(nx.triangles(g).values()) // 3)

  res = dict(
    node_counts=np.array(node_counts),
    edge_counts=np.array(edge_counts),
    node_degrees=np.array(degrees),
    radii=np.array(radii),
    triangles=np.array(triangles))

  return summarize_graph_stats(res) if summarize else res


@register_stat_computer("graph")
def graph_stat_computer(graphs, named_splits=None):
  all_stats = graphs_stats(graphs, summarize=False)

  if named_splits is None:
    return summarize_graph_stats(all_stats)

  stats = dict(all=summarize_graph_stats(all_stats))

  for split_name, split in named_splits.items():
    stats[split_name] = dict(
      model_selection=[dict(
        train=summarize_graph_stats(filter_graph_stats(all_stats, m["train"])),
        val=summarize_graph_stats(filter_graph_stats(
          all_stats, m["validation"])))
        for m in split["model_selection"]],
      test=summarize_graph_stats(filter_graph_stats(all_stats, split["test"]))
    )

  return stats
