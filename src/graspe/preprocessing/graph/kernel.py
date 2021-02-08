import grakel as gk
import funcy as fy

from graspe.preprocessing.kernel import KernelEncoder

class GrakelEncoder(KernelEncoder):
  def __init__(
    self, name, kernel,
    node_label_count=0, edge_label_count=0,
    node_feature_dim=0, edge_feature_dim=0,
    discrete_node_features=False, discrete_edge_features=False,
    **config):
    super().__init__()
    self.kernel = gk.GraphKernel(kernel=kernel)
    self.node_labels = node_label_count > 0
    self.edge_labels = edge_label_count > 0
    self.node_features = discrete_node_features and node_feature_dim > 0
    self.edge_features = discrete_edge_features and edge_feature_dim > 0
    self.name = name
    if self.node_features:
      self.name += "_nf"
    if self.edge_features:
      self.name += "_ef"

  def _compute_kernel(self, graphs):
    node_labels_tag = None
    edge_labels_tag = None

    if self.node_labels and not self.node_features:
      node_labels_tag = "label"
    elif not self.node_labels and self.node_features:
      node_labels_tag = "features"

    if self.edge_labels:
      edge_labels_tag = "label"
    elif not self.edge_labels and self.edge_features:
      edge_labels_tag = "features"

    enc_graphs = gk.graph_from_networkx(
      graphs,
      node_labels_tag=node_labels_tag,
      edge_labels_tag=edge_labels_tag)

    node_labeler = lambda i, r: r[1]
    edge_labeler = lambda i, r: r[2]
    relabel = False

    if not self.node_labels and not self.node_features:
      node_labeler = lambda i, r: {k: 1 for k in r[0].keys()}
      relabel = True
    elif self.node_labels and self.node_features:
      node_labeler = lambda i, r: {
        v: str(data["label"]) + "_" + ",".join(data["features"])
        for v, data in graphs[i].nodes(data=True)}
      relabel = True

    if self.edge_labels and self.edge_features:
      def edge_labeler(i, r):
        g = graphs[i]
        return {
          e: str(l) + "_" + ",".join(g.edges[e]["features"])
          for e, l in r[2].items()}
      relabel = True

    if relabel:
      enc_graphs = fy.map(
        lambda r: [
          r[1][0],
          node_labeler(r[0], r[1]),
          edge_labeler(r[0], r[1])],
        enumerate(enc_graphs))

    return self.kernel.fit_transform(enc_graphs)
