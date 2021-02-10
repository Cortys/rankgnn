import grakel as gk

from graspe.preprocessing.encoder import ObjectEncoder
from graspe.preprocessing.kernel import KernelEncoder

class GrakelEncoder(KernelEncoder, ObjectEncoder):
  def __init__(
    self, name, kernel, nystroem=False,
    node_label_count=0, edge_label_count=0,
    node_feature_dim=0, edge_feature_dim=0,
    discrete_node_features=False, discrete_edge_features=False,
    ignore_node_features=False, ignore_node_labels=False,
    ignore_edge_features=False, ignore_edge_labels=False,
    **config):
    if nystroem == -1:
      nystroem = 100

    super().__init__(nystroem=nystroem)
    self.kernel = gk.GraphKernel(kernel=kernel, Nystroem=nystroem)
    self.node_labels = node_label_count > 0 and not ignore_node_labels
    self.edge_labels = edge_label_count > 0 and not ignore_edge_labels
    self.node_features = (
      discrete_node_features and not ignore_node_features
      and node_feature_dim > 0)
    self.edge_features = (
      discrete_edge_features and not ignore_edge_features
      and edge_feature_dim > 0)
    self.name = name

    if nystroem:
      self.name += f"_nyst{nystroem}"

    if self.node_features:
      self.name += "_nf"
    if ignore_node_labels and node_label_count > 0:
      self.name += "_inl"
    if self.edge_features:
      self.name += "_ef"
    if ignore_edge_labels and edge_label_count > 0:
      self.name += "_iel"

  def _enc_graphs(self, graphs):
    node_labels_tag = None
    edge_labels_tag = None

    if self.node_labels and not self.node_features:
      node_labels_tag = "label"

    if self.edge_labels:
      edge_labels_tag = "label"

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
    elif not self.node_labels and self.node_features:
      node_labeler = lambda i, r: {
        v: ",".join(map(str, data["features"]))
        for v, data in graphs[i].nodes(data=True)}
      relabel = True
    elif self.node_labels and self.node_features:
      node_labeler = lambda i, r: {
        v: str(data["label"]) + "_" + ",".join(map(str, data["features"]))
        for v, data in graphs[i].nodes(data=True)}
      relabel = True

    if self.edge_labels and self.edge_features:
      def edge_labeler(i, r):
        g = graphs[i]
        return {
          e: str(l) + "_" + ",".join(map(str, g.edges[e]["features"]))
          for e, l in r[2].items()}
      relabel = True
    elif not self.edge_labels and self.edge_features:
      def edge_labeler(i, r):
        g = graphs[i]
        return {
          (v, u): ",".join(map(str, g.edges[(v, u)]["features"]))
          for v, ns in r[0].items() for u in ns.keys()}
      relabel = True

    if relabel:
      enc_graphs = [
        [r[0], node_labeler(i, r), edge_labeler(i, r)]
        for i, r in enumerate(enc_graphs)]

    return enc_graphs

  def _compute_kernel(self, graphs, gram=True):
    enc_graphs = self._enc_graphs(graphs)

    if gram:
      return self.kernel.fit_transform(enc_graphs)
    else:
      self.kernel.fit(enc_graphs)
      return self.kernel

  def _apply_kernel(self, kernel, graphs):
    enc_graphs = self._enc_graphs(graphs)
    return kernel.transform(enc_graphs)
