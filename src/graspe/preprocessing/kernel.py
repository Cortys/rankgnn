from abc import ABCMeta, abstractmethod
import grakel as gk
import funcy as fy

import graspe.preprocessing.encoder as encoder
import graspe.preprocessing.preprocessor as preproc

class KernelEncoder(encoder.Encoder, metaclass=ABCMeta):
  name = None

  def __init__(self, node_label_count=0, edge_label_count=0, **config):
    self.node_labels = node_label_count > 0
    self.edge_labels = edge_label_count > 0

  @abstractmethod
  def _compute_kernel(self, graphs):
    pass

  def preprocess(self, graphs):
    return self._compute_kernel(graphs)

  def slice(self, gram, indices, train_indices=None):
    assert train_indices is not False, "Presplit data not yet supported."

    if train_indices is None:
      train_indices = indices
    return gram[indices, :][:, train_indices]

  def encode_element(self, graph):
    raise AssertionError(
      "Graph-by-graph transformation does not work for gram computation.")

  def transform(self, graphs):
    return self.preprocess(graphs)

class GrakelEncoder(KernelEncoder):
  def __init__(self, name, kernel, **config):
    super().__init__(**config)
    self.name = name
    self.kernel = gk.GraphKernel(kernel=kernel)

  def _compute_kernel(self, graphs):
    graphs = gk.graph_from_networkx(
      graphs,
      node_labels_tag="label" if self.node_labels else None,
      edge_labels_tag="label" if self.edge_labels else None)

    if not self.node_labels:
      graphs = fy.map(
        lambda r: [
          r[0],
          {k: 1 for k in r[0].keys()},
          r[1]],
        graphs)

    return self.kernel.fit_transform(graphs)

def create_preprocessor(type, enc, in_encoder=None, out_encoder=None):
  class KernelPreprocessor(preproc.Preprocessor):
    if in_encoder is not None:
      in_encoder_gen = in_encoder
    if out_encoder is not None:
      out_encoder_gen = out_encoder

  preproc.register_preprocessor(type, enc, KernelPreprocessor)
  return KernelPreprocessor

def create_graph_preprocessors(in_enc, in_encoder_gen):
  # Regression:
  create_preprocessor(
    ("graph", "integer"), (in_enc, "float"),
    in_encoder_gen)
  create_preprocessor(
    ("graph", "float"), (in_enc, "float"),
    in_encoder_gen)

  # Classification:
  create_preprocessor(
    ("graph", "binary"), (in_enc, "class"),
    in_encoder_gen)
  create_preprocessor(
    ("graph", "integer"), (in_enc, "class"),
    in_encoder_gen)


input_encodings = ["gram_wlst", "gram_wlsp"]
output_encodings = ["class", "float"]

create_graph_preprocessors("gram_wlst", lambda T=5, **config: GrakelEncoder(
  f"gram_wlst_{T}", [dict(name="WL", n_iter=T), "VH"], **config))

create_graph_preprocessors("gram_wlsp", lambda T=5, **config: GrakelEncoder(
  f"gram_wlsp_{T}", [dict(name="WL", n_iter=T), "SP"], **config))
