from abc import ABCMeta, abstractmethod

import graspe.preprocessing.encoder as encoder

class KernelEncoder(encoder.Encoder, metaclass=ABCMeta):
  name = None

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
