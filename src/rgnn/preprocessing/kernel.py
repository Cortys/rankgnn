from abc import ABCMeta, abstractmethod

import rgnn.preprocessing.encoder as encoder

class KernelEncoder(encoder.Encoder, metaclass=ABCMeta):
  name = None
  uses_train_metadata = True

  def __init__(self, nystroem=False):
    super().__init__()
    self.nystroem = nystroem

    if nystroem:
      self.can_slice_encoded = False

  @abstractmethod
  def _compute_kernel(self, graphs, gram=True):
    pass

  @abstractmethod
  def _apply_kernel(self, kernel, graphs):
    pass

  def slice_encoded(self, gram, indices, train_indices=None):
    assert train_indices is not False, "Presplit data not supported."
    if train_indices is None:
      train_indices = indices
    return gram[indices, :][:, train_indices]

  def encode_element(self, graph):
    raise AssertionError(
      "Graph-by-graph transformation does not work for gram computation.")

  def produce_train_metadata(self, graphs):
    return self._compute_kernel(graphs, gram=False)

  def transform(self, graphs, train_metadata=None):
    if train_metadata is None:
      return self._compute_kernel(graphs)
    else:
      return self._apply_kernel(train_metadata, graphs)
