import numpy as np

import rgnn.utils as utils
import rgnn.preprocessing.transformer as transformer

class Encoder(transformer.Transformer):
  can_slice_raw = True
  can_slice_encoded = True
  uses_train_metadata = False

  def _make_slicable(self, elements):
    return np.asanyarray(elements)

  def preprocess(self, elements):
    return self._make_slicable(elements)

  def slice_raw(self, elements, indices):
    assert self.can_slice_raw, "Cannot slice raw data."
    return self._make_slicable(elements)[indices]

  def slice_encoded(self, elements, indices, train_indices=None):
    assert self.can_slice_encoded, "Cannot slice encoded data."
    return elements[indices]

  def encode_element(self, element):
    return element

  def append(self, aggregator, element):
    aggregator.append(self.encode_element(element))

  def produce_train_metadata(self, elements):
    return None

  def transform(self, elements, train_metadata=None):
    if self.__class__ == __class__:
      return self.finalize(elements)

    return super().transform(elements)

  def finalize(self, aggregator):
    return self._make_slicable(aggregator)

class ObjectEncoder(Encoder):
  name = "oid"

  def _make_slicable(self, elements):
    if isinstance(elements, np.ndarray):
      return elements

    return utils.obj_array(elements)

class NullEncoder(Encoder):
  name = "null"

  def _make_slicable(self, elements):
    return None

  def slice_raw(self, elements, indices):
    return None

  def slice_encoded(self, elements, indices, train_indices=None):
    return None

  def transform(self, elements, train_metadata=None):
    return None

class TupleEncoder(transformer.TupleTransformer, Encoder):
  def __init__(self, *encoders, size=2):
    can_slice_raw = True
    can_slice_encoded = True
    uses_train_metadata = False

    for enc in encoders:
      can_slice_raw = can_slice_raw and enc.can_slice_raw
      can_slice_encoded = can_slice_encoded and enc.can_slice_encoded
      uses_train_metadata = uses_train_metadata or enc.uses_train_metadata
      assert isinstance(enc, Encoder)

    self.can_slice_raw = can_slice_raw
    self.can_slice_encoded = can_slice_encoded
    super().__init__(*encoders, size=size)

  def slice_raw(self, elements, indices):
    return tuple(
      enc.slice_raw(el, indices)
      for enc, el in zip(self.transformers, elements))

  def slice_encoded(self, elements, indices, train_indices=None):
    return tuple(
      enc.slice_encoded(el, indices, train_indices)
      for enc, el in zip(self.transformers, elements))

  def produce_train_metadata(self, elements):
    if self.uses_train_metadata:
      return tuple(
        enc.produce_train_metadata(el)
        for enc, el in zip(self.transformers, elements))
    else:
      raise AssertionError("This encoder does not use train metadata.")


transformer.register_transformer(Encoder, TupleEncoder)
ObjectEncoder.identity = ObjectEncoder()
NullEncoder.identity = NullEncoder()
