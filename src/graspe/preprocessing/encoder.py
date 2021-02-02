import numpy as np

import graspe.utils as utils
import graspe.preprocessing.transformer as transformer

class Encoder(transformer.Transformer):
  def preprocess(self, elements):
    return np.asanyarray(elements)

  def slice(self, elements, indices, train_indices=None):
    return elements[indices]

  def encode_element(self, element):
    return element

  def append(self, aggregator, element):
    aggregator.append(self.encode_element(element))

  def transform(self, elements):
    if self.__class__ == __class__:
      return self.finalize(elements)

    return super().transform(elements)

  def finalize(self, aggregator):
    return np.asarray(aggregator)

class ObjectEncoder(Encoder):
  name = "oid"

  def preprocess(self, elements):
    if isinstance(elements, np.ndarray):
      return elements

    return utils.obj_array(elements)

  def finalize(self, aggregator):
    if isinstance(aggregator, np.ndarray):
      return aggregator

    return utils.obj_array(aggregator)

class TupleEncoder(transformer.TupleTransformer, Encoder):
  def __init__(self, *encoders, size=2):
    for enc in encoders:
      assert isinstance(enc, Encoder)

    super().__init__(*encoders, size=size)

  def slice(self, elements, indices, train_indices=None):
    return tuple(
      enc.slice(el, indices, train_indices)
      for enc, el in zip(self.transformers, elements))


transformer.register_transformer(Encoder, TupleEncoder)
ObjectEncoder.identity = ObjectEncoder()
