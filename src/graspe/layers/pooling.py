import tensorflow as tf
import tensorflow.keras as keras
from abc import ABCMeta, abstractmethod

class SegmentPooling(keras.layers.Layer, metaclass=ABCMeta):
  def __init__(self):
    super().__init__()

  def call(self, input):
    X = input["X"]
    graph_idx = input["graph_idx"]
    N = tf.shape(input["n"])[0]
    y = self.pool(X, graph_idx, num_segments=N)
    return y

  @staticmethod
  @abstractmethod
  def pool(X, graph_idx, num_segments): pass

class MeanPooling(SegmentPooling):
  pool = staticmethod(tf.math.unsorted_segment_mean)

class SumPooling(SegmentPooling):
  pool = staticmethod(tf.math.unsorted_segment_sum)

class MaxPooling(SegmentPooling):
  pool = staticmethod(tf.math.unsorted_segment_max)

class MinPooling(SegmentPooling):
  pool = staticmethod(tf.math.unsorted_segment_min)

class SoftmaxPooling(keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, input):
    X = input["X"]
    X_att = input["X_att"]
    graph_idx = input["graph_idx"]
    N = tf.shape(input["n"])[0]

    X_att = tf.exp(X_att)
    X = X_att * X
    y_att = tf.math.unsorted_segment_sum(
      X_att, graph_idx, num_segments=N)
    y = tf.math.unsorted_segment_sum(
      X, graph_idx, num_segments=N)
    return y / y_att

def merge_attention(inputs):
  input, input_att = inputs

  return {**input, "X_att": input_att["X"]}
