import tensorflow as tf
import tensorflow.keras as keras

class AvgPooling(keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, input):
    X = input["X"]
    graph_idx = input["graph_idx"]
    N = tf.shape(input["n"])[0]
    y = tf.math.unsorted_segment_mean(X, graph_idx, num_segments=N)

    return y
