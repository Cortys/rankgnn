import tensorflow as tf
from tensorflow import keras

class GINLayer(keras.layers.Layer):
  def __init__(
    self, units, use_bias=True, activation=None):
    super().__init__()
    self.units = units
    self.use_bias = use_bias
    self.activation = keras.activations.get(activation)

  def get_config(self):
    return dict(
      **super().get_config(),
      units=self.units,
      use_bias=self.use_bias,
      activation=keras.activations.serialize(self.activation))

  def build(self, input_shape):
    X_shape = input_shape["X"]
    vert_dim = X_shape[-1]
    hidden_dim = self.units

    self.W_hidden = self.add_weight(
      "W_hidden", shape=(vert_dim, hidden_dim),
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_out = self.add_weight(
      "W_out", shape=(hidden_dim, self.units),
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.use_bias:
      self.b_hidden = self.add_weight(
        "b_hidden", shape=(hidden_dim,),
        trainable=True, initializer=tf.initializers.Zeros)
      self.b_out = self.add_weight(
        "b_out", shape=(self.units,),
        trainable=True, initializer=tf.initializers.Zeros)

  def call(self, input):
    X = input["X"]
    ref_a = input["ref_a"]
    ref_b = input["ref_b"]
    X_agg = wl1_convolution(X, ref_a, ref_b)

    X_hid = tf.matmul(X_agg, self.W_hidden)
    if self.use_bias:
      X_hid = tf.nn.bias_add(X_hid, self.b_hidden)
    X_out = tf.matmul(X_hid, self.W_out)
    if self.use_bias:
      X_out = tf.nn.bias_add(X_out, self.b_out)
    X_out = self.activation(X_out)

    return {**input, "X": X_out}

def wl1_convolution(X, ref_a, ref_b):
  X_a = tf.gather(X, ref_a, axis=0)
  X_shape = tf.shape(X)
  backref = tf.expand_dims(ref_b, axis=-1)
  X_agg = tf.scatter_nd(backref, X_a, shape=X_shape)
  return X_agg
