import tensorflow as tf
from tensorflow import keras

class WL2Layer(keras.layers.Layer):
  def __init__(
    self, units, use_bias=True, activation=None, inner_activation=None):
    super().__init__()
    self.units = units
    self.use_bias = use_bias
    self.activation = keras.activations.get(activation)
    self.inner_activation = keras.activations.get(inner_activation)

  def get_config(self):
    return dict(
      **super().get_config(),
      units=self.units,
      use_bias=self.use_bias,
      activation=keras.activations.serialize(self.activation),
      inner_activation=keras.activations.serialize(self.inner_activation))

  def build(self, input_shape):
    X_shape = input_shape["X"]
    X_dim = X_shape[-1]

    self.W_local = self.add_weight(
      "W_local", shape=(X_dim, self.units),
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_filter = self.add_weight(
      "W_filter", shape=(X_dim, self.units),
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_neighbor = self.add_weight(
      "W_neighbor", shape=(X_dim, self.units),
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.use_bias:
      self.b = self.add_weight(
        "b", shape=(self.units,),
        trainable=True, initializer=tf.initializers.Zeros)
      self.b_neighbor = self.add_weight(
        "b_neighbor", shape=(self.units,),
        trainable=True, initializer=tf.initializers.Zeros)

  def call(self, input):
    X = input["X"]
    ref_a = input["ref_a"]
    ref_b = input["ref_b"]
    backref = input["backref"]
    XW_local = X @ self.W_local
    XW_filter = X @ self.W_filter
    XW_neighbor = X @ self.W_neighbor
    X_conv = wl2_convolution(
      XW_neighbor, ref_a, ref_b, backref, self.combine)
    X_agg = XW_local + XW_filter * X_conv

    if self.use_bias:
      X_agg = tf.nn.bias_add(X_agg, self.b)

    X_out = self.activation(X_agg)

    return {**input, "X": X_out}

  def combine(self, X_a, X_b):
    X_ab = X_a + X_b

    if self.use_bias:
      X_ab = tf.nn.bias_add(X_ab, self.b_neighbor)

    return self.inner_activation(X_ab)

def wl2_convolution(X, ref_a, ref_b, backref, combinator):
  X_a = tf.gather(X, ref_a, axis=0)
  X_b = tf.gather(X, ref_b, axis=0)
  X_ab = combinator(X_a, X_b)
  backref = tf.expand_dims(backref, axis=-1)
  X_shape = tf.shape(X)
  X_agg = tf.scatter_nd(backref, X_ab, shape=X_shape)
  return X_agg
