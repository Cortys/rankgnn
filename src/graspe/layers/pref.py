import tensorflow as tf
from tensorflow import keras

def pref_lookup(X, pref_a, pref_b):
  X_a = tf.gather(X, pref_a, axis=0)
  X_b = tf.gather(X, pref_b, axis=0)
  return X_a, X_b

class PrefLookupLayer(keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, input):
    X, pref_a, pref_b = input

    return pref_lookup(X, pref_a, pref_b)

class PrefDiffLayer(keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, input):
    X_a, X_b = input

    return X_b - X_a

class CmpLayer(keras.layers.Layer):
  def __init__(self, units, use_bias=True, activation=None):
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
    in_dim = input_shape[0][-1]
    assert in_dim == input_shape[1][-1], \
        "Compared objects must be of equal dimensionality."

    self.W_1 = self.add_weight(
      "W_1", shape=(in_dim, self.units),
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_2 = self.add_weight(
      "W_2", shape=(in_dim, self.units),
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.use_bias:
      self.b = self.add_weight(
        "b", shape=(self.units,),
        trainable=True, initializer=tf.initializers.Zeros)

  def call(self, input):
    X_a, X_b = input
    XW_a1 = X_a @ self.W_1
    XW_b1 = X_b @ self.W_1
    XW_a2 = X_a @ self.W_2
    XW_b2 = X_b @ self.W_2

    XW_ab = XW_a1 + XW_b2
    XW_ba = XW_b1 + XW_a2

    if self.use_bias:
      XW_ab = tf.nn.bias_add(XW_ab, self.b)
      XW_ba = tf.nn.bias_add(XW_ba, self.b)

    Y_ab = self.activation(XW_ab)
    Y_ba = self.activation(XW_ba)

    return Y_ab, Y_ba
