import tensorflow as tf
from tensorflow import keras

class PrefLookupLayer(keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, input):
    X, pref_a, pref_b = input

    return pref_lookup(X, pref_a, pref_b)

def pref_lookup(X, pref_a, pref_b):
  X_a = tf.gather(X, pref_a, axis=0)
  X_b = tf.gather(X, pref_b, axis=0)
  return X_b - X_a

class PrefToBinaryLayer(keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, X):
    X = tf.squeeze(X, -1)
    return pref_to_binary(X)

def pref_to_binary(X):
  return (X + 1) / 2
