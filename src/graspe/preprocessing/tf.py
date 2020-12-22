import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

import graspe.preprocessing.utils as enc_utils
import graspe.preprocessing.batcher as batcher
import graspe.preprocessing.encoder as encoder
import graspe.preprocessing.graph.wl1 as wl1

def wl1(meta):
  return {
    "types": {
      "X": tf.float32,
      "ref_a": tf.int32,
      "ref_b": tf.int32,
      "graph_idx": tf.int32,
      "n": tf.int32,
    },
    "shapes": {
      "X": tf.TensorShape([None, meta["feature_dim"]]),
      "ref_a": tf.TensorShape([None]),
      "ref_b": tf.TensorShape([None]),
      "graph_idx": tf.TensorShape([None]),
      "n": tf.TensorShape([None]),
    }
  }

def float32(meta):
  return {
    "types": tf.float32,
    "shapes": tf.TensorShape([None])
  }

def float32_vector(meta):
  return {
    "types": tf.float32,
    "shapes": tf.TensorShape([None, meta["feature_dim"]])
  }

def pair(encoder):
  def pair_enc(meta, meta2=None):
    encoding1 = encoder(meta)
    encoding2 = encoding1 if meta2 is None else encoder(meta2)

    return {
      "types": (encoding1["types"], encoding2["types"]),
      "shapes": (encoding1["shapes"], encoding2["shapes"])
    }

  return pair_enc


encodings = dict(
  wl1=wl1,
  wl1_pair=pair(wl1),
  float32=float32,
  float32_vector=float32_vector
)

def make_dataset(batch_generator, meta_in, meta_out=None):
  input_enc = encodings[meta_in["encoding"]](meta_in)

  if meta_out is None:
    types = input_enc["types"]
    shapes = input_enc["shapes"]
  else:
    output_enc = encodings[meta_out["encoding"]](meta_out)
    types = (input_enc["types"], output_enc["types"])
    shapes = (input_enc["shapes"], output_enc["shapes"])

  return tf.data.Dataset.from_generator(
    batch_generator,
    output_types=types,
    output_shapes=shapes)

def make_inputs(meta):
  enc = encodings[meta["encoding"]](meta)
  types = enc["types"]
  shapes = enc["shapes"]
  ks = types.keys()

  return tuple(
    keras.Input(name=k, dtype=types[k], shape=tuple(shapes[k].as_list()))
    for k in ks)
