import tensorflow as tf
from tensorflow import keras

import graspe.utils as utils
import graspe.chaining.model as cm
import graspe.chaining.keras as ck
import graspe.preprocessing.tf as tf_enc
import graspe.layers.wl1 as wl1
import graspe.layers.wl2 as wl2
import graspe.layers.pooling as pl

import warnings
warnings.filterwarnings(
  "ignore",
  "Converting sparse IndexedSlices*",
  UserWarning)

global_target_encs = ["float32", "binary", "multiclass"]

@cm.model_inputs
def inputs(in_enc, in_meta):
  return tf_enc.make_inputs(in_enc, in_meta)

@cm.model_step
def pool(input, pooling="mean"):
  if pooling == "mean":
    pool = pl.MeanPooling()
  elif pooling == "sum":
    pool = pl.SumPooling()
  elif pooling == "max":
    pool = pl.MaxPooling()
  elif pooling == "min":
    pool = pl.MinPooling()
  elif pooling == "softmax":
    pool = pl.SoftmaxPooling()
  else:
    raise AssertionError(f"Unknown pooling type '{pooling}'.")
  return pool(input)

@cm.model_step(macro=True)
def pooled_layers(_, conv_layer, pooling=None):
  conv_layers = cm.with_layers(conv_layer, prefix="conv")

  if pooling == "softmax":
    att_conv_layers = cm.with_layers(conv_layer, prefix="att_conv")
    return [
      (conv_layers, att_conv_layers),
      cm.merge_ios(pl.merge_attention),
      pool]
  else:
    return [conv_layers, pool]

@cm.model_step
def finalize(input, out_enc, out_meta=None, squeeze_output=False):
  if squeeze_output or out_enc == "binary":
    return tf.squeeze(input, -1)
  else:
    return input


Dense = utils.tolerant(keras.layers.Dense, ignore_varkwargs=True)

GIN = ck.create_model("GIN", [
  inputs,
  pooled_layers(wl1.GINLayer),
  cm.with_layers(Dense, prefix="fc"),
  finalize],
  input_encodings=["wl1"],
  output_encodings=global_target_encs)

WL2GNN = ck.create_model("WL2GNN", [
  inputs,
  pooled_layers(wl2.WL2Layer),
  cm.with_layers(Dense, prefix="fc"),
  finalize],
  input_encodings=["wl2"],
  output_encodings=global_target_encs)
