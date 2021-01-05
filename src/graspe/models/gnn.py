import tensorflow as tf
from tensorflow import keras

import graspe.utils as utils
import graspe.chaining.model as cm
import graspe.chaining.keras as ck
import graspe.preprocessing.tf as tf_enc
import graspe.layers.wl1 as wl1
import graspe.layers.wl2 as wl2
import graspe.layers.pooling as pooling

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
def pool(input, out_enc, out_meta=None):
  pool = pooling.AvgPooling()
  return pool(input)

@cm.model_step
def finalize(input, out_enc, out_meta=None, squeeze_output=False):
  if squeeze_output or out_enc == "binary":
    return tf.squeeze(input, -1)
  else:
    return input


Dense = utils.tolerant(keras.layers.Dense, ignore_varkwargs=True)

GIN = ck.create_model("GIN", [
  inputs,
  cm.with_layers(wl1.GINLayer, prefix="conv"),
  pool,
  cm.with_layers(Dense, prefix="fc"),
  finalize],
  input_encodings=["wl1"],
  output_encodings=global_target_encs)

WL2GNN = ck.create_model("WL2GNN", [
  inputs,
  cm.with_layers(wl2.WL2Layer, prefix="conv"),
  pool,
  cm.with_layers(Dense, prefix="fc"),
  finalize],
  input_encodings=["wl2"],
  output_encodings=global_target_encs)
