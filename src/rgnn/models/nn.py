import tensorflow as tf
from tensorflow import keras

import rgnn.utils as utils
import rgnn.chaining.model as cm
import rgnn.chaining.keras as ck
import rgnn.preprocessing.tf as tf_enc

@cm.model_inputs
def inputs(in_enc, in_meta={}):
  return tf_enc.make_inputs(in_enc, in_meta)

@cm.model_step
def finalize(input, out_enc, out_meta=None, squeeze_output=False):
  if squeeze_output or out_enc == "binary" or out_enc == "float" \
      or out_enc == "rank_normalized":
    return tf.squeeze(input, -1)
  else:
    return input


Dense = utils.tolerant(keras.layers.Dense, ignore_varkwargs=True)
Activation = utils.tolerant(keras.layers.Activation, ignore_varkwargs=True)

MLP = ck.create_model("MLP", [
  inputs,
  cm.with_layers(Dense, prefix="fc"),
  finalize],
  input_encodings=tf_enc.vector_input_encodings,
  output_encodings=tf_enc.output_encodings)
