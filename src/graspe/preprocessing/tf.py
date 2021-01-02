import tensorflow as tf
import tensorflow.keras as keras

from graspe.utils import fully_tolerant
import graspe.preprocessing.preprocessor as preprocessor
import graspe.preprocessing.graph.wl1 as wl1_enc
import graspe.preprocessing.classification as cls_enc

@fully_tolerant
def wl1(meta):
  feature_dim = wl1_enc.feature_dim(**meta)

  return {
    "types": {
      "X": tf.float32,
      "ref_a": tf.int32,
      "ref_b": tf.int32,
      "graph_idx": tf.int32,
      "n": tf.int32,
    },
    "shapes": {
      "X": tf.TensorShape([None, feature_dim]),
      "ref_a": tf.TensorShape([None]),
      "ref_b": tf.TensorShape([None]),
      "graph_idx": tf.TensorShape([None]),
      "n": tf.TensorShape([None]),
    }
  }

@fully_tolerant
def float32(meta):
  shape = [None, meta["feature_dim"]] if "feature_dim" in meta else [None]

  return {
    "types": tf.float32,
    "shapes": tf.TensorShape(shape)
  }

@fully_tolerant
def multiclass(meta):
  return float32(dict(feature_dim=meta.get("classes", 2)))

def pair(enc):
  @fully_tolerant
  def pair_enc(meta):
    encoding = enc(meta)

    return {
      "types": (encoding["types"], encoding["types"]),
      "shapes": (encoding["shapes"], encoding["shapes"])
    }

  return pair_enc


encodings = dict(
  wl1=wl1,
  wl1_pair=pair(wl1),
  float32=float32,
  binary=float32,
  multiclass=multiclass
)

def make_dataset(
  batch_generator, in_enc, in_meta=None, out_enc=None, out_meta=None):
  input_enc = encodings[in_enc](in_meta)

  if out_enc is None:
    types = input_enc["types"]
    shapes = input_enc["shapes"]
  else:
    output_enc = encodings[out_enc](out_meta)
    types = (input_enc["types"], output_enc["types"])
    shapes = (input_enc["shapes"], output_enc["shapes"])

  return tf.data.Dataset.from_generator(
    batch_generator,
    output_types=types,
    output_shapes=shapes)

def make_inputs(enc, meta={}):
  enc = encodings[enc](meta)
  types = enc["types"]
  shapes = enc["shapes"]
  ks = types.keys()

  return {
    k: keras.Input(
      name=k, dtype=types[k], shape=tuple(shapes[k].as_list()[1:]))
    for k in ks}

class TFPreprocessor(preprocessor.BatchingPreprocessor):
  enc = None

  def finalize(self, elements):
    batch_gen = self.batcher.batch_generator(elements)
    in_enc, out_enc = self.enc
    if not self.has_out:
      out_enc = None

    return make_dataset(
      batch_gen,
      in_enc, self.in_args,
      out_enc, self.out_args)

def create_preprocessor(
  type, enc,
  in_encoder=None, in_batcher=None,
  out_encoder=None, out_batcher=None):
  e = enc

  class Preprocessor(TFPreprocessor):
    enc = e
    if in_encoder is not None:
      in_encoder_gen = in_encoder
    if out_encoder is not None:
      out_encoder_gen = out_encoder
    if in_batcher is not None:
      in_batcher_gen = in_batcher
    if out_batcher is not None:
      out_batcher_gen = out_batcher

  preprocessor.register_preprocessor(type, enc, Preprocessor)
  return Preprocessor


WL1EmbedPreprocessor = create_preprocessor(
  ("graph", "vector"), ("wl1", "float32"),
  wl1_enc.WL1Encoder, wl1_enc.WL1Batcher)

WL1BinaryClassifyPreprocessor = create_preprocessor(
  ("graph", "binary"), ("wl1", "binary"),
  wl1_enc.WL1Encoder, wl1_enc.WL1Batcher)

WL1BinaryClassifyPreprocessor = create_preprocessor(
  ("graph", "binary"), ("wl1", "multiclass"),
  wl1_enc.WL1Encoder, wl1_enc.WL1Batcher,
  cls_enc.MulticlassEncoder)

WL1BinaryClassifyPreprocessor = create_preprocessor(
  ("graph", "class"), ("wl1", "multiclass"),
  wl1_enc.WL1Encoder, wl1_enc.WL1Batcher,
  cls_enc.MulticlassEncoder)
