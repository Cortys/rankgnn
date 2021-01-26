import tensorflow as tf
import tensorflow.keras as keras

import graspe.utils as utils
import graspe.preprocessing.preprocessor as preprocessor
import graspe.preprocessing.graph.wl1 as wl1_enc
import graspe.preprocessing.graph.wl2 as wl2_enc
import graspe.preprocessing.classification as cls_enc

def wl1(meta):
  feature_dim = wl1_enc.feature_dim(**meta)

  return {
    "X": tf.TensorSpec(shape=[None, feature_dim], dtype=tf.float32),
    "ref_a": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "ref_b": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "graph_idx": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "n": tf.TensorSpec(shape=[None], dtype=tf.int32),
  }

def wl2(meta):
  feature_dim = wl2_enc.feature_dim(**meta)

  return {
    "X": tf.TensorSpec(shape=[None, feature_dim], dtype=tf.float32),
    "ref_a": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "ref_b": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "backref": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "graph_idx": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "n": tf.TensorSpec(shape=[None], dtype=tf.int32),
  }

def vec32(meta):
  shape = [None, meta["feature_dim"]] if "feature_dim" in meta else [None]

  return tf.TensorSpec(shape=shape, dtype=tf.float32)

def multiclass(meta):
  if "classes" in meta:
    classes = meta["classes"]
  elif "max" in meta:
    classes = 1 + meta["max"] - meta.get("min", 0)
  else:
    classes = 2
  return vec32(dict(feature_dim=classes))

def pref(enc):
  def pref_enc(meta):
    signature = enc(meta)
    return {
      **signature,
      "pref_a": tf.TensorSpec(shape=[None], dtype=tf.int32),
      "pref_b": tf.TensorSpec(shape=[None], dtype=tf.int32)
    }
  return pref_enc


encodings = dict(
  wl1=wl1,
  wl1_pref=pref(wl1),
  wl2=wl2,
  wl2_pref=pref(wl2),
  float32=vec32,
  vec32=vec32,
  binary=vec32,
  multiclass=multiclass
)

def make_dataset(
  batch_generator, in_enc, in_meta=None, out_enc=None, out_meta=None,
  lazy_batching=True):
  input_sig = encodings[in_enc](in_meta)

  if out_enc is None:
    signature = input_sig
  else:
    output_sig = encodings[out_enc](out_meta)
    signature = (input_sig, output_sig)

  if lazy_batching:
    gen = batch_generator
  else:
    batches = list(batch_generator())
    gen = lambda: batches

  return tf.data.Dataset.from_generator(
    gen, output_signature=signature)

def make_inputs(enc, meta={}):
  spec = encodings[enc](meta)

  return {
    k: keras.Input(
      name=k, dtype=s.dtype, shape=tuple(s.shape.as_list()[1:]))
    for k, s in spec.items()}

def load_tfrecords(file):
  return tf.data.TFRecordDataset([str(file)])

def dump_tfrecords(dataset, file):
  writer = tf.data.experimental.TFRecordWriter(str(file))
  writer.write(dataset.map(tf.io.serialize_tensor))

class TFPreprocessor(preprocessor.BatchingPreprocessor):
  enc = None
  finalized_cacheable = False  # Not working yet
  finalized_format = utils.register_cache_format(
    "tfrecords", load_tfrecords, dump_tfrecords, type="custom")

  def __finalize_with_batcher(self, elements, batcher):
    batch_gen = batcher.batch_generator(elements)
    in_enc, out_enc = self.enc
    if not self.has_out:
      out_enc = None

    return make_dataset(
      batch_gen,
      in_enc, self.in_args,
      out_enc, self.out_args,
      batcher.lazy_batching)

  def finalize(self, elements):
    if self.reconfigurable_finalization:
      return lambda **config: self.__finalize_with_batcher(
        elements, self.batcher(config))

    return self.__finalize_with_batcher(elements, self.batcher)

def create_preprocessor(
  type, enc,
  in_encoder=None, in_batcher=None,
  out_encoder=None, out_batcher=None,
  io_encoder=False, io_batcher=False):
  e = enc

  class Preprocessor(TFPreprocessor):
    enc = e
    if in_encoder is not None:
      in_encoder_gen = in_encoder
    if out_encoder is not None or io_encoder:
      out_encoder_gen = out_encoder
    if in_batcher is not None:
      in_batcher_gen = in_batcher
    if out_batcher is not None or io_batcher:
      out_batcher_gen = out_batcher

  preprocessor.register_preprocessor(type, enc, Preprocessor)
  return Preprocessor

def create_graph_preprocessors(name, encoder, batcher, pref_util_batcher):
  # Regression:
  create_preprocessor(
    ("graph", "integer"), (name, "float32"),
    encoder, batcher)
  create_preprocessor(
    ("graph", "float"), (name, "float32"),
    encoder, batcher)
  create_preprocessor(
    ("graph", "vector"), (name, "vec32"),
    encoder, batcher)

  # Classification:
  create_preprocessor(
    ("graph", "binary"), (name, "binary"),
    encoder, batcher)
  create_preprocessor(
    ("graph", "binary"), (name, "multiclass"),
    encoder, batcher,
    cls_enc.MulticlassEncoder)
  create_preprocessor(
    ("graph", "integer"), (name, "multiclass"),
    encoder, batcher,
    cls_enc.MulticlassEncoder)

  # Ranking:
  create_preprocessor(
    ("graph", "integer"), (f"{name}_pref", "binary"),
    encoder, pref_util_batcher,
    io_batcher=True)  # pref_util_batcher processes (graphs, utils) tuples
  create_preprocessor(
    ("graph", "float"), (f"{name}_pref", "binary"),
    encoder, pref_util_batcher,
    io_batcher=True)


create_graph_preprocessors(
  "wl1", wl1_enc.WL1Encoder, wl1_enc.WL1Batcher,
  wl1_enc.WL1UtilityPreferenceBatcher)
create_graph_preprocessors(
  "wl2", wl2_enc.WL2Encoder, wl2_enc.WL2Batcher,
  wl2_enc.WL2UtilityPreferenceBatcher)
