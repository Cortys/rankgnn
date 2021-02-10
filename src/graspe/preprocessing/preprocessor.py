import funcy as fy
from collections import defaultdict

from graspe.utils import memoize, tolerant, \
  tolerant_method, select_prefixed_keys
import graspe.preprocessing.transformer as transformer
import graspe.preprocessing.encoder as encoder
import graspe.preprocessing.batcher as batcher

def add_io_prefixes(names, suffix=""):
  if names is None or len(names) == 0:
    return ()
  elif len(names) == 2:
    return ("in_" + names[0] + suffix, "out_" + names[1] + suffix)
  raise AssertionError("Unsupported names list.")

class Preprocessor:
  # Most efficient preprocessing allowed by default:
  slice_after_preprocess = True
  preprocessed_cacheable = True
  finalized_cacheable = False
  preprocessed_format = None
  finalized_format = None

  in_encoder_gen = lambda: encoder.Encoder.identity
  out_encoder_gen = in_encoder_gen

  def __init__(
    self, in_meta=None, out_meta=None, config=None,
    reconfigurable_finalization=False):
    super().__init__()
    if in_meta is None:
      in_meta = {}
    if config is None:
      config = {}

    self.config = config
    self.reconfigurable_finalization = reconfigurable_finalization

    if self.reconfigurable_finalization:
      self.finalized_cacheable = False

    self.in_config = select_prefixed_keys(config, "in_", True)
    self.out_config = select_prefixed_keys(config, "out_", True)
    self.in_meta = in_meta
    self.out_meta = out_meta
    self.in_args = fy.merge(self.in_config, in_meta)

    in_enc_cls = tolerant_method(self.in_encoder_gen)
    orthogonal_preprocess = False

    if out_meta is None:
      self.encoder = in_enc_cls(**self.in_args)
      self.out_args = None
      self.has_out = False
    else:
      self.out_args = fy.merge(self.out_config, out_meta)
      self.has_out = True
      if self.out_encoder_gen is None:
        self.encoder = in_enc_cls(self.in_args, self.out_args, **config)
      else:
        in_enc = in_enc_cls(**self.in_args)
        out_enc = tolerant_method(self.out_encoder_gen)(**self.out_args)
        self.encoder = transformer.tuple(in_enc, out_enc)
        orthogonal_preprocess = True

    self.orthogonal_preprocess = orthogonal_preprocess

    assert self.encoder.can_slice_raw or self.encoder.can_slice_encoded, \
        "Invalid encoder."

    if orthogonal_preprocess:
      self.uses_train_metadata = [
        t.uses_train_metadata for t in self.encoder.transformers]
    else:
      self.uses_train_metadata = self.encoder.uses_train_metadata

    if self.slice_after_preprocess:
      if orthogonal_preprocess:
        self.slice_after_preprocess = [
          t.can_slice_encoded for t in self.encoder.transformers]
      else:
        self.slice_after_preprocess = self.encoder.can_slice_encoded
    else:
      if orthogonal_preprocess:
        self.slice_after_preprocess = [
          t.can_slice_raw for t in self.encoder.transformers]
      else:
        self.slice_after_preprocess = self.encoder.can_slice_raw

  @property
  def preprocessed_name(self):
    return self.encoder.name

  @property
  def preprocessed_names(self):
    return add_io_prefixes(getattr(self.encoder, "names", None))

  @property
  def finalized_name(self):
    if self.slice_after_preprocess and (
      not self.orthogonal_preprocess or fy.any(self.slice_after_preprocess)):
      return self.encoder.name + "_sliced"
    raise Exception("This preprocessor has no post-encoding processing stage.")

  @property
  def finalized_names(self):
    if self.slice_after_preprocess:
      return add_io_prefixes(getattr(self.encoder, "names", None), "_sliced")
    raise Exception("This preprocessor has no post-encoding processing stage.")

  def preprocess(self, elements, only=None, train_metadata=None):
    if only is not None:
      assert self.orthogonal_preprocess, "Orthogonality required."
      if only == "in":
        only = 0
      elif only == "out":
        only = 1
      return self.encoder.transformers[only].transform(
        elements, train_metadata)

    return self.encoder.transform(elements, train_metadata)

  def slice_raw(self, elements, indices, only=None):
    if indices is None or indices is False:
      return elements

    enc = self.encoder
    if only is not None:
      assert self.orthogonal_preprocess, "Orthogonality required."
      if only == "in":
        only = 0
      elif only == "out":
        only = 1
      enc = enc.transformers[only]

    return enc.slice_raw(elements, indices)

  def slice_encoded(self, elements, indices, train_indices=None, only=None):
    if indices is None:
      return elements

    enc = self.encoder
    if only is not None:
      assert self.orthogonal_preprocess, "Orthogonality required."
      if only == "in":
        only = 0
      elif only == "out":
        only = 1
      enc = enc.transformers[only]

    return enc.slice_encoded(elements, indices, train_indices)

  def produce_train_metadata(self, elements, only=None):
    enc = self.encoder
    if only is not None:
      assert self.orthogonal_preprocess, "Orthogonality required."
      if only == "in":
        only = 0
      elif only == "out":
        only = 1
      enc = enc.transformers[only]

    return enc.produce_train_metadata(elements)

  def finalize(self, elements):
    if self.reconfigurable_finalization:
      return lambda **kwargs: elements  # no reconfiguration possible here.

    return elements

  def hooked_transform(
    self, get_elements, indices=None, train_indices=None,
    get_train_elements=None, finalize=True,
    preprocess_hook=None, train_metadata_hook=None):
    f = memoize(get_elements)
    if get_train_elements is None:
      ft = f
    else:
      ft = memoize(get_train_elements)
    if preprocess_hook is None:
      preprocess_hook = lambda f, i, slice: f(i, slice)
    if train_metadata_hook is None:
      train_metadata_hook = lambda f, i: f(i)

    if self.orthogonal_preprocess:
      g = lambda i: lambda elements: self.slice_encoded(
        elements, indices, train_indices, only=i)
      ds = tuple(
        (g(i) if slice_after else fy.identity)(preprocess_hook(
          lambda i, slice: self.preprocess(
            self.slice_raw(f()[i], indices, only=i) if slice else f()[i],
            train_metadata=train_metadata_hook(
              lambda i: self.produce_train_metadata(
                self.slice_raw(ft()[i], train_indices, only=i), only=i),
              i) if use_meta and slice else None,
            only=i),
          i, not slice_after))
        for i, (slice_after, use_meta) in enumerate(
          zip(self.slice_after_preprocess, self.uses_train_metadata)))
    else:
      ds = preprocess_hook(
        lambda i, slice: self.preprocess(
          self.slice_raw(f(), indices) if slice else f(),
          train_metadata=train_metadata_hook(
            lambda i: self.produce_train_metadata(
              self.slice_raw(ft(), train_indices))
          ) if self.uses_train_metadata else None),
        None, not self.slice_after_preprocess)

      if self.slice_after_preprocess:
        ds = self.slice_encoded(ds, indices, train_indices)

    return self.finalize(ds) if finalize else ds

  def transform(
    self, elements, indices=None, train_indices=None,
    train_elements=None, finalize=True):
    return self.hooked_transform(
      lambda: elements, indices, train_indices,
      lambda: train_elements or elements, finalize)

class BatchingPreprocessor(Preprocessor):
  in_batcher_gen = lambda **kwargs: tolerant(batcher.Batcher)(**kwargs)
  out_batcher_gen = in_batcher_gen

  def __init__(
    self, in_meta=None, out_meta=None, config=None,
    reconfigurable_finalization=False):
    super().__init__(in_meta, out_meta, config, reconfigurable_finalization)

    in_bat_cls = tolerant_method(self.in_batcher_gen)

    if out_meta is None:
      if self.reconfigurable_finalization:
        self.batcher = lambda config: in_bat_cls(
          **fy.merge(self.in_args, config))
      else:
        self.batcher = in_bat_cls(**self.in_args)
    else:
      if self.out_batcher_gen is None:
        if self.reconfigurable_finalization:
          self.batcher = lambda config: in_bat_cls(
            self.in_args, self.out_args, **fy.merge(self.config, config))
        else:
          self.batcher = in_bat_cls(self.in_args, self.out_args, **self.config)
      else:
        if self.reconfigurable_finalization:
          def bat_gen(config):
            in_bat = in_bat_cls(**fy.merge(self.in_args, config))
            out_bat = tolerant_method(self.out_batcher_gen)(
              **fy.merge(self.out_args, config))
            return transformer.tuple(in_bat, out_bat)

          self.batcher = bat_gen
        else:
          in_bat = in_bat_cls(**self.in_args)
          out_bat = tolerant_method(self.out_batcher_gen)(**self.out_args)
          self.batcher = transformer.tuple(in_bat, out_bat)

  @property
  def finalized_name(self):
    return self.batcher.name

  @property
  def finalized_names(self):
    return add_io_prefixes(getattr(self.batcher, "names", None))

  def finalize(self, elements):
    if self.reconfigurable_finalization:
      bat_gen = tolerant_method(self.batcher)

      def reconfigurable_finalizer(**config):
        return bat_gen(**config).transform(elements)

      return reconfigurable_finalizer

    return self.batcher.transform(elements)

class DefaultPreprocessor(Preprocessor):
  preprocessed_cacheable = False
  in_encoder_gen = lambda: encoder.ObjectEncoder.identity
  out_encoder_gen = in_encoder_gen


preprocessors = defaultdict(dict)

def register_preprocessor(type, enc, preprocessor):
  in_type = type[0]
  preprocessors[type][enc] = preprocessor
  preprocessors[in_type][enc] = preprocessor

  return preprocessor

def find_encodings(type):
  return preprocessors[type].keys()

def find_preprocessor(type, enc):
  if enc is None:
    return DefaultPreprocessor

  return preprocessors[type][enc]
