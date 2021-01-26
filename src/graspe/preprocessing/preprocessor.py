import funcy as fy
from collections import defaultdict

from graspe.utils import tolerant, tolerant_method, select_prefixed_keys
import graspe.preprocessing.transformer as transformer
import graspe.preprocessing.encoder as encoder
import graspe.preprocessing.batcher as batcher

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

    if out_meta is None:
      self.encoder = in_enc_cls(**self.in_args)
      self.out_args = None
      self.has_out = False
    else:
      self.out_args = fy.merge(self.out_config, out_meta)
      if self.out_encoder_gen is None:
        self.encoder = in_enc_cls(self.in_args, self.out_args, **config)
      else:
        in_enc = in_enc_cls(**self.in_args)
        out_enc = tolerant_method(self.out_encoder_gen)(**self.out_args)
        self.encoder = transformer.tuple(in_enc, out_enc)
        self.has_out = True

  @property
  def preprocessed_name(self):
    return self.encoder.name

  @property
  def finalized_name(self):
    if self.slice_after_preprocess:
      return self.encoder.name + "_sliced"
    raise Exception("This preprocessor has no post-encoding processing stage.")

  def preprocess(self, elements):
    return self.encoder.transform(elements)

  def slice(self, elements, indices, train_indices=None):
    if indices is None:
      return elements

    return self.encoder.slice(elements, indices, train_indices)

  def finalize(self, elements):
    if self.reconfigurable_finalization:
      return lambda **kwargs: elements  # no reconfiguration possible here.

    return elements

  def transform(
    self, elements, indices=None, train_indices=None, finalize=True):
    if indices is not None and not self.slice_after_preprocess:
      elements = self.slice(elements, indices, train_indices)

    elements = self.preprocess(elements)

    if indices is not None and self.slice_after_preprocess:
      elements = self.slice(elements, indices, train_indices)

    return self.finalize(elements) if finalize else elements

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
