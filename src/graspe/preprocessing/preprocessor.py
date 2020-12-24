import funcy as fy

from graspe.utils import tolerant, select_prefixed_keys
import graspe.preprocessing.transformer as transformer
import graspe.preprocessing.encoder as encoder
import graspe.preprocessing.batcher as batcher

def tolerant_method(f):
  return tolerant(getattr(f, "__func__", f))

class Preprocessor:
  # Most efficient preprocessing allowed by default:
  slice_after_preprocess = True
  preprocessed_cacheable = True
  finalized_cacheable = False

  in_encoder_gen = lambda: encoder.Encoder.identity
  out_encoder_gen = in_encoder_gen

  def __init__(self, in_meta=None, out_meta=None, config=None):
    super().__init__()
    if in_meta is None:
      in_meta = {}
    if config is None:
      config = {}

    self.in_config = select_prefixed_keys(config, "in_", True)
    self.out_config = select_prefixed_keys(config, "out_", True)
    self.in_meta = in_meta
    self.out_meta = out_meta
    self.in_args = fy.merge(self.in_config, in_meta)

    print(self.in_encoder_gen, self.out_encoder_gen)
    in_enc = tolerant_method(self.in_encoder_gen)(**self.in_args)

    if out_meta is None:
      self.encoder = in_enc
      self.out_args = None
      self.has_out = False
    else:
      self.out_args = fy.merge(self.out_config, out_meta)
      out_enc = tolerant_method(self.out_encoder_gen)(**self.out_args)
      self.encoder = transformer.tuple(in_enc, out_enc)
      self.has_out = True

  @property
  def preprocessed_name(self):
    return self.encoder.name

  def preprocess(self, elements):
    return self.encoder.transform(elements)

  def slice(self, elements, indices):
    return self.encoder.slice(elements, indices)

  def finalize(self, elements):
    return elements

  def transform(self, elements, indices=None):
    if indices is not None and not self.slice_after_preprocess:
      elements = self.slice(elements, indices)

    elements = self.preprocess(elements)

    if indices is not None and self.slice_after_preprocess:
      elements = self.slice(elements, indices)

    return self.finalize(elements)

class BatchingPreprocessor(Preprocessor):
  in_batcher_gen = lambda **kwargs: tolerant(batcher.Batcher)(**kwargs)
  out_batcher_gen = in_batcher_gen

  def __init__(self, in_meta=None, out_meta=None, config=None):
    super().__init__(in_meta, out_meta, config)

    in_bat = tolerant_method(self.in_batcher_gen)(**self.in_args)

    if out_meta is None:
      self.batcher = in_bat
    else:
      out_bat = tolerant_method(self.out_batcher_gen)(**self.out_args)
      self.batcher = transformer.tuple(in_bat, out_bat)

  @property
  def finalized_name(self):
    return self.batcher.name

  def finalize(self, elements):
    return self.batcher.transform(elements)


preprocessors = dict()

def register_preprocessor(type, enc, preprocessor):
  pass
