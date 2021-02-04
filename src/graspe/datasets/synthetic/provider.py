from abc import abstractmethod
import funcy as fy
import numpy as np

import graspe.utils as utils
import graspe.datasets.provider as provider
import graspe.datasets.loader as loader

class SyntheticDatasetLoader(loader.DatasetLoader):
  def __init__(self, generator, config):
    super().__init__()
    self.generator = generator
    self.config = config

  @abstractmethod
  def compute_meta(self, elements): pass

  @abstractmethod
  def compute_size(self, elements): pass

  @abstractmethod
  def compute_stratify_labels(self, elements): pass

  def load_dataset(self, only_meta=True):
    elements = self.generator()
    in_meta, out_meta = self.compute_meta(elements)

    return dict(
      elements=elements,
      in_meta=in_meta, out_meta=out_meta,
      size=self.compute_size(elements),
      stratify_labels=self.compute_stratify_labels(elements))

class SyntheticDatasetProvider(provider.CachingDatasetProvider):
  loaderClass = SyntheticDatasetLoader
  root_dir = provider.CACHE_ROOT / "synthetic"

  def __init__(self, generator, config, *args, **kwargs):
    loader = self.loaderClass(generator, config)
    self.name = generator.__name__

    super().__init__(loader, *args, **kwargs)

  @classmethod
  def register_splitter(cls, name, f=None):
    if f is None:
      return lambda f: cls.register_splitter(name, f)

    @fy.wraps(f)
    def splitter(ds):
      train, val, test = f(ds)
      return dict(
        model_selection=[dict(
          train=np.array(train),
          validation=np.array(val))],
        test=np.array(test))

    if not hasattr(cls, "splitters"):
      cls.splitters = dict()

    cls.splitters[name] = splitter
    return f

  def _make_named_splits(self):
    if not hasattr(self, "splitters"):
      return dict()

    ds = self.dataset
    return {
      name: splitter(ds)
      for name, splitter in self.splitters.items()}

class PresplitSyntheticDatasetProvider(
  provider.CachingDatasetProvider, provider.PresplitDatasetProvider):
  loaderClass = SyntheticDatasetLoader
  root_dir = provider.CACHE_ROOT / "synthetic"
  _generated_data = None

  def __init__(self, generator, config, **kwargs):
    self.generator = generator
    loader_train = self.loaderClass(
      fy.func_partial(self.generate, id="train"), config)
    loader_val = self.loaderClass(
      fy.func_partial(self.generate, id="val"), config)
    loader_test = self.loaderClass(
      fy.func_partial(self.generate, id="test"), config)
    self.name = generator.__name__
    super().__init__(loader_train, loader_val, loader_test, **kwargs)

  def generate(self, id=None):
    if self._generated_data is None:
      self._generated_data = self.generator()

    if id is not None:
      return self._generated_data[id]

    return self._generated_data

class SyntheticGraphEmbedDatasetLoader(SyntheticDatasetLoader):
  @property
  def dataset_type(self):
    return ("graph", self.config["type"])

  @property
  def stratifiable(self):
    return loader.is_stratifiable(self.config["type"], self.config)

  def compute_meta(self, elements):
    gs, ys = elements
    meta_in = utils.graphs_meta(gs)
    meta_out = self.config

    return meta_in, meta_out

  def compute_size(self, elements):
    return len(elements[0])

  def compute_stratify_labels(self, elements):
    return np.array(elements[1])

class SyntheticGraphEmbedDatasetProvider(SyntheticDatasetProvider):
  loaderClass = SyntheticGraphEmbedDatasetLoader

class PresplitSyntheticGraphEmbedDatasetProvider(
  PresplitSyntheticDatasetProvider):
  loaderClass = SyntheticGraphEmbedDatasetLoader

def synthetic_dataset_decorator(cls):
  def dataset_decorator(
    f=None, extends=None,
    **config):
    if f is None:
      return lambda f: dataset_decorator(f, extends, **config)

    if extends is not None:
      g = f
      h = utils.unwrap_method(getattr(extends, "generator", extends))
      config = fy.merge(config, getattr(extends, "config", {}))
      f = fy.compose(g, h)
      f.__name__ = g.__name__

    pc = {"outer_k", "inner_k", "outer_holdout", "inner_holdout", "stratify"}
    c = config
    provider_config = fy.select_keys(lambda k: k in pc, c)
    loader_config = fy.select_keys(lambda k: k not in pc, c)

    class SyntheticProvider(cls):
      generator = f
      config = c

      def __init__(self, *args, **kwargs):
        super().__init__(
          f, loader_config, *args, **fy.merge(provider_config, kwargs))

    SyntheticProvider.__name__ = f.__name__

    return SyntheticProvider

  return dataset_decorator


synthetic_graph_embed_dataset = synthetic_dataset_decorator(
  SyntheticGraphEmbedDatasetProvider)
presplit_synthetic_graph_embed_dataset = synthetic_dataset_decorator(
  PresplitSyntheticGraphEmbedDatasetProvider)
