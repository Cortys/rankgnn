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
    elements = utils.unwrap_method(self.generator)()
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
      self._generated_data = utils.unwrap_method(self.generator)()

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

  def stats(self, loaded_dataset):
    gs, ys = loaded_dataset["elements"]

    return dict(
      graphs=utils.graphs_stats(gs),
      targets=utils.statistics(ys),
      size=loaded_dataset["size"],
      in_meta=loaded_dataset["in_meta"],
      out_meta=loaded_dataset["out_meta"]
    )

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
      h = getattr(extends, "__original__", extends)
      config = fy.merge(config, getattr(extends, "__config__", {}))
      f = fy.compose(g, h)
      f.__name__ = g.__name__

    pc = {"outer_k", "inner_k", "outer_holdout", "inner_holdout", "stratify"}
    provider_config = fy.select_keys(lambda k: k in pc, config)
    loader_config = fy.select_keys(lambda k: k not in pc, config)
    res = fy.func_partial(cls, f, loader_config, **provider_config)
    res.__original__ = f
    res.__config__ = config
    return res

  return dataset_decorator


synthetic_graph_embed_dataset = synthetic_dataset_decorator(
  SyntheticGraphEmbedDatasetProvider)
presplit_synthetic_graph_embed_dataset = synthetic_dataset_decorator(
  PresplitSyntheticGraphEmbedDatasetProvider)
