from abc import abstractmethod

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

class SyntheticDatasetProvider(provider.DatasetProvider):
  loaderClass = SyntheticDatasetLoader

  def __init__(self, generator, config, *args, **kwargs):
    loader = self.loaderClass(generator, config)
    self.name = generator.__name__
    super().__init__(loader, *args, **kwargs)

class SyntheticGraphEmbedDatasetLoader(SyntheticDatasetLoader):
  @property
  def dataset_type(self):
    return ("graph", self.config["type"])

  @property
  def stratifiable(self):
    return self.config["type"] in {"binary", "multiclass"}

  def compute_meta(self, elements):
    gs, ys = elements
    meta_in = utils.graphs_meta(gs)
    meta_out = self.config

    return meta_in, meta_out

  def compute_size(self, elements):
    return len(elements[0])

  def compute_stratify_labels(self, elements):
    return elements[1]

class SyntheticGraphEmbedDatasetProvider(SyntheticDatasetProvider):
  loaderClass = SyntheticGraphEmbedDatasetLoader

def synthetic_graph_embed_dataset(cached=True, **config):
  def dataset_decorator(f):
    if cached:
      class Provider(
        SyntheticGraphEmbedDatasetProvider, provider.CachingDatasetProvider):
        root_dir = provider.CACHE_ROOT / "synthetic"
    else:
      Provider = SyntheticGraphEmbedDatasetProvider

    res = lambda *args, **kwargs: Provider(f, config, *args, **kwargs)
    res.original_function = f

    return res

  return dataset_decorator
