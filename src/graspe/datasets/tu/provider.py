import io
import zipfile
import requests
import funcy as fy
import numpy as np

import graspe.utils as utils
import graspe.datasets.loader as loader
import graspe.datasets.provider as provider
import graspe.datasets.tu.utils as tu_utils

class TUDatasetLoader(loader.DatasetLoader):
  root_dir = loader.RAW_ROOT / "tu"

  def __init__(self, name, config):
    super().__init__()
    self.name = name
    self.config = config
    self.raw_dir = self.root_dir / name

  @property
  def dataset_type(self):
    return ("graph", self.config["type"])

  @property
  def stratifiable(self):
    return loader.is_stratifiable(self.config["type"], self.config)

  @property
  def download_url(self):
    return f"https://www.chrsmrrs.com/graphkerneldatasets/{self.name}.zip"

  def _download_raw(self):
    if self.raw_dir.exists():
      return

    url = self.download_url
    response = requests.get(url)
    stream = io.BytesIO(response.content)
    with zipfile.ZipFile(stream) as z:
      for fname in z.namelist():
        z.extract(fname, self.root_dir)

  def _parse_tu_data(self):
    self._download_raw()
    return tu_utils.parse_tu_data(
      self.name, self.root_dir)

  def load_dataset(self, only_meta=True):
    graphs_data, in_meta = self._parse_tu_data()
    targets = graphs_data.pop("graph_targets")
    graphs, out_targets = [], []

    for i, target in enumerate(targets, 1):
      graph_data = {k: v[i] for k, v in graphs_data.items()}
      g = tu_utils.create_graph_from_tu_data(graph_data)
      if g.order() > 1 and g.size() > 0:
        graphs.append(g)
        out_targets.append(target)

    graphs_a = utils.obj_array(graphs)
    out_targets = np.array(out_targets)

    return dict(
      elements=(graphs_a, out_targets),
      in_meta=in_meta, out_meta=self.config,
      size=len(out_targets),
      stratify_labels=out_targets if self.stratifiable else None)

class TUDatasetProvider(provider.CachingDatasetProvider):
  root_dir = provider.CACHE_ROOT / "tu"

  def __init__(self, name, config, default_split=None, *args, **kwargs):
    loader = TUDatasetLoader(name, config)
    self.name = name
    self._tu_split = default_split
    ds = "tu" if default_split is not None else 0
    super().__init__(loader, *args, default_split=ds, **kwargs)

  def _make_named_splits(self):
    split = self._tu_split
    if split is None:
      return dict()

    return dict(tu=dict(
      model_selection=[dict(
        train=split[0],
        validation=split[1])],
      test=split[2]))

class PresplitTUDatasetProvider(
  provider.CachingDatasetProvider, provider.PresplitDatasetProvider):
  root_dir = provider.CACHE_ROOT / "tu"

  def __init__(self, name, name_train, name_val, name_test, config, **kwargs):
    loader_train = TUDatasetLoader(name_train, config)
    loader_val = TUDatasetLoader(name_val, config)
    loader_test = TUDatasetLoader(name_test, config)
    self.name = name
    super().__init__(
      loader_train=loader_train,
      loader_val=loader_val,
      loader_test=loader_test,
      **kwargs)

def tu_dataset(
  name, default_split=None, default_preprocess_config=None, **config):
  return fy.func_partial(
    TUDatasetProvider, name, config,
    default_split=default_split,
    default_preprocess_config=default_preprocess_config)

def presplit_tu_dataset(
  name, name_train, name_val, name_test,
  default_preprocess_config=None, **config):
  return fy.func_partial(
    PresplitTUDatasetProvider, name, name_train, name_val, name_test, config,
    default_preprocess_config=default_preprocess_config)
