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

  def stats(self, loaded_dataset):
    gs, ys = loaded_dataset["elements"]

    return dict(
      graphs=utils.graphs_stats(gs),
      targets=utils.statistics(ys),
      size=loaded_dataset["size"],
      in_meta=loaded_dataset["in_meta"],
      out_meta=loaded_dataset["out_meta"]
    )

class TUDatasetProvider(provider.CachingDatasetProvider):
  root_dir = provider.CACHE_ROOT / "tu"

  def __init__(self, name, config, *args, **kwargs):
    loader = TUDatasetLoader(name, config)
    self.name = name
    super().__init__(loader, *args, **kwargs)

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

def tu_dataset(name, **config):
  return fy.func_partial(TUDatasetProvider, name, config)

def presplit_tu_dataset(name, name_train, name_val, name_test, **config):
  return fy.func_partial(
    PresplitTUDatasetProvider,
    name, name_train, name_val, name_test, config)
