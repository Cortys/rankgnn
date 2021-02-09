import funcy as fy
import numpy as np
import ogb.graphproppred as ogb

import graspe.datasets.loader as loader
import graspe.datasets.provider as provider
import graspe.utils as utils
import graspe.datasets.ogb.utils as ogb_utils

class OGBDatasetLoader(loader.DatasetLoader):
  root_dir = loader.RAW_ROOT / "ogb"
  _ogb_dataset = None

  def __init__(self, name, config):
    super().__init__()
    self.name = name
    self.config = config

  @property
  def ogb_dataset(self):
    if self._ogb_dataset:
      return self._ogb_dataset

    ds = ogb.GraphPropPredDataset(
      name=self.name, root=self.root_dir)
    ds.meta_info["add_inverse_edge"] = "False"  # no edge duplication
    self._ogb_dataset = ds
    return ds

  @property
  def dataset_type(self):
    return ("graph", self.config["type"])

  @property
  def stratifiable(self):
    return loader.is_stratifiable(self.config["type"], self.config)

  def load_dataset(self, only_meta=True):
    ds = self.ogb_dataset
    targets = np.array(ds.labels)
    size = targets.size
    graphs = np.empty(size, dtype="O")
    for i, g in enumerate(ds.graphs):
      graphs[i] = ogb_utils.create_graph_from_ogb(g)
    in_meta = utils.graphs_meta(graphs, labels=False)
    config = self.config.copy()
    if "discrete_node_features" in config:
      in_meta["discrete_node_features"] = config["discrete_node_features"]
      del config["discrete_node_features"]
    if "discrete_edge_features" in config:
      in_meta["discrete_edge_features"] = config["discrete_edge_features"]
      del config["discrete_edge_features"]

    if config["type"] in {"binary", "integer", "float"}:
      targets = np.squeeze(targets)

    return dict(
      elements=(graphs, targets),
      in_meta=in_meta,
      out_meta=config,
      size=size,
      stratify_labels=targets if self.stratifiable else None)

  @property
  def named_splits(self):
    split = self.ogb_dataset.get_idx_split()
    return dict(ogb=dict(
      model_selection=[dict(
        train=split["train"],
        validation=split["valid"])],
      test=split["test"]))

class OGBDatasetProvider(provider.CachingDatasetProvider):
  root_dir = provider.CACHE_ROOT / "ogb"

  def __init__(self, name, config, *args, **kwargs):
    loader = OGBDatasetLoader(name, config)
    self.name = name
    super().__init__(loader, *args, default_split="ogb", **kwargs)

  def _make_named_splits(self):
    return self.loader.named_splits

def ogb_dataset(name, default_preprocess_config=None, **config):
  return fy.func_partial(
    OGBDatasetProvider, name, config,
    default_preprocess_config=default_preprocess_config)
