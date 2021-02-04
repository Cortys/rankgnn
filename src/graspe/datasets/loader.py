from abc import ABC, abstractmethod
from pathlib import Path

RAW_ROOT = Path("../raw/")

def is_stratifiable(out_type, out_meta):
  return type == "binary" \
      or (type == "integer" and out_meta.get("max", None) is not None)

class DatasetLoader(ABC):
  dataset_type = None
  stratifiable = False  # Should be set to True for classification datasets

  @abstractmethod
  def load_dataset(self, only_meta: bool): pass

class PresplitDatasetLoader(DatasetLoader):
  def __init__(
    self,
    train: DatasetLoader,
    val: DatasetLoader = None,
    test: DatasetLoader = None):
    self.train = train
    self.val = val
    self.test = test
    self.dataset_type = self.train.dataset_type
    if val:
      assert self.dataset_type == val.dataset_type
    if test:
      assert self.dataset_type == test.dataset_type

  def load_train_dataset(self, only_meta=True):
    return self.train.load_dataset(only_meta)

  def load_validation_dataset(self, only_meta=True):
    assert self.val is not None
    return self.val.load_dataset(only_meta)

  def load_test_dataset(self, only_meta=True):
    assert self.test is not None
    return self.test.load_dataset(only_meta)

  def load_dataset(self, only_meta=True):
    res = dict(train=self.train.load_dataset(only_meta))
    if self.val:
      res["val"] = self.val.load_dataset(only_meta)
    if self.test:
      res["test"] = self.test.load_dataset(only_meta)

    return res
