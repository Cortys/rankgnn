from abc import ABC, abstractmethod

class DatasetLoader(ABC):
  dataset_type = None
  stratifiable = False  # Should be set to True for classification datasets

  @abstractmethod
  def load_dataset(self, only_meta: bool): pass

  def stats(self, loaded_dataset=None): return

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

  def load_dataset(self, only_meta=True):
    res = dict(train=self.train.load_dataset(only_meta))
    if self.val:
      res["val"] = self.val.load_dataset(only_meta)
    if self.test:
      res["test"] = self.test.load_dataset(only_meta)

    return res

  def stats(self, loaded_dataset=None):
    return dict(
      train=self.train.stats(loaded_dataset["train"]),
      val=self.val.stats(loaded_dataset["val"]) if self.val else None,
      test=self.test.stats(loaded_dataset["test"]) if self.test else None)