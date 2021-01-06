import funcy as fy
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import graspe.utils as utils
import graspe.preprocessing.preprocessor as preproc
import graspe.datasets.loader as loader

CACHE_ROOT = Path("../data/")

class DatasetProvider:
  name = None
  _dataset = None
  _splits = None

  def __init__(
    self, loader: loader.DatasetLoader,
    outer_k=10, inner_k=None,
    holdout_size=0.1,
    stratify=True,
    name_suffix=""):
    self.loader = loader
    assert outer_k is None or outer_k > 1
    self.outer_k = outer_k
    assert inner_k is None or inner_k > 1
    self.inner_k = inner_k
    assert not (outer_k is None or inner_k is None) or holdout_size is not None
    self.holdout_size = holdout_size
    self.stratify = stratify and loader.stratifiable
    self.name_suffix = name_suffix

  @property
  def in_meta(self):
    self._cache_dataset()
    return self._dataset["in_meta"]

  @property
  def out_meta(self):
    self._cache_dataset()
    return self._dataset["out_meta"]

  @property
  def full_name(self):
    return self.name + self.name_suffix

  @property
  def dataset(self):
    self._cache_dataset(only_meta=False)
    return self._dataset["elements"]

  @property
  def dataset_size(self):
    self._cache_dataset()
    return self._dataset["size"]

  @property
  def stratify_labels(self):
    if not self.stratify:
      return None

    self._cache_dataset()
    return self._dataset["stratify_labels"]

  @property
  def dataset_type(self):
    return self.loader.dataset_type

  @property
  def splits(self):
    if self._splits is None:
      self._splits = self._make_splits()

    return self._splits

  def _load_dataset(self, only_meta=True):
    return self.loader.load_dataset(only_meta)

  def _cache_dataset(self, only_meta=True):
    if self._dataset is None or not (only_meta or "elements" in self._dataset):
      self._dataset = self._load_dataset(only_meta)

    return self._dataset

  def __make_holdout_split(self, idxs, strat_labels=None):
    if self.holdout_size == 0:
      return idxs, [], strat_labels
    else:
      train_split, test_split = train_test_split(
        np.arange(idxs.size),
        test_size=self.holdout_size,
        stratify=strat_labels)

      if strat_labels is not None:
        strat_labels = strat_labels[train_split]

      return idxs[train_split], idxs[test_split], strat_labels

  def __make_kfold_splits(self, n, idxs, strat_labels=None):
    if self.stratify:
      kfold = StratifiedKFold(n_splits=n, shuffle=True)
    else:
      kfold = KFold(n_splits=n, shuffle=True)

    for train_split, test_split in kfold.split(idxs, strat_labels):
      yield (
        idxs[train_split], idxs[test_split],
        strat_labels[train_split] if strat_labels is not None else None)

  def __make_model_selection_splits(self, train_o, strat_o=None):
    if self.inner_k is None:
      train_i, val_i, _ = self.__make_holdout_split(
        train_o, strat_o)
      return [dict(
        train=train_i,
        validation=val_i)]
    else:
      return [
        dict(train=train_i, validation=val_i)
        for train_i, val_i, _ in self.__make_kfold_splits(
          self.inner_k, train_o, strat_o)]

  def _make_splits(self):
    all_idxs = np.arange(self.dataset_size)
    strat_labels = self.stratify_labels

    if self.outer_k is None:
      train_o, test_o, strat_o = self.__make_holdout_split(
        all_idxs, strat_labels)
      return [dict(
        test=test_o,
        model_selection=self.__make_model_selection_splits(train_o, strat_o))]
    else:
      return [
        dict(
          test=test_o,
          model_selection=self.__make_model_selection_splits(train_o, strat_o))
        for train_o, test_o, strat_o in self.__make_kfold_splits(
          self.outer_k, all_idxs, strat_labels)]

  def _get_preprocessor(self, enc, config):
    return preproc.find_preprocessor(self.dataset_type, enc)(
      self.in_meta, self.out_meta, config)

  def _preprocess(
    self, pre: preproc.Preprocessor, ds, indices, train_indices,
    index_id, finalize=True):
    return pre.transform(ds, indices, train_indices, finalize)

  def get(
    self, enc=None, config=None,
    indices=None, train_indices=None,
    preprocessor=None,
    shuffle=False,
    index_id=None, finalize=True):
    if preprocessor is None:
      preprocessor = self._get_preprocessor(enc, config)
    ds = self.dataset

    if shuffle:
      if indices is None:
        indices = np.arange(self.dataset_size)
      else:
        indices = np.array(indices)
      np.random.shuffle(indices)
      if index_id is None:
        index_id = ()
      index_id += ("shuffled",)

    return self._preprocess(
      preprocessor, ds, indices, train_indices, index_id, finalize)

  def get_split(
    self, enc=None, config=None, outer_idx=None, inner_idx=None,
    only=None, finalize=True):
    outer_idx = outer_idx or 0
    inner_idx = inner_idx or 0
    assert (outer_idx == 0 or outer_idx < self.outer_k) \
        and (inner_idx == 0 or inner_idx < self.inner_k)
    split = self.splits[outer_idx]
    inner_fold = split["model_selection"][inner_idx]
    test_idxs = split["test"]
    train_idxs = inner_fold["train"]
    val_idxs = inner_fold["validation"]
    pre = self._get_preprocessor(enc, config)
    no_validation = config is not None and config.get("no_validation", False)

    if no_validation:
      train_idxs = np.concatenate([train_idxs, val_idxs])

    if only is None or only == "train":
      train_ds = self.get(
        enc, indices=train_idxs, preprocessor=pre,
        index_id=(outer_idx, inner_idx, "train"), finalize=finalize)
      if only == "train":
        return train_ds

    if only is None or only == "test":
      test_ds = self.get(
        enc, indices=test_idxs, train_indices=train_idxs, preprocessor=pre,
        index_id=(outer_idx, inner_idx, "test"), finalize=finalize)
      if only == "test":
        return test_ds

    if not no_validation:
      val_ds = self.get(
        enc, indices=train_idxs, train_indices=train_idxs, preprocessor=pre,
        index_id=(outer_idx, inner_idx, "val"), finalize=finalize)
      if only is None:
        return train_ds, val_ds, test_ds
      elif only == "val":
        return val_ds
      raise AssertionError(f"Invalid only-selector: {only}.")
    else:
      assert only is None, f"Invalid only-selector: {only}."
      return train_ds, test_ds

  def get_train_split(
    self, enc=None, config=None,
    outer_idx=None, inner_idx=None, finalize=True):
    return self.get_split(enc, config, outer_idx, inner_idx, "train", finalize)

  def get_validation_split(
    self, enc=None, config=None,
    outer_idx=None, inner_idx=None, finalize=True):
    return self.get_split(enc, config, outer_idx, inner_idx, "val", finalize)

  def get_test_split(
    self, enc=None, config=None,
    outer_idx=None, inner_idx=None, finalize=True):
    return self.get_split(enc, config, outer_idx, inner_idx, "test", finalize)

  def stats(self):
    self._cache_dataset(only_meta=False)
    return self.loader.stats(self._dataset)

  def find_compatible_encoding(self, input_encodings, output_encodings):
    compatible_encodings = preproc.find_encodings(self.dataset_type)
    in_set = set(input_encodings)
    out_set = set(output_encodings)

    return fy.filter(
      lambda enc: enc[0] in in_set and enc[1] in out_set,
      compatible_encodings)


class CachingDatasetProvider(DatasetProvider):
  root_dir = CACHE_ROOT

  def __init__(
    self, *args, preprocessed_cache=True, finalize_cache=False, **kwargs):
    super().__init__(*args, **kwargs)
    self.preprocessed_cache = preprocessed_cache
    self.finalize_cache = finalize_cache
    self.data_dir = utils.make_dir(self.root_dir / self.full_name)

  @utils.cached_method("processed")
  def _load_dataset(self, only_meta=True):
    return super()._load_dataset(only_meta=False)  # always load the elements

  @utils.cached_method("processed", suffix="_splits", format="json")
  def _make_splits(self):
    return super()._make_splits()

  def _preprocess(
    self, pre: preproc.Preprocessor, ds, indices, train_indices,
    index_id, finalize=True):
    suffix = "" if index_id is None else "_" + "_".join(
      fy.map(str, index_id))
    preprocessed_cache = pre.preprocessed_cacheable and self.preprocessed_cache
    preprocessed_dir = self.data_dir / pre.preprocessed_name
    if preprocessed_cache:
      utils.make_dir(preprocessed_dir)
    preprocessed_format = pre.preprocessed_format or "pickle"

    def preproc(ds):
      if indices is not None and not pre.slice_after_preprocess:
        ds = pre.slice(ds, indices, train_indices)
        preprocess_file = preprocessed_dir / \
            f"{self.name}{suffix}.{preprocessed_format}"
      else:
        preprocess_file = preprocessed_dir / \
            f"{self.name}.{preprocessed_format}"

      if preprocessed_cache:
        ds = utils.cache(lambda: pre.preprocess(ds), preprocess_file)
      else:
        ds = pre.preprocess(ds)

      if indices is not None and pre.slice_after_preprocess:
        ds = pre.slice(ds, indices, train_indices)

      return pre.finalize(ds) if finalize else ds

    if finalize and pre.finalized_cacheable and self.finalize_cache:
      finalized_dir = utils.make_dir(
        preprocessed_dir / f"final-{pre.finalized_name}")
      finalized_format = pre.finalized_format or "pickle"
      finalized_file = finalized_dir / \
          f"{self.name}{suffix}.{finalized_format}"
      return utils.cache(
        lambda: preproc(ds), finalized_file, finalized_format)
    else:
      return preproc(ds)

class PresplitDatasetProvider(DatasetProvider):
  def __init__(
    self, loader_train, loader_val=None, loader_test=None, name_suffix=""):
    super().__init__(
      loader.PresplitDatasetLoader(loader_train, loader_val, loader_test),
      outer_k=None,
      name_suffix=name_suffix)

  def _make_splits(self):
    raise Exception("This dataset has fixed train, val and test splits.")

  def get(self, *args, **kwargs):
    raise Exception("Presplit datasets cannot be resliced.")

  def get_split(
    self, enc=None, config=None, outer_idx=None, inner_idx=None,
    only=None, finalize=True):
    ds = self.dataset
    pre = self._get_preprocessor(enc, config)
    if only is None or only == "train":
      train_ds = self._preprocess(
        pre, ds["train"], None, None, ("train",), finalize)
      if only == "train":
        return train_ds

    res = (train_ds,)

    if self.loader.val and (only is None or only == "val"):
      val_ds = self._preprocess(
        pre, ds["val"], None, None, ("val",), finalize)
      if only == "val":
        return val_ds
      res += (val_ds,)
    if self.loader.test and (only is None or only == "test"):
      test_ds = self._preprocess(
        pre, ds["test"], None, None, ("test",), finalize)
      if only == "test":
        return test_ds
      res += (test_ds,)

    assert only is None, f"Invalid only-selector: {only}."
    return res
