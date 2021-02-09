import funcy as fy
import numpy as np
from pathlib import Path

import graspe.utils as utils
import graspe.stats as stats
import graspe.preprocessing.utils as preproc_utils
import graspe.preprocessing.preprocessor as preproc
import graspe.datasets.loader as loader

ldr = loader
CACHE_ROOT = Path("../data/")

class DatasetProvider:
  name = None
  loader: ldr.DatasetLoader = None
  _dataset = None
  _splits = None
  _named_splits = None

  def __init__(
    self, loader: ldr.DatasetLoader,
    outer_k=10, inner_k=None,
    inner_holdout=0.1,
    outer_holdout=0.1,
    default_split=0,
    default_preprocess_config=None,
    stratify=True,
    name_suffix="",
    in_memory_cache=True):
    self.loader = loader
    assert outer_k is None or outer_k > 1
    self.outer_k = outer_k
    assert inner_k is None or inner_k > 1
    self.inner_k = inner_k
    assert outer_k is not None or outer_holdout is not None
    assert inner_k is not None or inner_holdout is not None
    self.outer_holdout = outer_holdout
    self.inner_holdout = inner_holdout
    self.default_split = default_split
    self.default_preprocess_config = default_preprocess_config
    self.stratify = stratify and loader.stratifiable
    self.name_suffix = name_suffix
    self.in_memory_cache = in_memory_cache

  @property
  def in_meta(self):
    return self._cache_dataset(only_meta=True)["in_meta"]

  @property
  def out_meta(self):
    return self._cache_dataset(only_meta=True)["out_meta"]

  @property
  def full_name(self):
    return self.name + self.name_suffix

  @property
  def dataset(self):
    return self._cache_dataset(only_meta=False)["elements"]

  @property
  def dataset_size(self):
    return self._cache_dataset(only_meta=True)["size"]

  @property
  def stratify_labels(self):
    if not self.stratify:
      return None

    return self._cache_dataset(only_meta=False)["stratify_labels"]

  @property
  def dataset_type(self):
    return self.loader.dataset_type

  @property
  def stats(self):
    return self._get_stats()

  @property
  def splits(self):
    if self._splits is None:
      self._splits = self._make_splits()

    return self._splits

  @property
  def named_splits(self):
    if self._named_splits is None:
      self._named_splits = self._get_named_splits()

    return self._named_splits

  def _load_dataset(self, only_meta=False, id=None):
    return self.loader.load_dataset(only_meta)

  def _cache_dataset(self, only_meta=False, id=None):
    if not self.in_memory_cache:
      return self._load_dataset(only_meta, id)

    if self._dataset is None or not (only_meta or "elements" in self._dataset):
      self._dataset = self._load_dataset(only_meta, id)

    return self._dataset

  def unload_dataset(self):
    self._dataset = None

  def __make_model_selection_splits(self, train_o, strat_o=None):
    if self.inner_k is None:
      train_i, val_i, _ = preproc_utils.make_holdout_split(
        self.inner_holdout, train_o, strat_o)
      return [dict(
        train=train_i,
        validation=val_i)]
    else:
      return [
        dict(train=train_i, validation=val_i)
        for train_i, val_i, _ in preproc_utils.make_kfold_splits(
          self.inner_k, train_o, strat_o)]

  def _make_splits(self):
    all_idxs = np.arange(self.dataset_size)
    strat_labels = self.stratify_labels

    if self.outer_k is None:
      train_o, test_o, strat_o = preproc_utils.make_holdout_split(
        self.outer_holdout, all_idxs, strat_labels)
      return [dict(
        test=test_o,
        model_selection=self.__make_model_selection_splits(train_o, strat_o))]
    else:
      return [
        dict(
          test=test_o,
          model_selection=self.__make_model_selection_splits(train_o, strat_o))
        for train_o, test_o, strat_o in preproc_utils.make_kfold_splits(
          self.outer_k, all_idxs, strat_labels)]

  def _make_named_splits(self):
    return dict()

  def _get_named_splits(self):
    return self._make_named_splits()

  def _get_preprocessor(self, enc, config, reconfigurable_finalization=False):
    if self.default_preprocess_config is not None:
      if config is None:
        config = self.default_preprocess_config
      else:
        config = fy.merge(self.default_preprocess_config, config)

    return preproc.find_preprocessor(self.dataset_type, enc)(
      self.in_meta, self.out_meta, config, reconfigurable_finalization)

  def _preprocess(
    self, pre: preproc.Preprocessor, ds_get, indices, train_indices,
    index_id=None, finalize=True, ds_id=None):
    return pre.transform(ds_get(), indices, train_indices, finalize)

  def get(
    self, enc=None, config=None,
    indices=None, train_indices=None,
    preprocessor=None,
    shuffle=False,
    index_id=None, finalize=True,
    reconfigurable_finalization=False):
    if preprocessor is None:
      preprocessor = self._get_preprocessor(
        enc, config, reconfigurable_finalization)

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
      preprocessor, lambda: self.dataset,
      indices, train_indices, index_id, finalize)

  def get_split_indices(self, outer_idx=None, inner_idx=None, relative=False):
    outer_idx = outer_idx or self.default_split
    inner_idx = inner_idx or 0

    if isinstance(outer_idx, str):
      split = self.named_splits[outer_idx]
    else:
      assert (outer_idx == 0 or outer_idx < self.outer_k) \
          and (inner_idx == 0 or inner_idx < self.inner_k)
      split = self.splits[outer_idx]

    inner_fold = split["model_selection"][inner_idx]
    test_idxs = split["test"]
    train_idxs = inner_fold["train"]
    val_idxs = inner_fold["validation"]

    if relative:
      return (
        np.arange(train_idxs.size),
        np.arange(val_idxs.size),
        np.arange(test_idxs.size))

    return train_idxs, val_idxs, test_idxs

  def get_train_split_indices(
    self, outer_idx=None, inner_idx=None, relative=False):
    return self.get_split_indices(outer_idx, inner_idx, relative)[0]

  def get_validation_split_indices(
    self, outer_idx=None, inner_idx=None, relative=False):
    return self.get_split_indices(outer_idx, inner_idx, relative)[1]

  def get_test_split_indices(
    self, outer_idx=None, inner_idx=None, relative=False):
    return self.get_split_indices(outer_idx, inner_idx, relative)[2]

  def get_split(
    self, enc=None, config=None, outer_idx=None, inner_idx=None,
    only=None, finalize=True, reconfigurable_finalization=False,
    indices=None):
    assert indices is None or only is not None, \
        "Index subsets can only be provided if a single split is requested."
    outer_idx = outer_idx or self.default_split
    inner_idx = inner_idx or 0
    train_idxs, val_idxs, test_idxs = self.get_split_indices(
      outer_idx, inner_idx)
    pre = self._get_preprocessor(enc, config, reconfigurable_finalization)
    no_validation = config is not None and config.get("no_validation", False)

    if no_validation:
      train_idxs = np.concatenate([train_idxs, val_idxs])

    if only is None or only == "train":
      if indices is not None:
        train_idxs = train_idxs[indices]

      train_ds = self.get(
        enc, indices=train_idxs, preprocessor=pre,
        index_id=(outer_idx, inner_idx, "train"), finalize=finalize)
      if only == "train":
        return train_ds

    if only is None or only == "test":
      if indices is not None:
        test_idxs = test_idxs[indices]

      test_ds = self.get(
        enc, indices=test_idxs, train_indices=train_idxs, preprocessor=pre,
        index_id=(outer_idx, inner_idx, "test"), finalize=finalize)
      if only == "test":
        return test_ds

    if not no_validation:
      if indices is not None:
        val_idxs = val_idxs[indices]

      val_ds = self.get(
        enc, indices=val_idxs, train_indices=train_idxs, preprocessor=pre,
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
    outer_idx=None, inner_idx=None, finalize=True,
    reconfigurable_finalization=False, indices=None):
    return self.get_split(
      enc, config, outer_idx, inner_idx, "train", finalize,
      reconfigurable_finalization, indices)

  def get_validation_split(
    self, enc=None, config=None,
    outer_idx=None, inner_idx=None, finalize=True,
    reconfigurable_finalization=False, indices=None):
    return self.get_split(
      enc, config, outer_idx, inner_idx, "val", finalize,
      reconfigurable_finalization, indices)

  def get_test_split(
    self, enc=None, config=None,
    outer_idx=None, inner_idx=None, finalize=True,
    reconfigurable_finalization=False, indices=None):
    return self.get_split(
      enc, config, outer_idx, inner_idx, "test", finalize,
      reconfigurable_finalization, indices)

  def _compute_stats(self):
    t = self.dataset_type
    named_splits = self.named_splits

    if named_splits is None or len(named_splits) == 0:
      named_splits = None

    res = stats.find_stat_computer(t)(
      self.dataset, named_splits)

    if isinstance(t, tuple):
      return {"in": res[0], "out": res[1]}
    else:
      return {"in": res}

  def _get_stats(self):
    return dict(
      type=self.dataset_type,
      **self._compute_stats(),
      size=self.dataset_size,
      in_meta=self.in_meta,
      out_meta=self.out_meta)

  def find_compatible_encodings(
    self, input_encodings, output_encodings=None, family=None):
    if family is None and hasattr(input_encodings, "family"):
      family = input_encodings.family

    if output_encodings is None \
        and hasattr(input_encodings, "input_encodings") \
        and hasattr(input_encodings, "output_encodings"):
      output_encodings = input_encodings.output_encodings
      input_encodings = input_encodings.input_encodings

    compatible_encodings = preproc.find_encodings(self.dataset_type)
    in_set = set(input_encodings)
    out_set = set(output_encodings)

    return fy.filter(
      lambda enc: (
        enc[0] in in_set and enc[1] in out_set
        and (family is None or len(enc) < 3 or enc[2] == family)),
      compatible_encodings)


class CachingDatasetProvider(DatasetProvider):
  root_dir = CACHE_ROOT

  def __init__(
    self, *args, load_cache=True, preprocessed_cache=True,
    finalize_cache=False,
    cache=True, **kwargs):
    super().__init__(*args, **kwargs)
    self.load_cache = cache and load_cache
    self.preprocessed_cache = cache and preprocessed_cache
    self.finalize_cache = cache and finalize_cache
    self.data_dir = utils.make_dir(self.root_dir / self.full_name)

  def _load_dataset(self, only_meta=False, id=None):
    if not self.load_cache:
      return super()._load_dataset(only_meta, id)

    dir = utils.make_dir(self.data_dir / "processed")
    suffix = f"_{id}" if id is not None else ""
    meta_cache_file = dir / f"{self.name}{suffix}_meta.json"
    data_cache_file = dir / f"{self.name}{suffix}.pickle"
    checked_file = meta_cache_file if only_meta else data_cache_file
    checked_format = "json" if only_meta else "pickle"

    if checked_file.exists():
      return utils.cache_read(checked_file, checked_format)

    ds = super()._load_dataset(only_meta, id)

    if "elements" in ds:
      utils.cache_write(data_cache_file, ds, "pickle")
      if not meta_cache_file.exists():
        ds_meta = ds.copy()
        del ds_meta["elements"]
        if "stratify_labels" in ds:
          del ds_meta["stratify_labels"]
        utils.cache_write(meta_cache_file, ds_meta, "pretty_json")
    elif not meta_cache_file.exists():
      utils.cache_write(meta_cache_file, ds, "pretty_json")

    return ds

  @utils.cached_method("processed", suffix="_splits", format="json")
  def _make_splits(self):
    return super()._make_splits()

  @utils.cached_method("processed", suffix="_named_splits", format="json")
  def _get_named_splits(self):
    return super()._get_named_splits()

  @utils.cached_method(suffix="_stats", format="pretty_json")
  def _get_stats(self):
    return super()._get_stats()

  def _preprocess(
    self, pre: preproc.Preprocessor, ds_get, indices, train_indices,
    index_id=None, finalize=True, ds_id=None):
    ds_suffix = "" if ds_id is None else "_" + "_".join(fy.map(str, ds_id))
    idx_suffix = "" if index_id is None else "_" + "_".join(
      fy.map(str, index_id))
    preprocessed_cache = pre.preprocessed_cacheable and self.preprocessed_cache
    preprocessed_dir = self.data_dir / pre.preprocessed_name
    orthogonal_preprocess = pre.orthogonal_preprocess
    preprocessed_dirs = [self.data_dir / d for d in pre.preprocessed_names]
    if preprocessed_cache:
      if orthogonal_preprocess:
        for d in preprocessed_dirs:
          utils.make_dir(d)
      else:
        utils.make_dir(preprocessed_dir)
    preprocessed_format = pre.preprocessed_format or "pickle"

    def preproc():
      f = ds_get
      if indices is not None and not pre.slice_after_preprocess:
        f = lambda: pre.slice(ds_get(), indices, train_indices)
        pre_fname = f"{self.name}{ds_suffix}{idx_suffix}.{preprocessed_format}"
      else:
        pre_fname = f"{self.name}{ds_suffix}.{preprocessed_format}"

      if preprocessed_cache:
        if orthogonal_preprocess:
          f = utils.memoize(f)
          ds = tuple(
            utils.cache(lambda: pre.preprocess(f()[i], only=i), d / pre_fname)
            for i, d in enumerate(preprocessed_dirs))
        else:
          preprocess_file = preprocessed_dir / pre_fname
          ds = utils.cache(lambda: pre.preprocess(f()), preprocess_file)
      else:
        ds = pre.preprocess(f())

      if indices is not None and pre.slice_after_preprocess:
        ds = pre.slice(ds, indices, train_indices)

      return pre.finalize(ds) if finalize else ds

    if finalize and pre.finalized_cacheable and self.finalize_cache:
      finalized_dir = utils.make_dir(
        preprocessed_dir / f"final-{pre.finalized_name}")
      finalized_format = pre.finalized_format or "pickle"
      finalized_file = finalized_dir / \
          f"{self.name}{ds_suffix}{idx_suffix}.{finalized_format}"
      return utils.cache(
        preproc, finalized_file, finalized_format)
    else:
      return preproc()

class PresplitDatasetProvider(DatasetProvider):
  loader: ldr.PresplitDatasetLoader = None
  _train_dataset = None
  _val_dataset = None
  _test_dataset = None

  def __init__(
    self, loader_train, loader_val=None, loader_test=None, **kwargs):
    super().__init__(
      loader.PresplitDatasetLoader(loader_train, loader_val, loader_test),
      outer_k=None, inner_k=None, **kwargs)

  def _load_dataset(self, only_meta=False, id=None):
    if id == "train":
      return self.loader.load_train_dataset(only_meta)
    elif id == "val":
      return self.loader.load_validation_dataset(only_meta)
    elif id == "test":
      return self.loader.load_test_dataset(only_meta)
    raise AssertionError(f"Unknown dataset id {id}.")

  def _cache_dataset(self, only_meta=False, id=None):
    if not self.in_memory_cache:
      if id is None:
        return dict(
          train=self._load_dataset(only_meta, "train"),
          val=self._load_dataset(
            only_meta, "val") if self.loader.val else None,
          test=self._load_dataset(
            only_meta, "test") if self.loader.test else None)
      else:
        return self._load_dataset(only_meta, id)

    if id == "train":
      if self._train_dataset is None or not (
        only_meta or "elements" in self._train_dataset):
        self._train_dataset = self._load_dataset(only_meta, "train")
      return self._train_dataset
    elif id == "val":
      if self._val_dataset is None or not (
        only_meta or "elements" in self._val_dataset):
        self._val_dataset = self._load_dataset(only_meta, "val")
      return self._val_dataset
    elif id == "test":
      if self._test_dataset is None or not (
        only_meta or "elements" in self._test_dataset):
        self._test_dataset = self._load_dataset(only_meta, "test")
      return self._test_dataset
    else:
      self._cache_dataset(only_meta, "train")
      if self.loader.val:
        self._cache_dataset(only_meta, "val")
      if self.loader.test:
        self._cache_dataset(only_meta, "test")
      return self._dataset

  def unload_dataset(self):
    self._train_dataset = None
    self._val_dataset = None
    self._test_dataset = None

  @property
  def _dataset(self):
    return dict(
      train=self._train_dataset,
      val=self._val_dataset,
      test=self._test_dataset)

  @property
  def in_meta(self):
    return self._cache_dataset(only_meta=True, id="train")["in_meta"]

  @property
  def out_meta(self):
    return self._cache_dataset(only_meta=True, id="train")["out_meta"]

  @property
  def full_name(self):
    return self.name + self.name_suffix

  @property
  def dataset(self):
    ds = self._cache_dataset(only_meta=False)
    return dict(
      train=ds["train"]["elements"],
      val=ds["val"]["elements"] if ds["val"] else None,
      test=ds["test"]["elements"] if ds["test"] else None
    )

  @property
  def train_dataset(self):
    return self._cache_dataset(only_meta=False, id="train")["elements"]

  @property
  def validation_dataset(self):
    return self._cache_dataset(only_meta=False, id="val")["elements"]

  @property
  def test_dataset(self):
    return self._cache_dataset(only_meta=False, id="test")["elements"]

  @property
  def dataset_split_sizes(self):
    ds = self._cache_dataset(only_meta=True)
    return dict(
      train=ds["train"]["size"],
      val=ds["val"]["size"] if ds["val"] else None,
      test=ds["test"]["size"] if ds["test"] else None
    )

  @property
  def train_dataset_size(self):
    return self.dataset_split_sizes["train"]

  @property
  def validation_dataset_size(self):
    return self.dataset_split_sizes["val"]

  @property
  def test_dataset_size(self):
    return self.dataset_split_sizes["test"]

  @property
  def dataset_size(self):
    return sum(fy.keep(self.dataset_split_sizes.values()))

  @property
  def stratify_labels(self):
    raise Exception("This dataset has fixed train, val and test splits.")

  def _make_splits(self):
    raise Exception("This dataset has fixed train, val and test splits.")

  def get(self, *args, **kwargs):
    raise Exception("Presplit datasets cannot be resliced.")

  def get_train_split_indices(
    self, outer_idx=None, inner_idx=None, relative=True):
    assert relative, "Absolute indexing is not possible in presplit datasets."
    return np.arange(self.train_dataset_size)

  def get_validation_split_indices(
    self, outer_idx=None, inner_idx=None, relative=True):
    assert relative, "Absolute indexing is not possible in presplit datasets."
    return np.arange(self.validation_dataset_size)

  def get_test_split_indices(
    self, outer_idx=None, inner_idx=None, relative=True):
    assert relative, "Absolute indexing is not possible in presplit datasets."
    return np.arange(self.test_dataset_size)

  def get_split_indices(self, outer_idx=None, inner_idx=None, relative=True):
    assert relative, "Absolute indexing is not possible in presplit datasets."
    return (
      self.get_train_split_indices(),
      self.get_validation_split_indices(),
      self.get_test_split_indices())

  def get_split(
    self, enc=None, config=None, outer_idx=None, inner_idx=None,
    only=None, finalize=True, reconfigurable_finalization=False, indices=None):
    assert indices is None or only is not None, \
        "Index subsets can only be provided if a single split is requested."
    pre = self._get_preprocessor(enc, config, reconfigurable_finalization)
    res = ()
    if only is None or only == "train":
      train_ds = self._preprocess(
        pre, lambda: self.train_dataset,
        indices, None, None, finalize, ("train",))
      if only == "train":
        return train_ds
      else:
        res += (train_ds,)

    if self.loader.val and (only is None or only == "val"):
      val_ds = self._preprocess(
        pre, lambda: self.validation_dataset,
        indices, False, None, finalize, ("val",))
      if only == "val":
        return val_ds
      res += (val_ds,)
    if self.loader.test and (only is None or only == "test"):
      test_ds = self._preprocess(
        pre, lambda: self.test_dataset,
        indices, False, None, finalize, ("test",))
      if only == "test":
        return test_ds
      res += (test_ds,)

    assert only is None, f"Invalid only-selector: {only}."
    return res

  def _compute_stats(self):
    computer = stats.find_stat_computer(self.dataset_type)
    ds = self.dataset

    return dict(
      train=computer(ds["train"]),
      val=computer(ds["val"]) if ds["val"] else None,
      test=computer(ds["test"]) if ds["test"] else None)
