from abc import ABCMeta, abstractmethod
import numpy as np
import scipy as sp
import collections

import rgnn.preprocessing.batcher as batcher

class PreferenceAggregator:
  def __init__(self, objects):
    self.all_objects = objects
    self.lut = np.full(objects.size, -1, dtype=np.int32)
    self.objects = []
    self.pref_a = []
    self.pref_b = []
    self.i = 0

  def append(self, pref):
    a, b = pref
    lut = self.lut
    pref_a = self.pref_a
    pref_b = self.pref_b
    i = self.i

    if lut[a] == -1:
      lut[a] = i
      a_lut = i
      self.objects.append(self.all_objects[a])
      i += 1
    else:
      a_lut = lut[a]

    if lut[b] == -1:
      lut[b] = i
      b_lut = i
      self.objects.append(self.all_objects[b])
      i += 1
    else:
      b_lut = lut[b]

    self.i = i
    pref_a.append(a_lut)
    pref_b.append(b_lut)

  def compute_space(self, pref, object_space_fn):
    a, b = pref
    space = 0
    lut = self.lut

    if lut[a] == -1:
      space += object_space_fn(self.all_objects[a])
    if lut[b] == -1:
      space += object_space_fn(self.all_objects[b])

    return space

  def finalize(self, object_finalize_fn):
    obj = object_finalize_fn(self.objects)
    pref_a = self.pref_a
    pref_b = self.pref_b

    return {
      **obj,
      "pref_a": pref_a,
      "pref_b": pref_b
    }, np.ones(len(pref_a))

def is_pair(elements):
  return isinstance(elements, tuple) and len(elements) == 2

class UtilityPreferenceBatcher(batcher.Batcher, metaclass=ABCMeta):
  name = "util_pref"

  def __init__(
    self, in_meta=None, out_meta=None,
    mode="train_neighbors", neighbor_radius=1, min_distance=0,
    sample_ratio=1, linear_samping=True, seed=1337, pivot_partitions=None,
    **config):
    super().__init__(**config)
    self.mode = mode
    self.neighbor_radius = neighbor_radius
    self.min_distance = min_distance
    self.sample_ratio = sample_ratio
    self.linear_sampling = linear_samping
    self.seed = seed
    self.pivot_partitions = pivot_partitions

  def preprocess(self, elements):
    if self.mode == "train_neighbors" or self.mode == "train_random":
      assert is_pair(elements), "Utilities are required during training."
      objects, us = elements
      sort_idx = np.argsort(us)

      return objects, us, sort_idx
    elif self.mode == "pivot_partitions":
      objects = elements[0] if is_pair(elements) else elements
      return objects
    else:
      raise AssertionError(f"Unknown mode '{self.mode}'.")

  def __iterate_train_neighbors(self, elements):
    objects, us, sort_idx = elements
    olen = objects.size
    u_max = np.NINF
    prev_parts = collections.deque(maxlen=self.neighbor_radius)
    dist = self.min_distance
    curr_part = []
    i = 0

    while i < olen:
      idx = sort_idx[i]
      u = us[idx]

      if u - u_max > dist:
        prev_parts.append(curr_part)
        curr_part = []
        u_max = u

      for prev_part in prev_parts:
        for p_idx in prev_part:
          yield p_idx, idx

      curr_part.append(idx)
      i += 1

  def __iterate_train_random(self, elements):
    objects, us, sort_idx = elements
    olen = objects.size
    seed = self.seed + olen
    pair_count = (olen * (olen - 1)) // 2
    sample_base = olen if self.linear_sampling else pair_count
    sample_size = min(int(self.sample_ratio * sample_base), pair_count)
    rng = np.random.default_rng(seed)

    sample = rng.choice(pair_count, sample_size, replace=False)
    sample_b = (np.sqrt(sample * 2 + 1/4) + 1/2).astype(np.int)
    sample_a = sample - (sample_b * (sample_b - 1)) // 2
    idx_a = sort_idx[sample_a]
    idx_b = sort_idx[sample_b]

    yield from zip(idx_a, idx_b)

  def __iterate_pivot_partitions(self, objects):
    partitions = self.pivot_partitions
    assert partitions is not None

    for partition in partitions:
      pivot_idx = partition[0]
      rest_idxs = partition[1:]
      for idx in rest_idxs:
        yield pivot_idx, idx

  def iterate(self, elements):
    if self.mode == "train_neighbors":
      yield from self.__iterate_train_neighbors(elements)
    elif self.mode == "train_random":
      yield from self.__iterate_train_random(elements)
    elif self.mode == "pivot_partitions":
      yield from self.__iterate_pivot_partitions(elements)
    else:
      raise AssertionError(f"Unknown mode '{self.mode}'.")

  def create_aggregator(self, elements):
    objects = elements if self.mode == "pivot_partitions" else elements[0]

    return PreferenceAggregator(objects)

  def append(self, batch, pref):
    batch.append(pref)

  @abstractmethod
  def compute_object_space(self, object):
    pass

  def compute_space(self, element, batch):
    return batch.compute_space(element, self.compute_object_space)

  @abstractmethod
  def finalize_objects(self, objects):
    pass

  def finalize(self, batch):
    return batch.finalize(self.finalize_objects)

class UtilityToNormalizedRankBatcher(batcher.Batcher):
  name = "util_to_normalized_rank_batcher"

  def preprocess(self, elements):
    if len(elements) == 0:
      return np.array([])
    if len(elements) == 1:
      return np.array([0.0])

    ranks = sp.stats.rankdata(elements, method="dense") - 1
    return np.true_divide(ranks, np.max(ranks))
