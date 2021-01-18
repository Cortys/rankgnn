from abc import ABCMeta, abstractmethod
import numpy as np

import graspe.preprocessing.batcher as batcher

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
    pref_a = self.pref_a
    pref_b = self.pref_b

    return (
      object_finalize_fn(self.objects), pref_a, pref_b), np.ones(len(pref_a))

class UtilityPreferenceBatcher(batcher.Batcher, metaclass=ABCMeta):
  name = "util_pref"

  def __init__(self, in_meta, out_meta, **config):
    super().__init__(**config)

  def preprocess(self, elements):
    objects, us = elements
    sort_idx = np.argsort(us)

    return objects, us, sort_idx

  def iterate(self, elements):
    objects, us, sort_idx = elements
    olen = objects.size
    u_max = np.NINF
    prev_part = None
    curr_part = []
    i = 0

    while i < olen:
      idx = sort_idx[i]
      u = us[idx]

      if u_max < u:
        prev_part = curr_part
        curr_part = []
        u_max = u

      for p_idx in prev_part:
        yield p_idx, idx

      curr_part.append(idx)
      i += 1

  def create_aggregator(self, elements):
    objects = elements[0]

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
