import numpy as np
import funcy as fy
from typing import get_type_hints

def make_graph_batch(encoded_graphs, ref_keys, masking_fns=None, meta_fns={}):
  X_batch_size = 0
  ref_batch_size = 0
  batch_size = len(encoded_graphs)

  ref_a_key = ref_keys[0]

  for e in encoded_graphs:
    X_size, X_dim = e["X"].shape
    X_batch_size += X_size
    ref_batch_size += len(e[ref_a_key])

  X_batch = np.empty((X_batch_size, X_dim), dtype=float)
  graph_idx = np.empty((X_batch_size,), dtype=int)
  refs_batch = {
    ref_key: np.empty((ref_batch_size,), dtype=int)
    for ref_key in ref_keys}
  metadata_batch = {
    key: np.empty((batch_size,), dtype=get_type_hints(f).get("return", int))
    for key, f in meta_fns.items()}

  if masking_fns is None:
    masking_meta = {}
  else:
    masking_meta = {
      key: np.empty((ref_batch_size,), dtype=int)
      for key in masking_fns}
    masking_meta["X_idx"] = np.empty((X_batch_size,), dtype=int)

  X_offset = 0
  ref_offset = 0

  for i, e in enumerate(encoded_graphs):
    X = e["X"]
    X_size = len(X)

    if X_size == 0:  # discard empty graphs
      continue

    ref_size = len(e[ref_a_key])
    next_X_offset = X_offset + X_size
    next_ref_offset = ref_offset + ref_size

    X_batch[X_offset:next_X_offset] = X
    graph_idx[X_offset:next_X_offset] = [i] * X_size

    for ref_key in ref_keys:
      refs_batch[ref_key][ref_offset:next_ref_offset] = e[ref_key] + X_offset

    for meta_key, meta_fn in meta_fns.items():
      metadata_batch[meta_key][i] = meta_fn(e)

    if masking_fns is not None:
      masking_meta["X_idx"][X_offset:next_X_offset] = np.arange(X_size)

      for mask_key, mask_fn in masking_fns.items():
        masking_meta[mask_key][ref_offset:next_ref_offset] = mask_fn(e)

    X_offset = next_X_offset
    ref_offset = next_ref_offset

  return {
    "X": X_batch,
    **refs_batch,
    "graph_idx": graph_idx,
    **masking_meta,
    **metadata_batch
  }

def attacher(prop_name, flag=False):
  def check(f):
    return hasattr(f, prop_name)

  if flag:
    def attach(f):
      f.__dict__[prop_name] = True
      return f

    return attach, check

  def attach(val):
    def attacher(f):
      if val is not None:
        f.__dict__[prop_name] = val
      return f
    return attacher

  def get(f):
    return getattr(f, prop_name, None)

  return attach, get, check

combined, is_combined = attacher("__combined", True)
with_preprocessor, get_preprocessor, has_preprocessor = attacher("__preprocessor")
with_space_fn, get_space_fn, has_space_fn = attacher("__space_fn")

def combine_fns(fns, aggregator=tuple, cutoff=False):
  if fns is None:
    fns = ()
  elif callable(fns):
    if is_combined(fns):
      return fns

    fns = (fns,)

  fns_len = len(fns)

  @combined
  def combination(xs):
    res = aggregator(f(x) for f, x in zip(fns, xs[:fns_len]))

    if not cutoff:
      res += xs[fns_len:]

    return res

  return combination

def make_batch_generator(
  elements, batcher=None, batch_size_limit=100,
  element_space_fn=None, batch_space_limit=None):
  if has_preprocessor(batcher):
    elements = get_preprocessor(batcher)(elements)

  if element_space_fn is None and has_space_fn(batcher):
    element_space_fn = get_space_fn(batcher)

  if not isinstance(elements, tuple):
    elements = (elements,)
    if not is_combined(batcher):
      batcher = fy.compose(batcher, fy.first)
  else:
    batcher = combine_fns(batcher)

  element_space_fn = combine_fns(element_space_fn, sum, True)
  re = range(len(elements))

  if batch_size_limit == 1:
    def batch_generator():
      for e in zip(*elements):
        yield batcher(tuple(fy.map(lambda x: [x], e)))
  else:
    def batch_generator():
      batches = tuple([] for _ in re)
      batch_size = 0
      batch_space = 0
      batch_full = False

      for e in zip(*elements):
        if batch_space_limit is not None:
          e_space = element_space_fn(e)
          assert e_space <= batch_space_limit
          batch_space += e_space

          if batch_space > batch_space_limit:
            batch_space = e_space
            batch_full = True

        if batch_full or batch_size >= batch_size_limit:
          yield batcher(batches)
          batches = tuple([] for _ in re)
          batch_size = 0
          batch_full = False

        for batch, x in zip(batches, e):
          batch.append(x)
        batch_size += 1

      if batch_size > 0:
        yield batcher(batches)

  return batch_generator
