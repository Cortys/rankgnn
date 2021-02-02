import numpy as np
from typing import get_type_hints
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

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

def make_holdout_split(holdout, objects, strat_labels=None):
  if holdout == 0:
    return objects, [], strat_labels
  else:
    train_split, test_split = train_test_split(
      np.arange(objects.size),
      test_size=holdout,
      stratify=strat_labels)

    if strat_labels is not None:
      strat_labels = strat_labels[train_split]

    return objects[train_split], objects[test_split], strat_labels

def make_kfold_splits(n, objects, strat_labels=None):
  if strat_labels is None:
    kfold = KFold(n_splits=n, shuffle=True)
  else:
    kfold = StratifiedKFold(n_splits=n, shuffle=True)

  for train_split, test_split in kfold.split(objects, strat_labels):
    yield (
      objects[train_split],
      objects[test_split],
      strat_labels[train_split] if strat_labels is not None else None
    )
