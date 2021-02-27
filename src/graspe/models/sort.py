import numpy as np

import graspe.metrics.rank as rank_metric

def sort(indices, compare):
  n = len(indices)

  if n <= 1:
    return indices

  sorted_indices = np.zeros(n, dtype=np.int32)
  ranges = [0]
  partitions = [indices]
  print(f"[dbg] started sorting {n} objects")
  m = 1
  while m > 0:
    todo = sum(len(p) for p in partitions)
    print(f"[dbg] {m} partitions, still unsorted: {todo}/{len(indices)}")
    comparisons = compare(partitions)
    new_ranges = []
    new_partitions = []
    comp_start = 0

    for i in range(m):
      p_start = ranges[i]
      p = partitions[i]
      p_len = len(p)
      lp = []
      rp = []

      for j in range(p_len - 1):
        if comparisons[comp_start + j] < 0.5:
          lp.append(p[j + 1])
        else:
          rp.append(p[j + 1])

      lp_len = len(lp)
      rp_len = len(rp)
      sorted_indices[p_start + lp_len] = p[0]
      comp_start += p_len - 1

      if lp_len == 1:
        sorted_indices[p_start] = lp[0]
      elif lp_len > 1:
        new_partitions.append(lp)
        new_ranges.append(p_start)

      if rp_len == 1:
        sorted_indices[p_start + lp_len + 1] = rp[0]
      elif rp_len > 1:
        new_partitions.append(rp)
        new_ranges.append(p_start + lp_len + 1)

    ranges = new_ranges
    partitions = new_partitions
    m = len(partitions)

  return sorted_indices

def model_sort(indices, provider_get, model=None, enc=None, **config):
  enc = enc or (model and model.enc)
  assert enc is not None, "Unknown encoding."

  data_getter = provider_get(enc, config={
    **config, "mode": "pivot_partitions"},
    reconfigurable_finalization=True)

  def run(model):
    def compare(partitions):
      data = data_getter(pivot_partitions=partitions)
      return model.predict(data)

    return sort(indices, compare)

  return run if model is None else run(model)

def evaluate_model_sort(indices, provider_get, model=None, enc=None, **config):
  enc = enc or (model and model.enc)
  assert enc is not None, "Unknown encoding."
  in_enc = enc[0]
  out_enc = enc[1]

  print("Loading target rankings...")
  _, object_rankings = provider_get(
    enc="null_in", indices=indices, config=config)
  print(f"Loaded {len(object_rankings)} target rankings.")

  if "pref" in in_enc:
    predicted_ordering = model_sort(indices, provider_get, enc=enc, **config)
  elif out_enc == "float":
    data = provider_get(enc, indices=indices, config=config)

    def predicted_ordering(model):
      predicted_rankings = model.predict(data)
      return indices[np.argsort(predicted_rankings)]
  else:
    raise Exception(f"Unsupported enc {enc}.")

  def rank_metrics(model):
    print("Computing sort metrics...")
    return rank_metric.bucket_sorted_metrics(
      indices, object_rankings, predicted_ordering(model))

  return rank_metrics if model is None else rank_metrics(model)
