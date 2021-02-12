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

def model_sort(indices, provider_get, model, **config):
  data_getter = provider_get(model.enc, config={
    **config, "mode": "pivot_partitions"},
    reconfigurable_finalization=True)

  def compare(partitions):
    data = data_getter(pivot_partitions=partitions)
    return model.predict(data)

  return sort(indices, compare)

def evaluate_model_sort(indices, provider_get, model, **config):
  print("Loading target rankings...")
  _, object_rankings = provider_get(indices=indices, config=config)
  print(f"Loaded {len(object_rankings)} target rankings.")

  if "pref" in model.in_enc:
    predicted_ordering = model_sort(indices, provider_get, model, **config)
  elif model.out_enc == "float":
    data = provider_get(model.enc, indices=indices, config=config)
    predicted_rankings = model.predict(data)
    predicted_ordering = indices[np.argsort(predicted_rankings)]
  else:
    raise Exception(f"Unsupported enc {model.enc}.")

  print("Computing sort metrics...")
  return rank_metric.uocked_sorted_metrics(
    indices, object_rankings, predicted_ordering)
