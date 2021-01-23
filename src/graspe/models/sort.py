import numpy as np

def sort(indices, compare):
  n = len(indices)

  if n <= 1:
    return indices

  sorted_indices = np.zeros(n, dtype=np.int32)
  ranges = [0]
  partitions = [indices]
  m = 1

  while m > 0:
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
  enc = (model.in_enc, model.out_enc)

  def compare(partitions):
    data = provider_get(enc, config={
      **config,
      "mode": "pivot_partitions",
      "pivot_partitions": partitions})
    return model.predict(data)

  return sort(indices, compare)