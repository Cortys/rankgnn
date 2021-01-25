import scipy as sci

def bucket_sorted_tau(objects, object_ranks, predicted_ordering):
  lut = dict(zip(objects, object_ranks))
  real_ranks = []
  predicted_ranks = []

  curr_object_rank = None
  curr_pred_rank = -1

  for o in predicted_ordering:
    o_rank = lut[o]

    if curr_object_rank != o_rank:
      curr_object_rank = o_rank
      curr_pred_rank += 1

    real_ranks.append(o_rank)
    predicted_ranks.append(curr_pred_rank)

  return sci.stats.kendalltau(predicted_ranks, real_ranks).correlation
