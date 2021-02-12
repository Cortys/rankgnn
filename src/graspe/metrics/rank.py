import numpy as np
import scipy as sci
import sklearn.metrics as metrics

def bucked_sorted_list_to_ranking(objects, object_ranks, predicted_ordering):
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

  return predicted_ranks, real_ranks

def bucket_sorted_tau(objects, object_ranks, predicted_ordering):
  predicted_ranks, real_ranks = bucked_sorted_list_to_ranking(
    objects, object_ranks, predicted_ordering)

  return sci.stats.kendalltau(predicted_ranks, real_ranks).correlation

def bucket_sorted_ndcg(objects, object_ranks, predicted_ordering):
  predicted_ranks, real_ranks = bucked_sorted_list_to_ranking(
    objects, object_ranks, predicted_ordering)

  return metrics.ndcg_score(real_ranks, predicted_ranks)

def uocked_sorted_metrics(objects, object_ranks, predicted_ordering):
  predicted_ranks, real_ranks = bucked_sorted_list_to_ranking(
    objects, object_ranks, predicted_ordering)

  print(predicted_ranks, real_ranks)

  return dict(
    tau=sci.stats.kendalltau(predicted_ranks, real_ranks).correlation,
    ndcg=metrics.ndcg_score([real_ranks], [predicted_ranks]))
