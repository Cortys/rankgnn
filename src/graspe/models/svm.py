import sklearn.svm as svm

import graspe.utils as utils
import graspe.chaining.scikit as cs
import graspe.preprocessing.kernel as kernel

def svm_model(in_enc, out_enc, **config):
  if out_enc == "class":
    cls = svm.SVC
    metric = "accuracy"
  else:
    cls = svm.SVR
    metric = "r2"

  return utils.tolerant(cls, ignore_varkwargs=True)(**config), metric


WL_st = cs.create_model(
  "WL_st", svm_model,
  kernel="precomputed",
  input_encodings=["gram_wlst"],
  output_encodings=kernel.output_encodings)

WL_sp = cs.create_model(
  "WL_sp", svm_model,
  kernel="precomputed",
  input_encodings=["gram_wlsp"],
  output_encodings=kernel.output_encodings)
