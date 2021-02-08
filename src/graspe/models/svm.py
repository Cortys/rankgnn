import sklearn.svm as svm

import graspe.utils as utils
import graspe.chaining.scikit as cs
import graspe.preprocessing.scikit as sk_enc

def svm_model(in_enc, out_enc, **config):
  if out_enc == "class":
    cls = svm.SVC
    metric = "accuracy"
  else:
    cls = svm.SVR
    metric = "r2"

  return utils.tolerant(cls, ignore_varkwargs=True)(**config), metric


SVM = cs.create_model(
  "SVM", svm_model,
  input_encodings=sk_enc.vector_input_encodings,
  output_encodings=sk_enc.svm_output_encodings)

KernelSVM = cs.create_model(
  "KernelSVM", svm_model,
  kernel="precomputed",
  input_encodings=sk_enc.kernel_input_encodings,
  output_encodings=sk_enc.svm_output_encodings)
