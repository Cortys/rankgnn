import sklearn.svm as svm

import rgnn.utils as utils
import rgnn.chaining.scikit as cs
import rgnn.preprocessing.scikit as sk_enc

def svm_model(in_enc, out_enc, **config):
  if out_enc == "class":
    if config.get("kernel", "linear") == "linear":
      cls = svm.LinearSVC
      if "dual" not in config:
        config["dual"] = False
    else:
      cls = svm.SVC
    metric = "accuracy"
  else:
    if config.get("kernel", "linear") == "linear":
      cls = svm.LinearSVR
      if "dual" not in config:
        config["dual"] = False
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
