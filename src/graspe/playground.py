import tensorflow as tf
from tensorflow import keras
import funcy as fy
import networkx as nx
import numpy as np
import collections
from datetime import datetime

import graspe.datasets.provider as prov
import graspe.datasets.synthetic.datasets as syn
import graspe.datasets.tu.datasets as tu
import graspe.datasets.ogb.datasets as ogb
import graspe.preprocessing.utils as enc_utils
import graspe.preprocessing.graph.graph2vec as g2v
import graspe.preprocessing.transformer as transformer
import graspe.preprocessing.encoder as encoder
import graspe.preprocessing.preprocessor as preprocessor
import graspe.preprocessing.tf as tf_enc
import graspe.utils as utils
import graspe.models.nn as nn
import graspe.models.gnn as gnn
import graspe.models.svm as svm
import graspe.models.sort as sort

# -%% codecell

def time_str():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def experiment(
  provider, model, log=True, verbose=2, epochs=1000,
  prefer_in_enc=None, prefer_out_enc=None, **config):
  encs = list(provider.find_compatible_encodings(model))
  if len(encs) > 1:
    filtered_encs = list(fy.filter(
      lambda enc: (
        (prefer_in_enc is None or enc[0] == prefer_in_enc)
        and (prefer_out_enc is None or enc[1] == prefer_out_enc)
      ), encs))
    if len(filtered_encs) > 0:
      encs = filtered_encs

  enc = encs[0]
  in_enc = enc[0]
  out_enc = enc[1]
  dim = 32
  depth = 5
  fc_layer_args = None

  print(f"Evaluating model with enc {enc}...")

  if "pref" in in_enc:
    edim = dim
    fc_layer_args = {-1: dict(activation=None)}
  elif out_enc == "binary":
    edim = 1
  elif out_enc == "float":
    edim = 1
    fc_layer_args = {-1: dict(activation=None)}
  else:
    edim = None
  m = model(
    enc=enc,
    in_meta=provider.in_meta, out_meta=provider.out_meta,
    conv_layer_units=[dim] * depth,
    att_conv_layer_units=[dim] * depth,
    fc_layer_units=[dim, dim, edim],
    fc_layer_args=fc_layer_args,
    cmp_layer_units=[edim],
    activation="sigmoid", inner_activation="relu",
    # att_conv_activation="relu",
    pooling="softmax",
    learning_rate=0.0001,
    # kernel="rbf",
    C=0.1)
  print("Instanciated model.")
  if provider.dataset_size < 10:
    ds_train = provider.get(enc, config=config)
    ds_val, ds_test = ds_train, ds_train
    #  targets = provider.dataset[1]
  else:
    ds_train, ds_val, ds_test = provider.get_split(
      enc, config=config)
    #  targets = provider.get_test_split(outer_idx=5)[1]

  print("Loaded encoded datasets.")
  provider.unload_dataset()

  if isinstance(m, keras.Model):
    t = time_str()
    log_dir = f"../logs/{t}_{m.name}_{provider.name}/"
    tb = keras.callbacks.TensorBoard(
      log_dir=log_dir,
      histogram_freq=50,
      profile_batch="100,115")
    print("Compiled model.")
    print("Fitting Keras model...")

    m.fit(
      ds_train.cache(),
      validation_data=ds_val.cache(),
      epochs=epochs, verbose=verbose,
      callbacks=[tb] if log else [])
  else:
    print("Fitting model...")
    m.fit(ds_train, validation_data=ds_val)

  print("Completed training.", m.evaluate(ds_test))
  print(np.around(m.predict(ds_test), 2))
  return m

def sort_experiment(provider, model, **config):
  print("Staring sort experiment...")
  bsl = 40000
  m = experiment(
    provider, model, batch_size_limit=bsl,
    mode="train_random",
    neighbor_radius=2, sample_ratio=20,  # => ~4 comps. per graph (i.e. linear)
    min_distance=0.001, log=False, verbose=2, **config)
  train_idxs, val_idxs, test_idxs = provider.get_split_indices(relative=True)
  train_get = provider.get_train_split
  val_get = provider.get_validation_split
  test_get = provider.get_test_split

  bsl = 1000
  print("Train", sort.evaluate_model_sort(train_idxs, train_get, m, batch_size_limit=bsl, **config))
  print("Val", sort.evaluate_model_sort(val_idxs, val_get, m, batch_size_limit=bsl, **config))
  print("Test", sort.evaluate_model_sort(test_idxs, test_get, m, batch_size_limit=bsl, **config))
  return m


# provider = syn.triangle_classification_dataset()
# provider = syn.triangle_count_dataset()
# provider = syn.triangle_count_dataset(default_split="count_extrapolation")
provider = tu.ZINC_full(in_memory_cache=False)
# provider = tu.TRIANGLES(in_memory_cache=False)
# provider = ogb.Mollipo()
# provider = ogb.Molfreesolv()

# model = gnn.CmpGIN
model = gnn.DirectRankGIN
# model = gnn.DirectRankWL2GNN
# model = gnn.WL2GNN
# model = gnn.GIN
# model = svm.KernelSVM
# model = svm.SVM
# model = nn.MLP

# print("no feats:")
# m = sort_experiment(provider, model, epochs=1000, T=5, prefer_in_enc="wlst", ignore_node_features=True, nystroem=500)
# print()
print("with feats:")
sort_experiment(provider, model, epochs=1000, T=5, prefer_in_enc="wlst", ignore_node_features=False, nystroem=500)
