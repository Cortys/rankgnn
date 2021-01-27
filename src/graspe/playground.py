import tensorflow as tf
from tensorflow import keras
import funcy as fy
import networkx as nx
import numpy as np
import collections
from datetime import datetime

import graspe.datasets.synthetic.datasets as syn
import graspe.datasets.tu.datasets as tu
import graspe.preprocessing.utils as enc_utils
import graspe.preprocessing.graph.wl1 as wl1_enc
import graspe.preprocessing.transformer as transformer
import graspe.preprocessing.encoder as encoder
import graspe.preprocessing.preprocessor as preprocessor
import graspe.preprocessing.tf as tf_enc
import graspe.utils as utils
import graspe.models.gnn as gnn
import graspe.models.sort as sort

# -%% codecell

def time_str():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def experiment(provider, model, log=True, verbose=2, **config):
  enc = fy.first(provider.find_compatible_encoding(model))
  in_enc, out_enc = enc
  dim = 128
  depth = 5
  fc_layer_args = None
  if "pref" in in_enc:
    edim = dim
    fc_layer_args = {-1: dict(activation=None)}
    config["batch_size_limit"] *= config.get("neighbor_radius", 1)
  elif out_enc == "binary":
    edim = 1
  elif out_enc == "float32":
    edim = 1
    fc_layer_args = {-1: dict(activation=None)}
  m = model(
    in_enc=in_enc, out_enc=out_enc,
    in_meta=provider.in_meta, out_meta=provider.out_meta,
    conv_layer_units=[dim] * depth,
    att_conv_layer_units=[dim] * depth,
    fc_layer_units=[dim, dim, edim],
    fc_layer_args=fc_layer_args,
    cmp_layer_units=[edim],
    activation="sigmoid", inner_activation="relu",
    # att_conv_activation="relu",
    pooling="softmax")
  print("Instanciated model.", enc)
  if provider.dataset_size < 10:
    ds_train = provider.get(enc)
    ds_val, ds_test = ds_train, ds_train
    #  targets = provider.dataset[1]
  else:
    ds_train, ds_val, ds_test = provider.get_split(
      enc, config=config,
      outer_idx=5)
    #  targets = provider.get_test_split(outer_idx=5)[1]

  print("Loaded encoded datasets.")
  provider.unload_dataset()
  opt = keras.optimizers.Adam(0.0005)

  if out_enc == "multiclass":
    loss = "categorical_crossentropy"
    metrics = ["categorical_accuracy"]
  elif out_enc == "binary":
    loss = "binary_crossentropy"
    metrics = ["binary_accuracy"]
  else:
    loss = "mean_squared_error"
    metrics = ["mean_absolute_error"]

  m.compile(
    optimizer=opt, loss=loss, metrics=metrics)

  t = time_str()
  log_dir = f"../logs/{t}_{m.name}_{provider.name}/"
  tb = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=50,
    profile_batch="100,115")
  print("Compiled model.")
  m.fit(
    ds_train.cache().shuffle(200),
    validation_data=ds_val.cache(),
    epochs=1000, verbose=verbose,
    callbacks=[tb] if log else [])

  print(np.around(m.predict(ds_test), 2))
  return m


# provider = syn.triangle_classification_dataset()
# provider = syn.triangle_count_dataset()
provider = tu.ZINC(in_memory_cache=False)
# provider = tu.Mutag()
# provider = tu.Reddit5K()

# provider.dataset
model = gnn.CmpGIN
# model = gnn.DirectRankGIN

bsl = 2000
m = experiment(provider, model, batch_size_limit=bsl, neighbor_radius=50, log=False, verbose=1)
train_idxs, val_idxs, test_idxs = provider.get_split_indices(outer_idx=5)
# train_get = provider.get
# val_get = provider.get
# test_get = provider.get
train_get = provider.get_train_split
val_get = provider.get_validation_split
test_get = provider.get_test_split
# bsl = 1000
print("Train", sort.evaluate_model_sort(train_idxs, train_get, m, batch_size_limit=bsl))
print("Val", sort.evaluate_model_sort(val_idxs, val_get, m, batch_size_limit=bsl))
print("Test", sort.evaluate_model_sort(test_idxs, test_get, m, batch_size_limit=bsl))

# splits = provider.get_split(("wl1", "float32"), dict(batch_size_limit=500))
# provider.get_test_split(outer_idx=5)[1]
# fy.last(splits[2])
# fy.first(splits[2])
# provider.unload_dataset()
# provider.dataset
# utils.draw_graph(provider.train_dataset[0][10000])
# list(provider.get_test_split(("wl1", "float32"), dict(batch_size_limit=10)))[0]
# provider.dataset
# fy.first(provider.get(("wl1_pref", "binary"), config=dict(batch_size_limit=3)))
