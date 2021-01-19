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

# -%% codecell

def time_str():
  return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def experiment(provider, model, log=True, **config):
  enc = fy.first(provider.find_compatible_encoding(
    model.input_encodings, model.output_encodings))
  in_enc, out_enc = enc
  dim = 64
  edim = 64
  m = model(
    in_enc=in_enc, out_enc=out_enc,
    in_meta=provider.in_meta, out_meta=provider.out_meta,
    conv_layer_units=[dim, dim, dim],
    att_conv_layer_units=[dim, dim, 1],
    fc_layer_units=[dim, dim, edim],
    activation="sigmoid", inner_activation="relu",
    # att_conv_activation="relu",
    pooling="softmax")
  print("Instanciated model.")
  if provider.dataset_size < 10:
    ds_train = provider.get(enc)
    ds_val, ds_test = ds_train, ds_train
    targets = provider.dataset[1]
  else:
    ds_train, ds_val, ds_test = provider.get_split(
      enc, config=config,
      outer_idx=5)
    targets = provider.get_test_split(outer_idx=5)[1]

  print("Loaded encoded datasets.")
  provider.unload_dataset()
  opt = keras.optimizers.Adam(0.0001)

  if out_enc == "float32":
    loss = "mean_squared_error"
    metrics = []
  else:
    loss = "binary_crossentropy"
    metrics = ["binary_accuracy"]

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
    ds_train.cache(),
    validation_data=ds_val.cache(),
    epochs=5000, verbose=2,
    callbacks=[tb] if log else [])
  m.evaluate(ds_test)
  print(np.around(m.predict(ds_test), 2))
  print(targets)


# provider = syn.triangle_classification_dataset()
provider = syn.triangle_count_dataset()
# provider = tu.ZINC()
# provider = tu.Mutag()
model = gnn.RankWL2GNN
# model = gnn.RankGIN

experiment(provider, model, batch_size_limit=10000, log=False)

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
