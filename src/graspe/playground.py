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

def experiment(provider, model, **config):
  enc = fy.first(provider.find_compatible_encoding(
    model.input_encodings, model.output_encodings))
  in_enc, out_enc = enc
  m = model(
    in_enc=in_enc, out_enc=out_enc,
    in_meta=provider.in_meta, out_meta=provider.out_meta,
    conv_layer_units=[32, 32, 32],
    att_conv_layer_units=[32, 32, 1],
    fc_layer_units=[32, 32, 1],
    activation="sigmoid", inner_activation="relu",
    # att_conv_activation="relu",
    pooling="softmax")
  if provider.dataset_size < 10:
    ds_train = provider.get(enc)
    ds_val, ds_test = ds_train, ds_train
    targets = provider.dataset[1]
  else:
    ds_train, ds_val, ds_test = provider.get_split(
      enc, config=config,
      outer_idx=5)
    targets = provider.get_test_split(outer_idx=5)[1]

  provider.unload_dataset()
  opt = keras.optimizers.Adam(0.0001)
  m.compile(
    optimizer=opt, loss="binary_crossentropy", metrics=["binary_accuracy"])

  t = time_str()
  log_dir = f"../logs/{t}_{m.name}_{provider.name}/"
  tb = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=50,
    profile_batch="100,115")

  m.fit(
    ds_train.cache(),
    validation_data=ds_val.cache(),
    epochs=5000, verbose=2,
    callbacks=[tb])
  m.evaluate(ds_test)
  print(np.around(m.predict(ds_test), 2))
  print(targets)


# provider = syn.triangle_classification_dataset()
provider = tu.ZINC()
# provider = tu.Mutag()
model = gnn.GIN
experiment(provider, model, batch_size_limit=100)
# provider.dataset
