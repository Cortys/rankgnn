import tensorflow as tf
from tensorflow import keras
import funcy as fy
import networkx as nx
import numpy as np
import collections

import graspe.datasets.synthetic.datasets as syn
import graspe.preprocessing.utils as enc_utils
import graspe.preprocessing.graph.wl1 as wl1_enc
import graspe.preprocessing.transformer as transformer
import graspe.preprocessing.encoder as encoder
import graspe.preprocessing.preprocessor as preprocessor
import graspe.preprocessing.tf as tf_enc
import graspe.utils as utils
import graspe.models.gnn as gnn

# -%% codecell

def experiment(provider, model):
  enc = fy.first(provider.find_compatible_encoding(
    model.input_encodings, model.output_encodings))
  in_enc, out_enc = enc
  m = model(
    in_enc=in_enc, out_enc=out_enc,
    in_meta=provider.in_meta, out_meta=provider.out_meta,
    conv_layer_units=[32, 32, 32], fc_layer_units=[32, 32, 1],
    activation="sigmoid", inner_activation="relu")
  if provider.dataset_size < 10:
    ds_train = provider.get(enc)
    ds_val, ds_test = ds_train, ds_train
  else:
    ds_train, ds_val, ds_test = provider.get_split(
      enc, config=dict(batch_size_limit=228),
      outer_idx=5)

  opt = keras.optimizers.Adam(0.0001)
  m.compile(
    optimizer=opt, loss="binary_crossentropy", metrics=["binary_accuracy"])
  m.fit(ds_train, validation_data=ds_val, epochs=5000, verbose=2)
  m.evaluate(ds_test)
  # return m.predict(ds_test), provider.get_test_split()[1]


provider = syn.triangle_classification_dataset()
model = gnn.GIN
experiment(provider, model)
