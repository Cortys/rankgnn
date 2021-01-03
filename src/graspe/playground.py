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

provider = syn.triangle_classification_dataset()

provider.in_meta
provider.dataset_size

# -%% codecell

in_enc, out_enc = fy.first(provider.find_compatible_encoding(gnn.GIN.input_encodings, gnn.GIN.output_encodings))

model = gnn.GIN(in_enc=in_enc, out_enc=out_enc, in_meta=provider.in_meta, out_meta=provider.out_meta, conv_layer_units=[32, 32, 32], fc_layer_units=[32, 1], activation="sigmoid")
ds_train, ds_val, ds_test = provider.get_processed_split((in_enc, out_enc), config=dict(batch_size_limit=100))
# ds_train = provider.get_processed((in_enc, out_enc))
# ds_val, ds_test = ds_train, ds_train
opt = keras.optimizers.Adam(0.0001)

model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["binary_accuracy"])
model.fit(ds_train, validation_data=ds_val, epochs=1000, verbose=2)
model.predict(ds_test)
list(ds_test)[0][1]
model.evaluate(ds_test)
list(ds_train)[0]
