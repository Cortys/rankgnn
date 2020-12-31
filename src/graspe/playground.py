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

dataset = syn.triangle_classification_dataset()

wl1_ds = dataset.get_processed(
  ("wl1", "binary"), config=dict(batch_size_limit=100),
  shuffle=True)

dataset.in_meta
list(wl1_ds)

# len(dataset[0])
#
# preproc = tf_enc.WL1EmbedPreprocessor(out_meta={}, config=dict(batch_size_limit=100))
# wl1_ds = preproc.transform(dataset)
#
# list(wl1_ds)

#gnn.GIN()
