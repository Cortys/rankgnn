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
import graspe.preprocessing.tf as tf_enc
import graspe.utils as utils
import graspe.models.gnn as gnn

dataset = syn.small_grid_dataset()
gs = dataset[0]

wl1_encoder = wl1_enc.WL1Encoder()
encoder = transformer.tuple(wl1_encoder)

wl1_encoder.preprocess(dataset[0])
encoder.transform(dataset)

encs, ys = encoder.preprocess(dataset)
encs, ys = encoder.transform(dataset)

encs
ys

batcher = transformer.tuple(transformer.pair(wl1_enc.WL1Batcher()))
gen = batcher.batch_generator(((encs, encs), ys), batch_size_limit=3)
ds = tf_enc.make_dataset(gen, dict(encoding="wl1_pair", feature_dim=1), dict(encoding="float32"))

list(ds)

#gnn.GIN()
