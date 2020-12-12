import tensorflow as tf
from tensorflow import keras
import funcy as fy
import networkx as nx
import numpy as np
import collections

import graspe.datasets.synthetic.datasets as syn
import graspe.encoders.utils as enc_utils
import graspe.encoders.wl1 as wl1_enc
import graspe.encoders.problems as problem_enc
import graspe.encoders.tf as tf_enc
import graspe.utils as utils



gs, ys = syn.small_grid_dataset()

encs = fy.lmap(wl1_enc.encode_graph, gs)

# wl1_enc.encode_graph(gs[0], (5, 1, 2, 0, 4, 3))

encs

batcher = problem_enc.pair_embed_batcher(wl1_enc.make_batcher())
gen = enc_utils.make_batch_generator(((encs, encs),ys), batcher, batch_size_limit=2)
ds = tf_enc.make_dataset(gen, (dict(encoding="wl1_pair", feature_dim=1), dict(encoding="float32")))

list(ds)
