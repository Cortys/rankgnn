import tensorflow as tf
from tensorflow import keras
import funcy as fy
import networkx as nx
import numpy as np
import collections

import graspe.datasets.synthetic.datasets as syn
import graspe.encoders.wl1 as wl1_enc
import graspe.encoders.utils as enc_utils
import graspe.utils as utils



gs, ys = syn.small_grid_dataset()

encs = fy.lmap(wl1_enc.encode_graph, gs)

# wl1_enc.encode_graph(gs[0], (5, 1, 2, 0, 4, 3))

encs

list(enc_utils.make_batch_generator((encs,ys), wl1_enc.make_batch, batch_size_limit=4)())
