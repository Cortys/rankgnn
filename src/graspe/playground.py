import tensorflow as tf
from tensorflow import keras
import funcy as fy
import networkx as nx

import graspe.datasets.synthetic.datasets as syn
import graspe.datasets.encoders.wl1 as wl1_enc
import graspe.utils as utils


gs, ys = syn.threesix_dataset()

encs = fy.lmap(wl1_enc.encode_graph, gs)

gs[0].nodes[0]

encs

wl1_enc.make_batch(encs)
