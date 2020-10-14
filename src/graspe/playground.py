import tensorflow as tf
from tensorflow import keras
import funcy as fy

import graspe.datasets.synthetic.datasets as syn
import graspe.utils as utils

ds = syn.triangle_classification_dataset()

fy.take(10, fy.map(lambda e: utils.draw_graph(*e), zip(*ds)))
