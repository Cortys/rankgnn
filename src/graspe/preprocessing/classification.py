import numpy as np

import graspe.preprocessing.encoder as encoder

class MulticlassEncoder(encoder.Encoder):
  name = "multiclass"

  def __init__(self, classes=2):
    self.classes = classes
    self.e = np.eye(classes)

  def encode_element(self, c):
    return self.e[c - 1]
