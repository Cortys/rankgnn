import numpy as np

import rgnn.preprocessing.encoder as encoder

class MulticlassEncoder(encoder.Encoder):
  name = "multiclass"

  def __init__(self, min=0, max=1, classes=None):
    if classes is None:
      classes = 1 + max - min
    self.min = min
    self.classes = classes
    self.e = np.eye(classes)

  def encode_element(self, c):
    return self.e[c - self.min]
