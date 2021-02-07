import funcy as fy

import graspe.utils as utils
import graspe.chaining.model as model

class History:
  def __init__(self, history):
    self.history = history

class SkModel:
  model_ctr = None
  defaults = {}

  def __init__(self, in_enc=None, out_enc=None, **config):
    self._validate_enc(in_enc, out_enc)
    self.in_enc = in_enc
    self.out_enc = out_enc
    self.config = config
    model, metric = utils.tolerant_method(self.model_factory)(
      in_enc=in_enc, out_enc=out_enc,
      **fy.merge(self.defaults, config))
    self.model = model
    self.metric_name = metric

  @property
  def metrics_names(self):
    return [self.metric_name]

  def _validate_enc(self, in_enc, out_enc):
    model.validate_model_encs(
      self.input_encodings, self.output_encodings)(None, in_enc, out_enc)

  def fit(
    self, training_data, validation_data=None, **kwargs):
    train = self.model.fit(*training_data).score(*training_data)

    history = {
      self.metric_name: [train]
    }

    if validation_data is not None:
      val = self.model.score(*validation_data)
      history["val_" + self.metric_name] = [val]

    return History(history)

  def evaluate(self, data, **kwargs):
    return [self.model.score(*data)]

  def predict(self, data, **kwargs):
    if isinstance(data, tuple) and len(data) == 2:
      data = data[0]
    return self.model.predict(data)

def create_model(
  name, factory,
  input_encodings=[], output_encodings=[], **config):
  n = name
  ie = input_encodings
  oe = output_encodings

  class Model(SkModel):
    name = n
    input_encodings = ie
    output_encodings = oe
    model_factory = factory
    defaults = config

  return Model
