import funcy as fy

import graspe.utils as utils
import graspe.chaining.model as model

class History:
  def __init__(self, history):
    self.history = history

class SkModel:
  model_ctr = None
  defaults = {}
  family = "scikit"

  def __init__(self, **config):
    self.config = self._validate_enc(config)
    self.in_enc = self.config.get("in_enc", None)
    self.out_enc = self.config.get("out_enc", None)
    self.config = config
    model, metric = utils.tolerant_method(self.model_factory)(
      **fy.merge(self.defaults, config))
    self.model = model
    self.metric_name = metric

  @property
  def enc(self):
    return (self.in_enc, self.out_enc, self.family)

  @property
  def metrics_names(self):
    return [self.metric_name]

  def _validate_enc(self, config):
    return model.process_model_encs(
      self.input_encodings, self.output_encodings, self.family)(config)

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
