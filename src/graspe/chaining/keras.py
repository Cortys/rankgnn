from tensorflow import keras
import funcy as fy

import graspe.chaining.pipeline as pipeline
import graspe.chaining.model as model

@pipeline.pipeline_step
def as_model(io, name):
  inputs, outputs = io

  if isinstance(inputs, dict):
    inputs = list(inputs.values())

  return keras.Model(
    inputs=inputs, outputs=outputs, name=name)


create_model = fy.partial(model.create_model, as_model, model_family="tf")
