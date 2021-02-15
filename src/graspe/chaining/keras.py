from tensorflow import keras
import funcy as fy

import graspe.utils as utils
import graspe.chaining.pipeline as pipeline
import graspe.chaining.model as model

@pipeline.pipeline_step
def as_model(
  io, name, compile=True,
  optimizer="adam",
  learning_rate=None,
  loss="mse",
  metrics=[]):
  inputs, outputs = io

  if isinstance(inputs, dict):
    inputs = list(inputs.values())

  m = keras.Model(
    inputs=inputs, outputs=outputs, name=name)

  if compile:
    if learning_rate is not None and isinstance(optimizer, str):
      optimizer = dict(
        class_name=optimizer,
        config=dict(learning_rate=learning_rate))
    optimizer = keras.optimizers.get(optimizer)
    loss = keras.losses.get(loss)
    metrics = [] if metrics is None else [
      keras.metrics.get(m) for m in metrics]

    m.compile(
      optimizer=optimizer, loss=loss, metrics=metrics)

  return m

def arg_transformer(evaluation_selector):
  if evaluation_selector is None:
    return fy.identity

  def transformer(kwargs):
    if not kwargs.get("compile", True):
      return kwargs

    loss, metrics = utils.tolerant(evaluation_selector)(**kwargs)

    if loss is not None and "loss" not in kwargs:
      kwargs["loss"] = loss
    if metrics is not None and "metrics" not in kwargs:
      kwargs["metrics"] = metrics

    return kwargs

  return transformer

def default_evaluation_selector(out_enc):
  if out_enc == "multiclass":
    loss = "categorical_crossentropy"
    metrics = ["categorical_accuracy"]
  elif out_enc == "binary":
    loss = "binary_crossentropy"
    metrics = ["binary_accuracy"]
  else:
    loss = "mse"
    metrics = ["mae"]

  return loss, metrics

def create_model(
  name, steps, evaluation_selector=default_evaluation_selector,
  **kwargs):
  return model.create_model(
    as_model, name, steps,
    model_family="tf",
    arg_transformer=arg_transformer(evaluation_selector),
    **kwargs)
