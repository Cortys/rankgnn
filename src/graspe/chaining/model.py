import funcy as fy

import graspe.chaining.pipeline as pipeline

def model_inputs(f):
  f = pipeline.tolerant(f)

  @fy.wraps(f)
  def wrapper(*args, **kwargs):
    inputs = f(*args, **kwargs)

    return inputs, inputs

  wrapper.__tolerant__ = True

  return pipeline.pipeline_start(wrapper)

def model_step(f=None, macro=False):
  if f is None:
    return lambda f: model_step(f, macro)

  f = pipeline.tolerant(f)

  @fy.wraps(f)
  def wrapper(io, *args, **kwargs):
    inputs, outputs = io
    new_outputs = f(outputs, *args, **kwargs)

    if macro:
      return new_outputs
    else:
      return inputs, new_outputs

  wrapper.__tolerant__ = True

  return pipeline.pipeline_step(wrapper, macro)

def process_model_encs(input_encodings, output_encodings, model_family):
  def processor(kwargs):
    in_enc = kwargs.get("in_enc", None)
    out_enc = kwargs.get("out_enc", None)
    family = kwargs.get("family", None)

    if "enc" in kwargs:
      enc = kwargs["enc"]
      del kwargs["enc"]
      if in_enc is None:
        in_enc = enc[0]
      if out_enc is None:
        out_enc = enc[1]
      if family is None and len(enc) > 2:
        family = enc[2]

    if in_enc is not None:
      assert in_enc in input_encodings,\
        f"'{in_enc}' inputs are not supported."
    else:
      assert len(input_encodings) == 1,\
        "Unknown model input encoding."
      in_enc = fy.first(input_encodings)

    if out_enc is not None:
      assert out_enc in output_encodings,\
        f"'{out_enc}' outputs are not supported."
    else:
      assert len(output_encodings) == 1,\
        "Unknown model output encoding."
      out_enc = fy.first(output_encodings)

    if family is not None and model_family is not None:
      assert family == model_family,\
        f"This model is not from the family '{family}'."
    else:
      family = model_family

    kwargs["in_enc"] = in_enc
    kwargs["out_enc"] = out_enc
    kwargs["family"] = family
    kwargs["enc"] = (in_enc, out_enc, family)
    return kwargs
  return processor

def add_enc(model, in_enc=None, out_enc=None, family=None):
  try:
    if in_enc is not None:
      model.in_enc = in_enc
    if out_enc is not None:
      model.out_enc = out_enc
    if family is not None:
      model.family = family
    model.enc = (in_enc, out_enc, family)
  finally:
    return model

def create_model(
  as_model, name, steps, extend_at=None,
  input_encodings=[], output_encodings=[],
  model_family=None,
  **kwargs):
  input_encodings = set(input_encodings)
  output_encodings = set(output_encodings)
  modelFactory = pipeline.create_pipeline(
    [*steps, as_model(name), add_enc],
    arg_transformer=process_model_encs(
      input_encodings, output_encodings, model_family),
    **kwargs)

  def extend(
    name, additional_steps, at=None,
    input_encodings=input_encodings, output_encodings=output_encodings,
    **additional_kwargs):
    ext_kwargs = fy.merge(kwargs, additional_kwargs)
    at = at or extend_at
    if at is not None and at != "end":
      before = steps[:at]
      after = steps[at:]
    else:
      before = steps
      after = []

    return create_model(
      as_model, name, before + additional_steps + after,
      extend_at=extend_at,
      input_encodings=input_encodings, output_encodings=output_encodings,
      **ext_kwargs)

  modelFactory.extend = extend
  modelFactory.name = name
  modelFactory.input_encodings = input_encodings
  modelFactory.output_encodings = output_encodings
  modelFactory.family = model_family

  return modelFactory

@pipeline.pipeline_step
def with_layers(
  io, layer, layer_units=[], layer_args=None,
  stack_tf=None, stack_tf_lookup=None,
  **kwargs):
  input, output = io
  layer = pipeline.tolerant(layer)
  hs = [output]
  layer_count = len(layer_units)

  if stack_tf is not None:
    if stack_tf_lookup is not None:
      stack_tf = stack_tf_lookup[stack_tf]

    if not callable(stack_tf):
      raise TypeError(
        "Stack transformers need to be callable or resolve to a callable.")

    stack_tf = pipeline.tolerant(stack_tf)

  for i in range(layer_count):
    if layer_args is None:
      args = kwargs
    elif isinstance(layer_args, dict):
      args = fy.merge(
        kwargs, layer_args.get(i, layer_args.get(i - layer_count, {})))
    elif i < layer_count and layer_args[i] is not None:
      args = fy.merge(kwargs, layer_args[i])
    else:
      args = kwargs

    units = layer_units[i]
    h = hs[i]

    if stack_tf is not None:
      h, units = stack_tf(
        h, units,
        input=input, hs=hs, layer_units=layer_units, i=i)

    hs.append(layer(units=units, **args)(h))

  return input, hs[-1]

@pipeline.pipeline_step
def with_layer(io, layer, with_inputs=False, **kwargs):
  if with_inputs:
    return pipeline.tolerant(layer)(**kwargs)(io)

  input, output = io

  return input, pipeline.tolerant(layer)(**kwargs)(output)

@pipeline.pipeline_step
def merge_ios(ios, merger=tuple):
  assert len(ios) > 0
  input = ios[0][0]

  return input, merger(output for _, output in ios)

# Layer Stack Transformers:

def stack_tf_seq(*transformers):
  transformers = [pipeline.tolerant(t) for t in transformers]

  def tf_seq(h, units, **kwargs):
    for t in transformers:
      h, units = t(h, units, **kwargs)

    return h, units

  return tf_seq

def add_input_tf(h, units, input):
  return (input, h), units
