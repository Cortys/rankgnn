import funcy as fy
import collections as coll

from graspe.utils import tolerant, select_prefixed_keys

def pipeline_step(f=None, macro=False, share_prefix=False):
  if f is None:
    return lambda f: pipeline_step(f, macro, share_prefix)

  if hasattr(f, "__pipeline_step__"):
    return f

  f = tolerant(f)

  @fy.wraps(f)
  def step(*args, prefix=None, **kwargs1):
    @fy.wraps(f)
    def execute(input, **kwargs2):
      kwargs = fy.merge(kwargs2, kwargs1)

      if prefix is not None:
        select_prefixed_keys(kwargs2, prefix + "_", target=kwargs)

      if share_prefix:
        res = f(input, *args, prefix=prefix, **kwargs)
      else:
        res = f(input, *args, **kwargs)

      if macro:
        return create_pipeline(res)(input, **kwargs)
      else:
        return res

    execute.__tolerant__ = True

    return execute

  step.__pipeline_step__ = True

  return step

def pipeline_start(f):
  f = tolerant(f)

  # input is ignored because f returns the initial input:
  @fy.wraps(f)
  def wrapper(_, **kwargs):
    return f(**kwargs)

  wrapper.__tolerant__ = True

  return pipeline_step(wrapper)


def to_executable_step(f):
  if hasattr(f, "__pipeline_step__"):
    return f()

  if isinstance(f, coll.Iterable):
    pipelines = [(
      create_pipeline(s) if isinstance(s, coll.Iterable)
      else to_executable_step(s))
      for s in f]

    def split_step(input, **kwargs):
      return tuple(p(input, **kwargs) for p in pipelines)

    return split_step

  return tolerant(f)

def create_pipeline(steps, **kwargs1):
  executable_steps = [to_executable_step(step) for step in steps]

  def pipeline(input=None, **kwargs2):
    a = input
    kwargs = fy.merge(kwargs1, kwargs2)

    for executable_step in executable_steps:
      a = executable_step(a, **kwargs)

    return a

  return pipeline
