import graspe.encoders.utils as enc_utils

def embed_batcher(object_batcher):
  @enc_utils.combined
  @enc_utils.with_space_fn(enc_utils.get_space_fn(object_batcher))
  def batcher(s):
    x, y = s

    return object_batcher(x), y

  return batcher

def pair_embed_batcher(object_batcher):
  def linearize(s):
    (x1, x2), y = s
    return x1, x2, y

  if enc_utils.has_space_fn(object_batcher):
    sfn = enc_utils.get_space_fn(object_batcher)

    @enc_utils.combined
    def space_fn(s):
      x1, x2, y = s

      return sfn(x1) + sfn(x2)
  else:
    space_fn = None

  @enc_utils.combined
  @enc_utils.with_preprocessor(linearize)
  @enc_utils.with_space_fn(space_fn)
  def batcher(s):
    x1, x2, y = s

    return (object_batcher(x1), object_batcher(x2)), y

  return batcher
