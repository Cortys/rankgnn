class Transformer:
  name = "id"

  def preprocess(self, elements):
    return elements

  def iterate(self, elements):
    return elements

  def create_aggregator(self, elements):
    return []

  def append(self, aggregator, element):
    aggregator.append(element)

  def finalize(self, aggregator):
    return aggregator

  def transform(self, elements):
    elements = self.preprocess(elements)
    agg = self.create_aggregator(elements)

    for element in self.iterate(elements):
      self.append(agg, element)

    return self.finalize(agg)


Transformer.identity = Transformer()

class TupleTransformer(Transformer):
  def __init__(self, *transformers, size=2):
    super().__init__()

    lb = len(transformers)
    size = max(lb, size)
    if lb < size:
      transformers += (self.identity,) * (size - lb)

    self.transformers = transformers
    self.size = size
    self.names = [t.name for t in transformers]
    self.name = "-".join(self.names)

  def preprocess(self, elements):
    return tup(
      t.preprocess(el) for t, el in zip(self.transformers, elements))

  def iterate(self, elements):
    return zip(*(t.iterate(e) for t, e in zip(self.transformers, elements)))

  def create_aggregator(self, elements):
    return tup(
      t.create_aggregator(e) for t, e in zip(self.transformers, elements))

  def append(self, aggregator, element):
    for t, sub_agg, sub_e in zip(self.transformers, aggregator, element):
      t.append(sub_agg, sub_e)

  def finalize(self, aggregator):
    return tup(t.finalize(a) for t, a in zip(self.transformers, aggregator))


tuple_transformers = dict()

def register_transformer(transformer_cls, tuple_cls):
  transformer_cls.identity = transformer_cls()
  tuple_transformers[transformer_cls] = tuple_cls


tup = tuple
def tuple(*transformers, size=2, **kwargs):
  assert len(transformers) > 0, "At least one transformer is required."
  classes = transformers[0].__class__.__mro__
  tuple_cls = TupleTransformer
  for cls in classes:
    if cls in tuple_transformers:
      tuple_cls = tuple_transformers[cls]
      break

  return tuple_cls(*transformers, size=size, **kwargs)

def pair(transformer=Transformer.identity):
  t = tuple(transformer, transformer)
  t.name = f"{transformer.name}_pair"
  return t
