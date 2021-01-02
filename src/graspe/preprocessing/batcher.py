import graspe.preprocessing.transformer as transformer

class Batcher(transformer.Transformer):
  def __init__(self, batch_size_limit=None, batch_space_limit=None, **kwargs):
    super().__init__()

    self.batch_size_limit = batch_size_limit
    self.batch_space_limit = batch_space_limit
    self.basename = self.name
    name = f"{self.name}_size{batch_size_limit}"

    if batch_space_limit is not None:
      name += f"_space{batch_space_limit}"

    self.name = name

  def compute_space(self, element):
    return 0

  def batch_generator(self, elements):
    batch_size_limit = self.batch_size_limit
    batch_space_limit = self.batch_space_limit
    elements = self.preprocess(elements)

    if batch_size_limit == 1:
      def batch_generator():
        for e in self.iterate(elements):
          if batch_space_limit is not None:
            assert self.compute_space(e) <= batch_space_limit

          batch = self.create_aggregator(elements)
          self.append(batch, e)
          yield self.finalize(batch)
    else:
      def batch_generator():
        batch = self.create_aggregator(elements)
        batch_size = 0
        batch_space = 0
        batch_full = False

        for e in self.iterate(elements):
          if batch_space_limit is not None:
            e_space = self.compute_space(e)
            assert e_space <= batch_space_limit
            batch_space += e_space

            if batch_space > batch_space_limit:
              batch_space = e_space
              batch_full = True

          if batch_size_limit is not None and batch_size >= batch_size_limit:
            batch_full = True

          if batch_full:
            yield self.finalize(batch)
            batch = self.create_aggregator(elements)
            batch_size = 0
            batch_full = False

          self.append(batch, e)
          batch_size += 1

        if batch_size > 0:
          yield self.finalize(batch)

    return batch_generator

  def transform(self, elements):
    return list(self.batch_generator(elements))

class TupleBatcher(transformer.TupleTransformer, Batcher):
  def __init__(self, *batchers, size=2, **kwargs):
    batch_space_limit = 0
    batch_size_limit = None
    space_limiting_batchers = []

    for i, bat in enumerate(batchers):
      assert isinstance(bat, Batcher)
      if bat.batch_size_limit is not None:
        if batch_size_limit is None:
          batch_size_limit = bat.batch_size_limit
        else:
          batch_size_limit = min(batch_size_limit, bat.batch_size_limit)
      if bat.batch_space_limit is not None:
        batch_space_limit += bat.batch_space_limit
        space_limiting_batchers.append(i)

    super().__init__(*batchers, size=size)
    self.batch_size_limit = kwargs.get(
      "batch_size_limit", batch_size_limit)
    self.batch_space_limit = kwargs.get(
      "batch_space_limit", batch_space_limit)
    self.space_limiting_batchers = space_limiting_batchers
    base = "-".join(b.basename for b in batchers)
    name = f"{base}-size{self.batch_size_limit}"

    if len(space_limiting_batchers) > 0:
      name += "_space{self.batch_space_limit}"

    self.name = name

  def compute_space(self, element):
    batchers = self.transformers
    return sum(
      batchers[i].compute_space(element[i])
      for i in self.space_limiting_batchers)


transformer.register_transformer(Batcher, TupleBatcher)
