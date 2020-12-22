import graspe.preprocessing.transformer as transformer

class Batcher(transformer.Transformer):
  def compute_space(self, element):
    return 0

  def batch_generator(
    self, elements,
    batch_size_limit=100, batch_space_limit=None):
    elements = self.preprocess(elements)

    if batch_size_limit == 1:
      def batch_generator():
        for e in self.iterate(elements):
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

          if batch_full or batch_size >= batch_size_limit:
            yield self.finalize(batch)
            batch = self.create_aggregator(elements)
            batch_size = 0
            batch_full = False

          self.append(batch, e)
          batch_size += 1

        if batch_size > 0:
          yield self.finalize(batch)

    return batch_generator

class TupleBatcher(transformer.TupleTransformer, Batcher):
  def __init__(self, *batchers, size=2):
    for bat in batchers:
      assert isinstance(bat, Batcher)

    super().__init__(*batchers, size=size)

  def compute_space(self, element):
    batchers = self.transformers
    return sum(
      batchers[i].compute_space(element[i]) for i in range(self.size))


transformer.register_transformer(Batcher, TupleBatcher)
