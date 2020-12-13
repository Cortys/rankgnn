class Batcher:
  def preprocess(self, elements):
    return elements

  def iterate(self, elements):
    return elements

  def create_aggregator(self, elements):
    return []

  def append(self, aggregator, element):
    aggregator.append(element)

  def compute_space(self, element):
    return 0

  def batch(self, aggregator):
    return aggregator

  def unit_batch(self, element):
    agg = self.create_aggregator(element)
    self.append(agg, element)

    return self.batch(agg)

identity = Batcher()

class TupleBatcher(Batcher):
  def __init__(self, *batchers, size=2):
    super().__init__()

    lb = len(batchers)
    size = max(lb, size)
    if lb < size:
      batchers += (identity,) * (size - lb)

    self.batchers = batchers
    self.size = size

  def iterate(self, elements):
    return zip(*(b.iterate(e) for b, e in zip(self.batchers, elements)))

  def create_aggregator(self, elements):
    return tuple(b.create_aggregator(e) for b, e in zip(self.batchers, elements))

  def append(self, aggregator, element):
    for b, sub_agg, sub_e in zip(self.batchers, aggregator, element):
      b.append(sub_agg, sub_e)

  def compute_space(self, element):
    batchers = self.batchers
    return sum(batchers[i].compute_space(element[i]) for i in range(self.size))

  def batch(self, aggregator):
    return tuple(b.batch(a) for b, a in zip(self.batchers, aggregator))

class PairBatcher(TupleBatcher):
  def __init__(self, batcher=identity):
    super().__init__(batcher, batcher)
