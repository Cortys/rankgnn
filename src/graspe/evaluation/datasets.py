import funcy as fy

import graspe.datasets.synthetic.datasets as syn
import graspe.datasets.tu.datasets as tu
import graspe.datasets.ogb.datasets as ogb

base_preprocess_config = dict(
  batch_size_limit=40000)
random_preprocess_config = dict(
  **base_preprocess_config,
  mode="train_random",
  sample_ratio=20)

# Synthetic:

triangle_count = fy.partial(
  syn.triangle_count_dataset,
  outer_k=1,
  default_preprocess_config=base_preprocess_config)

# TU:
ZINC = fy.partial(
  tu.ZINC_full,
  default_preprocess_config={
    **random_preprocess_config,
    "batch_size_limit": 10000})

# OGB:

Mollipo = fy.partial(
  ogb.Mollipo,
  default_preprocess_config=random_preprocess_config)
Molesol = fy.partial(
  ogb.Molesol,
  default_preprocess_config=random_preprocess_config)
Molfreesolv = fy.partial(
  ogb.Molfreesolv,
  default_preprocess_config=random_preprocess_config)


default = [
  "triangle_count",
  "Mollipo", "Molesol", "Molfreesolv",
  "ZINC"
]
