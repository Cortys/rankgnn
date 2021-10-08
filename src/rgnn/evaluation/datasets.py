import funcy as fy

import rgnn.datasets.synthetic.datasets as syn
import rgnn.datasets.tu.datasets as tu
import rgnn.datasets.ogb.datasets as ogb

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

triangle_count_size_extra = fy.partial(
  syn.triangle_count_dataset,
  outer_k=1,
  default_preprocess_config=base_preprocess_config,
  default_split="size_extrapolation")

triangle_count_count_extra = fy.partial(
  syn.triangle_count_dataset,
  outer_k=1,
  default_preprocess_config=base_preprocess_config,
  default_split="count_extrapolation")

# TU:
ZINC = fy.partial(
  tu.ZINC_full,
  in_memory_cache=False,
  default_preprocess_config={
    **random_preprocess_config,
    "batch_size_limit": 10000,
    "sample_ratio": 3})

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
