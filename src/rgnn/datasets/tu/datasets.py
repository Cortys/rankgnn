import numpy as np

from rgnn.datasets.tu.provider import tu_dataset, presplit_tu_dataset

# Binary:
Mutag = tu_dataset("MUTAG", type="binary")
NCI1 = tu_dataset("NCI1", type="binary")
RedditBinary = tu_dataset("REDDIT-BINARY", type="binary")
Proteins = tu_dataset("PROTEINS_full", type="binary")
DD = tu_dataset("DD", type="binary")
IMDBBinary = tu_dataset("IMDB-BINARY", type="binary")

# Multiclass:
Reddit5K = tu_dataset("REDDIT-MULTI-5K", type="integer", min=1, max=5)
IMDBMulti = tu_dataset("IMDB-MULTI", type="integer", min=1, max=3)

# Regression:
rng = np.random.default_rng(42)
TRIANGLES = tu_dataset(
  "TRIANGLES", type="integer",
  default_split=(
    rng.permutation(30000),
    rng.permutation(5000) + 30000,
    rng.permutation(10000) + 35000),
  discrete_node_features=True,  # nodes are annotated with triangle counts
  default_preprocess_config=dict(ignore_node_features=True))  # ignore counts

rng = np.random.default_rng(42)
ZINC_full = tu_dataset(
  "ZINC_full", type="float",
  default_split=(
    rng.permutation(220011),
    rng.permutation(24445) + 225011,
    rng.permutation(5000) + 220011))

ZINC_presplit = presplit_tu_dataset(
  "ZINC", "ZINC_train", "ZINC_val", "ZINC_test",
  type="float")
