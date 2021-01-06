from graspe.datasets.tu.provider import tu_dataset

# Binary:
Mutag = tu_dataset("MUTAG", type="binary")
NCI1 = tu_dataset("NCI1", type="binary")
RedditBinary = tu_dataset("REDDIT-BINARY", type="binary")
Proteins = tu_dataset("PROTEINS_full", type="binary")
DD = tu_dataset("DD", type="binary")
IMDBBinary = tu_dataset("IMDB-BINARY", type="binary")

# Multiclass:
Reddit5K = tu_dataset("REDDIT-MULTI-5K", type="multiclass", classes=5)
IMDBMulti = tu_dataset("IMDB-MULTI", type="multiclass", classes=3)
