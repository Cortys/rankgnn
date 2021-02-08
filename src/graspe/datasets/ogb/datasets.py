from graspe.datasets.ogb.provider import ogb_dataset

disc = dict(
  discrete_node_features=True,
  discrete_edge_features=True)

# Regression:
Molesol = ogb_dataset("ogbg-molesol", type="float", **disc)
Molfreesolv = ogb_dataset("ogbg-molfreesolv", type="float", **disc)
Mollipo = ogb_dataset("ogbg-mollipo", type="float", **disc)
