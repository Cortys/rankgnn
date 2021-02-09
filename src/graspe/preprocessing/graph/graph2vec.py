# Code adapted from https://github.com/benedekrozemberczki/karateclub

import random
import numpy as np
import networkx as nx
import hashlib
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from typing import List, Dict

import graspe.preprocessing.encoder as encoder

default_dim = 128

class WeisfeilerLehmanHashing:
  """
  Weisfeiler-Lehman feature extractor class.
  Args:
    graph (NetworkX graph): NetworkX graph for which we do WL hashing.
    wl_iterations (int): Number of WL iterations.
    erase_base_feature (bool): Deleting the base features.
    node_attributes: List of node attribute names to be considered.
    edge_attributes: List of edge attribute names to be considered.
  """

  def __init__(
    self, graph: nx.classes.graph.Graph, wl_iterations: int,
    erase_base_features: bool,
    node_attributes=None, edge_attributes=None):
    """
    Initialization method which also executes feature extraction.
    """
    self.wl_iterations = wl_iterations
    self.graph = graph
    self.node_attributes = node_attributes
    self.edge_attributes = edge_attributes
    self.erase_base_features = erase_base_features
    self._set_features()
    self._do_recursions()

  def _set_features(self):
    """
    Creating the features.
    """
    g = self.graph
    if self.node_attributes:
      attrs = self.node_attributes
      self.features = {
        n: [d[a] for a in attrs]
        for n, d in g.nodes.items()}
    else:
      self.features = {
        node: g.degree(node)
        for node in g.nodes()}

    if self.edge_attributes and len(self.edge_attributes) > 0:
      eattrs = self.edge_attributes
      self.edge_features = {
        (v, u): str([g.edges[v, u][a] for a in eattrs])
        for v in g.nodes
        for u in g.neighbors(v)
        }
    else:
      self.edge_features = None

    self.extracted_features = {k: [str(v)] for k, v in self.features.items()}

  def _erase_base_features(self):
    """
    Erasing the base features
    """
    for k, v in self.extracted_features.items():
      del self.extracted_features[k][0]

  def _do_a_recursion(self):
    """
    The method does a single WL recursion.
    Return types:
      * **new_features** *(dict of strings)* - The hash table with WL features.
    """
    new_features = {}
    for node in self.graph.nodes():
      nebs = self.graph.neighbors(node)
      if self.edge_features:
        neb_fs = [
          (self.edge_features[(node, neb)], self.features[neb])
          for neb in nebs]
      else:
        neb_fs = [self.features[neb] for neb in nebs]
      features = [str(self.features[node])]+sorted([str(f) for f in neb_fs])
      features = "_".join(features)
      hash_object = hashlib.md5(features.encode())
      hashing = hash_object.hexdigest()
      new_features[node] = hashing
    self.extracted_features = {
      k: self.extracted_features[k] + [v]
      for k, v in new_features.items()}
    return new_features

  def _do_recursions(self):
    """
    The method does a series of WL recursions.
    """
    for _ in range(self.wl_iterations):
      self.features = self._do_a_recursion()
    if self.erase_base_features:
      self._erase_base_features()

  def get_node_features(self) -> Dict[int, List[str]]:
    """
    Return the node level features.
    """
    return self.extracted_features

  def get_graph_features(self) -> List[str]:
    """
    Return the graph level features.
    """
    return [
      feature
      for node, features in self.extracted_features.items()
      for feature in features]

class Graph2Vec:
  r"""An implementation of `"Graph2Vec" <https://arxiv.org/abs/1707.05005>`_
  from the MLGWorkshop '17 paper "Graph2Vec: Learning Distributed
  Representations of Graphs". The procedure creates Weisfeiler-Lehman tree
  features for nodes in graphs.
  Using these features a document (graph) - feature co-occurence matrix is
  decomposed in order to generate representations for the graphs.
  The procedure assumes that nodes have no string feature present and the
  WL-hashing defaults to the degree centrality. However, if a node feature with
  the key "feature" is supported for the nodes the feature extraction happens
  based on the values of this key.
  Args:
    wl_iterations (int): Number of Weisfeiler-Lehman iterations. Default is 2.
    node_attributes: Label names of vertex attributes. Default is None.
    edge_attributes: Label names of edge attributes. Default is None.
    dimensions (int): Dimensionality of embedding. Default is 128.
    workers (int): Number of cores. Default is 4.
    down_sampling (float): Down sampling frequency. Default is 0.0001.
    epochs (int): Number of epochs. Default is 10.
    learning_rate (float): HogWild! learning rate. Default is 0.025.
    min_count (int): Minimal count of graph feature occurrences. Default is 5.
    seed (int): Random seed for the model. Default is 42.
    erase_base_features (bool): Erasing the base features. Default is False.
  """

  def __init__(
    self, wl_iterations: int = 2,
    node_attributes=None, edge_attributes=None,
    dimensions: int = default_dim, workers: int = 4,
    down_sampling: float = 0.0001, epochs: int = 10,
    learning_rate: float = 0.025, min_count: int = 5,
    seed: int = 42, erase_base_features: bool = False):

    self.wl_iterations = wl_iterations
    self.node_attributes = node_attributes
    self.edge_attributes = edge_attributes
    self.dimensions = dimensions
    self.workers = workers
    self.down_sampling = down_sampling
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.min_count = min_count
    self.seed = seed
    self.erase_base_features = erase_base_features
    self._model = None

  def _graph_to_words(self, graph):
    wl_hashes = WeisfeilerLehmanHashing(
      graph, self.wl_iterations, self.erase_base_features,
      self.node_attributes, self.edge_attributes)

    return wl_hashes.get_graph_features()

  def fit(self, graphs: List[nx.classes.graph.Graph]):
    """
    Fitting a Graph2Vec model.
    Arg types:
      * **graphs** *(List of NetworkX graphs)* - The graphs to be embedded.
    """
    random.seed(self.seed)
    np.random.seed(self.seed)
    documents = [
      TaggedDocument(words=self._graph_to_words(g), tags=[str(i)])
      for i, g in enumerate(graphs)]

    model = Doc2Vec(
      documents,
      vector_size=self.dimensions,
      window=0,
      min_count=self.min_count,
      dm=0,
      sample=self.down_sampling,
      workers=self.workers,
      epochs=self.epochs,
      alpha=self.learning_rate,
      seed=self.seed)

    self._model = model
    self._embedding = [model.dv[str(i)] for i, _ in enumerate(documents)]

  def get_embedding(self, graphs=None) -> np.array:
    r"""Getting the embedding of graphs.
    Return types:
      * **embedding** *(Numpy array)* - The embedding of graphs.
    """
    assert self._model is not None, "Model not yet fitted."

    if graphs is None:
      return np.array(self._embedding)

    return np.array([
      self._model.infer_vector(self._graph_to_words(g))
      for g in graphs])

class Graph2VecEncoder(encoder.Encoder):
  name = "graph2vec"

  def __init__(
    self, T: int = 3, embedding_dim: int = default_dim,
    node_label_count=0, edge_label_count=0,
    node_feature_dim=0, edge_feature_dim=0,
    discrete_node_features=False, discrete_edge_features=False,
    ignore_node_features=False, ignore_node_labels=False,
    ignore_edge_features=False, ignore_edge_labels=False):
    self.name = f"graph2vec_T{T}_d{embedding_dim}"
    self.T = T
    self.embedding_dim = embedding_dim
    self.node_labels = node_label_count > 0 and not ignore_node_labels
    self.edge_labels = edge_label_count > 0 and not ignore_edge_labels
    self.node_features = (
      discrete_node_features and not ignore_node_features
      and node_feature_dim > 0)
    self.edge_features = (
      discrete_edge_features and not ignore_edge_features
      and edge_feature_dim > 0)
    self.node_attributes = []
    self.edge_attributes = []

    if self.node_features:
      self.node_attributes.append("features")
      self.name += "_nf"
    if self.node_labels:
      self.node_attributes.append("label")
    elif ignore_node_labels and node_label_count > 0:
      self.name += "_inl"

    if self.edge_features:
      self.edge_attributes.append("features")
      self.name += "_ef"
    if self.edge_labels:
      self.edge_attributes.append("label")
    elif ignore_edge_labels and edge_label_count > 0:
      self.name += "_iel"

  def preprocess(self, graphs):
    g2v = Graph2Vec(
      wl_iterations=self.T, dimensions=self.embedding_dim,
      node_attributes=self.node_attributes,
      edge_attributes=self.edge_attributes)
    g2v.fit(graphs)

    return g2v.get_embedding()

  def slice(self, embeddings, indices, train_indices=None):
    assert train_indices is not False, "Presplit data not yet supported."
    return super().slice(embeddings, indices, train_indices)

  def encode_element(self, graph):
    raise AssertionError(
      "Graph-by-graph transformation does not work for gram computation.")

  def transform(self, graphs):
    return self.preprocess(graphs)
