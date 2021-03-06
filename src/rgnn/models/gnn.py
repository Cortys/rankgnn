import rgnn.chaining.model as cm
import rgnn.chaining.keras as ck
import rgnn.preprocessing.tf as tf_enc
from rgnn.models.nn import inputs, finalize, Dense
import rgnn.layers.wl1 as wl1
import rgnn.layers.wl2 as wl2
import rgnn.layers.pooling as pl
import rgnn.layers.pref as pref

import warnings
warnings.filterwarnings(
  "ignore",
  "Converting sparse IndexedSlices*",
  UserWarning)

@cm.model_step
def pool(input, pooling="mean"):
  if pooling == "mean":
    pool = pl.MeanPooling()
  elif pooling == "sum":
    pool = pl.SumPooling()
  elif pooling == "max":
    pool = pl.MaxPooling()
  elif pooling == "min":
    pool = pl.MinPooling()
  elif pooling == "softmax":
    pool = pl.SoftmaxPooling()
  else:
    raise AssertionError(f"Unknown pooling type '{pooling}'.")
  return pool(input)

@cm.model_step(macro=True)
def graph_embed(_, conv_layer, pooling=None):
  conv_layers = cm.with_layers(conv_layer, prefix="conv")

  if pooling == "softmax":
    att_conv_layers = cm.with_layers(conv_layer, prefix="att_conv")
    return [
      (conv_layers, att_conv_layers),
      cm.merge_ios(pl.merge_attention),
      pool]
  else:
    return [conv_layers, pool]

def index_selector(idx):
  @cm.model_step
  def selector(input):
    return input[idx]
  return selector


GCN = ck.create_model("GCN", [
  inputs,
  cm.with_layer(wl1.GCNPreprocessLayer),
  graph_embed(wl1.GCNLayer),
  cm.with_layers(Dense, prefix="fc"),
  finalize],
  input_encodings=["wl1"],
  output_encodings=tf_enc.output_encodings)

GIN = ck.create_model("GIN", [
  inputs,
  graph_embed(wl1.GINLayer),
  cm.with_layers(Dense, prefix="fc"),
  finalize],
  input_encodings=["wl1"],
  output_encodings=tf_enc.output_encodings)

WL2GNN = ck.create_model("WL2GNN", [
  inputs,
  graph_embed(wl2.WL2Layer),
  cm.with_layers(Dense, prefix="fc"),
  finalize],
  input_encodings=["wl2"],
  output_encodings=tf_enc.output_encodings)

def createDirectRankGNN(name, gnnLayer, enc, prepend=()):
  top = [
    inputs,
    *prepend,
    ([graph_embed(gnnLayer),
      cm.with_layers(Dense, prefix="fc")],
     index_selector("pref_a"), index_selector("pref_b")),
    cm.merge_ios,
    cm.with_layer(pref.PrefLookupLayer)]

  dr = ck.create_model(name, [
    *top,
    cm.with_layer(pref.PrefDiffLayer),
    cm.with_layer(Dense, units=1, activation="sigmoid", use_bias=False),
    finalize],
    input_encodings=[f"{enc}_pref"],
    output_encodings=["binary"])

  # For debugging and analysis purposes:
  dr.decomposed = ck.create_model(name, [
    *top,
    cm.with_layer(pref.PrefFirstLayer),
    cm.with_layer(Dense, units=1, activation=None, use_bias=False),
    finalize],
    input_encodings=[f"{enc}_pref"],
    output_encodings=["binary"])

  return dr

def createCmpGNN(name, gnnLayer, enc, prepend=()):
  return ck.create_model(name, [
    inputs,
    *prepend,
    ([graph_embed(gnnLayer),
      cm.with_layers(Dense, prefix="fc")],
     index_selector("pref_a"), index_selector("pref_b")),
    cm.merge_ios,
    cm.with_layer(pref.PrefLookupLayer),
    cm.with_layers(pref.CmpLayer, prefix="cmp"),
    cm.with_layer(pref.CmpLayer, units=1, prefix="cmp"),
    cm.with_layer(pref.PrefSigmoidDiffLayer),
    finalize],
    input_encodings=[f"{enc}_pref"],
    output_encodings=["binary"])


DirectRankGCN = createDirectRankGNN(
  "DirectRankGCN",
  wl1.GCNLayer, "wl1", [cm.with_layer(wl1.GCNPreprocessLayer)])
DirectRankGIN = createDirectRankGNN("DirectRankGIN", wl1.GINLayer, "wl1")
DirectRankWL2GNN = createDirectRankGNN("DirectRankWL2GNN", wl2.WL2Layer, "wl2")

CmpGCN = createCmpGNN(
  "CmpGCN", wl1.GCNLayer, "wl1", [cm.with_layer(wl1.GCNPreprocessLayer)])
CmpGIN = createCmpGNN("CmpGIN", wl1.GINLayer, "wl1")
CmpWL2GNN = createCmpGNN("CmpWL2GNN", wl2.WL2Layer, "wl2")
