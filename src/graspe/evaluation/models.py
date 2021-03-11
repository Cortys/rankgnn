from graspe.utils import cart, cart_merge
import graspe.models.nn as nn
import graspe.models.gnn as gnn
import graspe.models.svm as svm

def model_factory(
  model_cls, prefer_in_enc=None, prefer_out_enc=None, ds_config=None):
  pie = prefer_in_enc
  poe = prefer_out_enc

  def dec(hp_f):
    class ModelFactory:
      name = hp_f.__name__
      prefer_in_enc = pie
      prefer_out_enc = poe
      config = ds_config or {}

      @staticmethod
      def get_model():
        return model_cls

      @staticmethod
      def get_hyperparams(enc, in_meta, out_meta, **hp_args):
        return [
          {"i": i, **c}
          for i, c in enumerate(hp_f(enc, in_meta, out_meta, **hp_args))]

    return ModelFactory
  return dec

def default_gnn_hyperparams(
  enc,
  layer_widths=[32, 64],
  layer_depths=[3, 5],
  learning_rates=[0.01, 0.001, 0.0001],
  pooling=["mean", "sum", "softmax"],
  **additional_params):
  in_enc, out_enc = enc[:2]
  edim = None
  fc_layer_args = [None]

  if "pref" in in_enc:
    fc_layer_args = [{-1: dict(activation=None)}]
  elif out_enc == "binary":
    edim = 1
  elif out_enc == "float":
    edim = 1
    fc_layer_args = [{-1: dict(activation=None)}]

  hidden = [
    ([b] * l, [b, b, edim or b])
    for b, l in cart(layer_widths, layer_depths)]

  hidden_hp = [dict(
    conv_layer_units=[*ch],
    att_conv_layer_units=[*ch],
    fc_layer_units=[*fh],
    cmp_layer_units=[fh[-1]]
  ) for ch, fh in hidden]

  return cart_merge(cart(
    conv_activation=["sigmoid"],
    att_conv_activation=["sigmoid"],
    fc_activation=["sigmoid"],
    fc_layer_args=fc_layer_args,
    conv_use_bias=[True],
    fc_use_bias=[True],
    pooling=pooling,
    learning_rate=learning_rates,
    **additional_params
  ), hidden_hp)

def default_nn_hyperparams(
  enc,
  layer_widths=[32, 64],
  layer_depths=[3, 5],
  learning_rates=[0.01, 0.001, 0.0001],
  **additional_params):
  in_enc, out_enc = enc
  edim = None
  fc_layer_args = [None]

  if "pref" in in_enc:
    fc_layer_args = [{-1: dict(activation=None)}]
  elif out_enc == "binary":
    edim = 1
  elif out_enc == "float":
    edim = 1
    fc_layer_args = [{-1: dict(activation=None)}]

  hidden = [
    ([b] * l, [b, b, edim or b])
    for b, l in cart(layer_widths, layer_depths)]

  hidden_hp = [dict(
    fc_layer_units=[*fh],
    cmp_layer_units=[fh[-1]]
  ) for ch, fh in hidden]

  return cart_merge(cart(
    fc_activation=["sigmoid"],
    fc_layer_args=fc_layer_args,
    fc_use_bias=[True],
    learning_rate=learning_rates,
    **additional_params
  ), hidden_hp)

def default_svm_hyperparams():
  return cart(C=[1, 0.1, 0.01, 0.001, 0.0001])

@model_factory(gnn.GCN)
def GCN(enc, in_meta, out_meta):
  return default_gnn_hyperparams(enc)

@model_factory(gnn.DirectRankGCN, )
def DirectRankGCN(enc, in_meta, out_meta):
  return default_gnn_hyperparams(enc)

@model_factory(gnn.CmpGCN)
def CmpGCN(enc, in_meta, out_meta):
  return default_gnn_hyperparams(enc)

@model_factory(gnn.GIN)
def GIN(enc, in_meta, out_meta):
  return default_gnn_hyperparams(enc)

@model_factory(gnn.DirectRankGIN, )
def DirectRankGIN(enc, in_meta, out_meta):
  return default_gnn_hyperparams(enc)

@model_factory(gnn.CmpGIN)
def CmpGIN(enc, in_meta, out_meta):
  return default_gnn_hyperparams(enc)

@model_factory(gnn.WL2GNN)
def WL2GNN(enc, in_meta, out_meta):
  return default_gnn_hyperparams(
    enc,
    conv_inner_activation=["relu"])

@model_factory(gnn.DirectRankWL2GNN)
def DirectRankWL2GNN(enc, in_meta, out_meta):
  return default_gnn_hyperparams(
    enc,
    conv_inner_activation=["relu"])

@model_factory(gnn.CmpWL2GNN)
def CmpWL2GNN(enc, in_meta, out_meta):
  return default_gnn_hyperparams(
    enc,
    conv_inner_activation=["relu"])

@model_factory(nn.MLP, prefer_in_enc="graph2vec")
def Graph2Vec_MLP(enc, in_meta, out_meta):
  return default_nn_hyperparams()

@model_factory(svm.SVM, prefer_in_enc="graph2vec")
def Graph2Vec_SVM(enc, in_meta, out_meta):
  return default_svm_hyperparams()

@model_factory(svm.KernelSVM, prefer_in_enc="wlst")
def WL_st_SVM(enc, in_meta, out_meta):
  return default_svm_hyperparams()

@model_factory(svm.KernelSVM, prefer_in_enc="wlsp")
def WL_sp_SVM(enc, in_meta, out_meta):
  return default_svm_hyperparams()
