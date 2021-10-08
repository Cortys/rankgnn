import rgnn.preprocessing.preprocessor as preproc
import rgnn.preprocessing.graph.kernel as gk
import rgnn.preprocessing.graph.graph2vec as g2v

def create_preprocessor(type, enc, in_encoder=None, out_encoder=None):
  class Preprocessor(preproc.Preprocessor):
    if in_encoder is not None:
      in_encoder_gen = in_encoder
    if out_encoder is not None:
      out_encoder_gen = out_encoder

  preproc.register_preprocessor(type, enc, Preprocessor)
  return Preprocessor

def create_graph_preprocessors(in_enc, in_encoder_gen, ):
  # Regression:
  create_preprocessor(
    ("graph", "integer"), (in_enc, "float", "scikit"),
    in_encoder_gen)
  create_preprocessor(
    ("graph", "float"), (in_enc, "float", "scikit"),
    in_encoder_gen)

  # Classification:
  create_preprocessor(
    ("graph", "binary"), (in_enc, "class", "scikit"),
    in_encoder_gen)
  create_preprocessor(
    ("graph", "integer"), (in_enc, "class", "scikit"),
    in_encoder_gen)


vector_input_encodings = ["graph2vec"]
kernel_input_encodings = ["wlst", "wlsp"]
svm_output_encodings = ["class", "float"]

create_graph_preprocessors(
  "graph2vec", g2v.Graph2VecEncoder)

create_graph_preprocessors(
  "wlst", lambda T=5,
  **config: gk.GrakelEncoder(
    f"gram_wlst_{T}", [dict(name="WL", n_iter=T), "VH"], **config))

create_graph_preprocessors(
  "wlsp", lambda T=5,
  **config: gk.GrakelEncoder(
    f"gram_wlsp_{T}", [dict(name="WL", n_iter=T), "SP"], **config))
