import argparse
import funcy as fy

import graspe.evaluation.evaluate as evaluate
import graspe.evaluation.summary as summary
import graspe.evaluation.models as models
import graspe.evaluation.datasets as datasets

def quick_run(mf, dsm, **kwargs):

  evaluate.quick_evaluate(mf, dsm, **kwargs)

def run(mf, dsm, **kwargs):

  evaluate.evaluate(mf, dsm,  **kwargs)

def resume(mf, dsm, **kwargs):

  return evaluate.resume_evaluation(mf, dsm, **kwargs)

def summarize(mf, dsm, **kwargs):
  if dsm.default_split != 0:
    split = dsm.default_split
  else:
    split = None

  return summary.summarize_evaluation(
    evaluate.find_eval_dir(mf, dsm, split=split))

def epoch_times(mf, dsm, **kwargs):
  return evaluate.evaluate_epoch_time(mf, dsm)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Evaluate different model/dataset combinations.')
  parser.add_argument(
    "-d", "--dataset", action="append",
    help="Evaluate with this dataset.")
  parser.add_argument(
    "-m", "--model", action="append",
    help="Evaluate with this model.")
  parser.add_argument(
    "-q", "--quick", action="store_true",
    help="Only do a quick run to check whether there might be OOM issues.")
  parser.add_argument(
    "-t", "--time", action="store_true",
    help="Only measure the epoch times of the first hp config.")
  parser.add_argument(
    "-s", "--summarize", action="store_true",
    help="Only run the summarizer on existing evaluations.")
  parser.add_argument(
    "--dry", action="store_true",
    help="Only generate eval metadata without starting evaluation steps.")
  args = parser.parse_args()

  type = "evaluation"
  f = resume

  if args.quick:
    type = "quick evaluation"
    f = quick_run
  elif args.summarize:
    type = "summarization"
    f = summarize
  elif args.time:
    type = "epoch time measurement"
    f = epoch_times
  elif args.dry:
    type = "dry evaluation"
    f = fy.partial(resume, dry=True)

  if args.dataset is None or len(args.dataset) == 0:
    if args.time:
      ds = datasets.timing
    else:
      ds = datasets.default
  else:
    ds = args.dataset

  if args.model is None or len(args.model) == 0:
    if args.quick:
      ms = []
    else:
      ms = [
        "GIN", "WL2GNN",
        "DirectRankGIN", "DirectRankWL2GNN",
        "CmpGIN", "CmpWL2GNN",
        "WL_st_SVM"
      ]
  else:
    ms = args.model

  dsl = len(ds)
  msl = len(ms)

  print(f"Starting {type}...")
  print(f"Will use the following {dsl} datasets:")
  for d in ds:
    print(f"- {d}")
  print(f"Will use the following {msl} models:")
  for m in ms:
    print(f"- {m}")

  print()
  while True:
    a = input("Continue [Y/n]?\n")

    if a == "" or a == "Y" or a == "y" or a == "yes":
      break

    if a == "N" or a == "n" or a == "no":
      print("Canceled evaluation.")
      exit(0)

    print("Enter either y(es) or n(o).")

  print("----------------------------------------------------------\n")

  for m in ms:
    split_m = m.split("?", 1)

    if len(split_m) == 2:
      m, hp_i = split_m
      hp_i = int(hp_i)  # Evaluate only a single hp.
    else:
      hp_i = None  # Evaluate all hps.

    model = getattr(models, m)
    for d in ds:
      dsm = getattr(datasets, d)
      print(f"- Model: {m}, Dataset: {d}. -")
      f(model, dsm(), single_hp=hp_i)
      print("\n----------------------------------------------------------\n")

  print(f"Grid evaluation (# datasets = {dsl}, # models = {msl}) completed.")
