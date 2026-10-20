"""Rebuild the submission from the saved checkpoints only (no training, about a minute on CPU).

Each member folder checkpoints/<name>/ holds 15 student models (fold<k>_seed<i>.pt) and args.json.
A member's test probabilities = mean over its 15 models; the submission is then built as in ensemble.py.

Example:
    python src/predict.py --tau 1.25 --out submission/from_checkpoints.csv --members m01_base m02_cosine_head ...
"""
import argparse
import glob
import json

import numpy as np
import torch

from data import load, N_CLASSES
from ensemble import write_submission
from methods import add_deform_feature
from models import build_model
from train import DEFAULTS, predict_proba


def member_proba(name, d):
    args = json.load(open(f"checkpoints/{name}/args.json"))
    cfg = {**DEFAULTS, **json.loads(args["cfg"])}
    arch = args["arch"] if args["arch"] in ("multitask", "multitask_cnn") else "multitask"
    T = d["T"]
    if cfg.get("deform_feature"):       # rebuild the predicted-deformation input feature (deterministic)
        _, T = add_deform_feature(d, T)
    probs = []
    for path in sorted(glob.glob(f"checkpoints/{name}/*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(arch, T.shape[1], N_CLASSES, cfg["hidden"], cfg["dropout"],
                            supp_dim=state["supp_head.weight"].shape[0], cos_scale=cfg["cos_head"])
        model.load_state_dict(state)
        probs.append(predict_proba(model, T))
    print(f"{name}: {len(probs)} models")
    return np.mean(probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--tau", type=float, default=1.25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = load()
    P = np.mean([member_proba(m, d) for m in args.members], axis=0)
    write_submission(P, args.tau, args.out)


if __name__ == "__main__":
    main()
