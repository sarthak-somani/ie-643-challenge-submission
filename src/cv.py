"""Train one ensemble member with 5-fold cross-validation on the 1284 labeled rows.

Each fold trains the teacher and 3 student seeds on 4/5 of the labeled rows (+ all unlabeled rows) and predicts
the held-out fold (out-of-fold, for the CV score) and the test set. The member's test probabilities are the
average of the 15 fold models (5 folds x 3 seeds).

Writes:
    checkpoints/<name>/fold<k>_seed<i>.pt, args.json    the 15 student models
    outputs/probs/<name>.npy                            (490, 11) test probabilities
    outputs/oof/<name>.npy                              (2076, 11) out-of-fold probabilities (labeled rows)
    outputs/cv_results.jsonl                            CV metrics (one line per member)

Example:
    python src/cv.py --name m01_base --cfg '{"kd_alpha": 0.3}'
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from data import load, N_CLASSES
from methods import train_member


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="member name (folder under checkpoints/)")
    ap.add_argument("--arch", default="mlp", choices=["mlp", "multitask", "multitask_cnn"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cfg", default="{}", help='member options as JSON, e.g. \'{"kd_alpha": 0.3}\'')
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    for folder in (f"checkpoints/{args.name}", "outputs/probs", "outputs/oof"):
        os.makedirs(folder, exist_ok=True)

    d = load()
    y = d["y"]
    labeled = np.where(y >= 0)[0]
    oof = np.zeros((len(y), N_CLASSES), np.float32)
    test = np.zeros((len(d["T"]), N_CLASSES), np.float32)

    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, (tr, va) in enumerate(folds.split(labeled, y[labeled])):
        lab, val = labeled[tr], labeled[va]
        P, models = train_member(d, lab, np.r_[d["X"][val], d["T"]], seeds=range(args.seeds), arch=args.arch, **cfg)
        oof[val] = P[:len(val)]
        test += P[len(val):] / 5
        for i, m in enumerate(models):
            torch.save(m.state_dict(), f"checkpoints/{args.name}/fold{k}_seed{i}.pt")
        print(f"fold {k} done", flush=True)

    np.save(f"outputs/oof/{args.name}.npy", oof)
    np.save(f"outputs/probs/{args.name}.npy", test)
    with open(f"checkpoints/{args.name}/args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # CV metrics. Prior adjustment: argmax(log p - tau * log prior), see ensemble.py.
    yt = y[labeled]
    prior = np.bincount(yt, minlength=N_CLASSES) / len(yt)
    res = dict(name=args.name, arch=args.arch, cfg=cfg, acc=round(float((oof[labeled].argmax(1) == yt).mean()), 4))
    for tau in (0, 0.5, 1.0, 1.25, 1.5):
        pred = (np.log(oof[labeled] + 1e-9) - tau * np.log(prior)).argmax(1)
        res[f"bal_acc_tau{tau}"] = round(balanced_accuracy_score(yt, pred), 4)
    print(json.dumps(res))
    with open("outputs/cv_results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
