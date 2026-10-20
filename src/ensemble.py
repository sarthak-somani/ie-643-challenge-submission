"""Average the members' test probabilities and apply the prior adjustment -> submission CSV.

    label = argmax( log(mean_m p_m) - tau * log(prior) ),  prior = class frequencies of the labeled train rows.

tau > 0 removes the training prior from the decision (tau = 1.25 for the final submission).

Example:
    python src/ensemble.py --tau 1.25 --out submission/submission.csv --members m01_base m02_cosine_head ...
"""
import argparse
import os

import numpy as np
import pandas as pd

from data import load, N_CLASSES


def decide(P, prior, tau):
    return (np.log(P + 1e-9) - tau * np.log(prior)).argmax(1)


def write_submission(P, tau, out):
    d = load()
    y = d["y"][d["y"] >= 0]
    prior = np.bincount(y, minlength=N_CLASSES) / len(y)
    labels = decide(P, prior, tau)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame({"ID": d["test_ids"], "Label": labels}).to_csv(out, index=False)
    print("label counts:", np.bincount(labels, minlength=N_CLASSES).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="member names (outputs/probs/<name>.npy)")
    ap.add_argument("--tau", type=float, default=1.25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    P = np.mean([np.load(f"outputs/probs/{m}.npy") for m in args.members], axis=0)
    write_submission(P, args.tau, args.out)


if __name__ == "__main__":
    main()
