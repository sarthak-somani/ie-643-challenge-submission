#!/bin/bash
# Rebuild the final submission from the saved checkpoints (no training, about a minute on CPU).
# Run from the repository root, with the competition CSVs in data/.
set -e
PY=${PY:-python}
$PY src/predict.py --tau 1.25 --out submission/submission_from_checkpoints.csv --members $(awk '{print $1}' members.txt)
cmp -s submission/submission_from_checkpoints.csv submission/sub27_ens9_g0_tau1.25.csv \
  && echo "Identical to the submitted file submission/sub27_ens9_g0_tau1.25.csv" \
  || echo "WARNING: differs from submission/sub27_ens9_g0_tau1.25.csv"
