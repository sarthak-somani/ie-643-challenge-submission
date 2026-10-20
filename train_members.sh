#!/bin/bash
# Train all 9 ensemble members from scratch (5-fold CV each), then build the submission.
# One member after another: about 1.5 hours on CPU. Members are independent, so they can also run in parallel
#   PARALLEL=1 bash train_members.sh      (about 10 minutes with 36 CPU cores; each member uses 4 threads)
# Run from the repository root, with the competition CSVs in data/.
set -e
PY=${PY:-python}
export OMP_NUM_THREADS=4
mkdir -p logs
while read -r NAME ARGS; do
  [ -z "$NAME" ] && continue
  if [ -n "$PARALLEL" ]; then
    eval $PY src/cv.py --name $NAME $ARGS > logs/$NAME.log 2>&1 &
  else
    echo "== $NAME"; eval $PY src/cv.py --name $NAME $ARGS > logs/$NAME.log 2>&1
  fi
done < members.txt
wait
$PY src/ensemble.py --tau 1.25 --out submission/submission_retrained.csv --members $(awk '{print $1}' members.txt)
