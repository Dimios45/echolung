#!/usr/bin/env bash
# 5-Fold Stratified CV for POCUS 3-Class — Random Init Baseline (no pretraining)
set -euo pipefail

for k in 0 1 2 3 4; do
  echo "=== Fold ${k} (random-init) ==="
  OVERRIDE_TRAIN_DATA=data/csv/pocus_fold${k}_train.csv \
  OVERRIDE_VAL_DATA=data/csv/pocus_fold${k}_val.csv \
  uv run python -m evals.main \
    --fname configs/pocus/pocus_cv_randinit.yaml \
    --devices cuda:0 \
    --folder experiments/eval/pocus_cv_randinit/fold${k} \
    --override_config_folder
done

echo "=== All folds complete (random-init). ==="
