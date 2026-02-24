#!/usr/bin/env bash
# 5-Fold Stratified CV for COVID-BLUES Binary Classification (pretrained)
set -euo pipefail

for k in 0 1 2 3 4; do
  echo "=== Fold ${k} ==="
  OVERRIDE_TRAIN_DATA=data/csv/covid_blues_binary_fold${k}_train.csv \
  OVERRIDE_VAL_DATA=data/csv/covid_blues_binary_fold${k}_val.csv \
  uv run python -m evals.main \
    --fname configs/pocus/covid_blues_binary_cv.yaml \
    --devices cuda:0 \
    --folder experiments/eval/covid_blues_cv/fold${k} \
    --override_config_folder
done

echo "=== All folds complete. Run: uv run python scripts/aggregate_cv.py --cv_dir experiments/eval/covid_blues_cv --tag echojepa-vitl-covid-blues-cv --output results/covid_blues_cv_summary.json ==="
