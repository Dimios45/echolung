#!/usr/bin/env bash
# Full evaluation pipeline: 4 CV experiments + aggregation + comprehensive metrics
# Expected runtime: ~6-8 hours on RTX 4090
set -euo pipefail

echo "============================================================"
echo " EchoLung Full Eval Pipeline"
echo " $(date)"
echo "============================================================"

# 1. POCUS pretrained
echo ""
echo "[1/4] POCUS 5-fold CV — pretrained EchoJEPA"
bash scripts/run_pocus_cv.sh

echo ""
echo "[2/4] POCUS 5-fold CV — random-init baseline"
bash scripts/run_pocus_cv_randinit.sh

echo ""
echo "[3/4] COVID-BLUES 5-fold CV — pretrained EchoJEPA"
bash scripts/run_covid_blues_cv.sh

echo ""
echo "[4/4] COVID-BLUES 5-fold CV — random-init baseline"
bash scripts/run_covid_blues_cv_randinit.sh

# 2. Aggregate accuracy
echo ""
echo "[5/6] Aggregating CV results"
uv run python scripts/aggregate_cv.py \
  --cv_dir experiments/eval/pocus_cv \
  --tag echojepa-vitl-pocus-cv \
  --output results/pocus_cv_summary.json

uv run python scripts/aggregate_cv.py \
  --cv_dir experiments/eval/pocus_cv_randinit \
  --tag randinit-vitl-pocus-cv \
  --output results/pocus_cv_randinit_summary.json

uv run python scripts/aggregate_cv.py \
  --cv_dir experiments/eval/covid_blues_cv \
  --tag echojepa-vitl-covid-blues-cv \
  --output results/covid_blues_cv_summary.json

uv run python scripts/aggregate_cv.py \
  --cv_dir experiments/eval/covid_blues_cv_randinit \
  --tag randinit-vitl-covid-blues-cv \
  --output results/covid_blues_cv_randinit_summary.json

# 3. Comprehensive metrics
echo ""
echo "[6/6] Computing comprehensive metrics"
uv run python scripts/evaluate_metrics.py

echo ""
echo "============================================================"
echo " Done: $(date)"
echo " Results: results/cv_metrics.json"
echo "============================================================"
