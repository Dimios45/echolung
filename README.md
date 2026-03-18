# EchoLung

**Cross-anatomy transfer learning: cardiac echocardiography SSL to lung ultrasound classification**

Self-supervised representations learned from cardiac echo videos transfer effectively to lung ultrasound classification, even with very few labeled samples. We use EchoJEPA (V-JEPA2 ViT-L with RoPE) pretrained on 17,840 cardiac echo videos, then evaluate frozen attentive probes on two lung US benchmarks.

## Key Results

| Dataset | EchoJEPA (pretrained) | Random Init | Transfer Gain | p-value |
|---------|----------------------|-------------|---------------|---------|
| POCUS 3-class | **97.5% +/- 2.3%** | 73.3% +/- 7.6% | **+24.2%** | **0.0025** |
| COVID-BLUES binary | **75.2% +/- 5.3%** | 63.3% +/- 4.2% | **+11.9%** | **0.0054** |

*5-fold patient-level stratified cross-validation. Frozen ViT-L backbone + attentive probe. Paired t-test.*

![Pretrained vs Random Init](figures/cv_pretrained_vs_randinit.png)

![Per-Fold Results](figures/per_fold_results.png)

## Pipeline

```
V-JEPA 2 ViT-L (Meta, public weights)
  | continue SSL pretraining (100 epochs)
Domain-adapted EchoJEPA-L (17,840 cardiac echo videos)
  | freeze backbone + train attentive probe
Lung US classification (POCUS 3-class, COVID-BLUES binary)
```

## Datasets

### Pretraining (cardiac echo, unlabeled)

| Source | Videos | View |
|--------|--------|------|
| EchoNet-Dynamic | 10,030 | A4C |
| EchoNet-Pediatric A4C | 3,284 | A4C |
| EchoNet-Pediatric PSAX | 4,526 | PSAX |
| **Total** | **17,840** | |

### Downstream (lung US, labeled)

| Dataset | Videos | Patients | Task | Classes |
|---------|--------|----------|------|---------|
| POCUS convex | 122 | - | Pathology | COVID / Pneumonia / Regular |
| COVID-BLUES | 362 | 63 | COVID diagnosis | COVID+ / COVID- |

#### POCUS Samples

| COVID | Pneumonia | Regular |
|:-----:|:---------:|:-------:|
| ![COVID](figures/pocus_covid_sample.gif) | ![Pneumonia](figures/pocus_pneumonia_sample.gif) | ![Regular](figures/pocus_regular_sample.gif) |

#### COVID-BLUES Samples

| COVID+ | COVID- |
|:------:|:------:|
| ![COVID+](figures/blues_covidpos_sample.gif) | ![COVID-](figures/blues_covidneg_sample.gif) |

<details>
<summary>Static frame samples</summary>

![POCUS Samples](figures/pocus_samples.png)

![COVID-BLUES Samples](figures/covid_blues_samples.png)

</details>

## Cross-Validation Results

### POCUS 3-Class (COVID / Pneumonia / Regular)

| | Pretrained | Random Init |
|---|-----------|-------------|
| Fold 0 | 100.0% | 79.2% |
| Fold 1 | 95.8% | 70.8% |
| Fold 2 | 95.8% | 66.7% |
| Fold 3 | 95.8% | 83.3% |
| Fold 4 | 100.0% | 66.7% |
| **Mean +/- Std** | **97.5% +/- 2.3%** | **73.3% +/- 7.6%** |

### COVID-BLUES Binary (COVID+ / COVID-)

| | Pretrained | Random Init |
|---|-----------|-------------|
| Fold 0 | 70.3% | 60.8% |
| Fold 1 | 76.4% | 69.4% |
| Fold 2 | 75.0% | 64.5% |
| Fold 3 | 83.3% | 63.6% |
| Fold 4 | 70.8% | 58.3% |
| **Mean +/- Std** | **75.2% +/- 5.3%** | **63.3% +/- 4.2%** |

### Training Curves

![Training Curves](figures/training_curves.png)

### Test Set Confusion Matrices

Single train/val/test split evaluation on held-out test sets:

![POCUS Confusion Matrix](figures/cm_pocus_3class_norm.png)

![COVID-BLUES Confusion Matrix](figures/cm_covid_blues_binary_norm.png)

### Comprehensive Metrics (5-Fold CV)

| Metric | POCUS Pretrained | POCUS Randinit | COVID-BLUES Pretrained | COVID-BLUES Randinit |
|--------|-----------------|----------------|----------------------|----------------------|
| Accuracy | **97.5% ± 2.3%** | 73.3% ± 7.6% | **75.2% ± 5.3%** | 63.3% ± 4.2% |
| Balanced Accuracy | **93.8% ± 3.5%** | 42.7% ± 9.0% | **61.6% ± 14.3%** | 51.0% ± 2.3% |
| AUC-ROC | — | — | **0.683 ± 0.124** | 0.525 ± 0.071 |
| Cohen's Kappa | **0.896 ± 0.057** | 0.117 ± 0.109 | **0.233 ± 0.287** | 0.023 ± 0.055 |
| MCC | **0.899 ± 0.057** | 0.207 ± 0.190 | **0.238 ± 0.286** | 0.047 ± 0.116 |
| Paired t-test p | **0.0025** | — | **0.0054** | — |
| Cohen's d | **3.02 (large)** | — | **2.45 (large)** | — |

*AUC-ROC for POCUS requires per-class probabilities from the best probe head (pending full rerun with multi-class probability logging).*

### Negative Result: Severity Classification

COVID-BLUES 4-class severity (scores 0-3) achieved only **7.9% test accuracy** with the frozen probe, indicating that fine-grained severity distinctions are not captured by the frozen cardiac echo representations. This is expected: severity scoring requires subtle B-line quantification that differs substantially from the motion/structure patterns learned from echocardiography.

## SSL Pretraining

100 epochs on 17,840 cardiac echo videos (EchoNet-Dynamic + EchoNet-Pediatric).

- Architecture: V-JEPA2 ViT-L (24 blocks, 1024-dim, RoPE positional encoding)
- Input: 16 frames at 224x224, tubelet size 2x16x16
- Loss: 0.575 -> 0.492

## Setup

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

Symlink the pretrained checkpoint:
```bash
mkdir -p checkpoints
ln -s /path/to/vitl.pt checkpoints/vitl.pt
```

## Data Preparation

```bash
# 1. Combined pretraining pool
uv run python data/prepare_pretrain_combined.py \
    --dynamic_dir /path/to/EchoNet-Dynamic \
    --pediatric_dir /path/to/pediatric_echo_avi

# 2. POCUS lung US (3-class)
uv run python data/prepare_pocus.py \
    --pocus_dir /path/to/pocus_videos/convex

# 3. COVID-BLUES (binary + severity + 5-fold CV)
uv run python data/prepare_covid_blues.py \
    --blues_dir /path/to/COVID-BLUES \
    --cv_folds 5
```

## Reproducing Results

### Option A — Full pipeline in one command (~6-8h on RTX 4090)

```bash
bash scripts/run_full_eval.sh
```

This runs all 4 CV experiments, aggregates results, and computes comprehensive metrics.

### Option B — Step by step

#### 1. SSL Pretraining

```bash
uv run python -m app.main \
    --fname configs/4090/train/pretrain-echonet-224px-16f.yaml \
    --devices cuda:0
```

Checkpoint saved to `experiments/pretrain/echonet_combined_vitl_224px_16f/latest.pt`.

#### 2. 5-Fold Cross-Validation

```bash
# POCUS pretrained
bash scripts/run_pocus_cv.sh

# POCUS random-init baseline
bash scripts/run_pocus_cv_randinit.sh

# COVID-BLUES pretrained
bash scripts/run_covid_blues_cv.sh

# COVID-BLUES random-init baseline
bash scripts/run_covid_blues_cv_randinit.sh
```

Per-fold predictions are saved to `results/cv_predictions/` at the best validation epoch.

#### 3. Aggregate Accuracy

```bash
# POCUS pretrained
uv run python scripts/aggregate_cv.py \
    --cv_dir experiments/eval/pocus_cv \
    --tag echojepa-vitl-pocus-cv \
    --output results/pocus_cv_summary.json

# POCUS randinit
uv run python scripts/aggregate_cv.py \
    --cv_dir experiments/eval/pocus_cv_randinit \
    --tag randinit-vitl-pocus-cv \
    --output results/pocus_cv_randinit_summary.json

# COVID-BLUES pretrained
uv run python scripts/aggregate_cv.py \
    --cv_dir experiments/eval/covid_blues_cv \
    --tag echojepa-vitl-covid-blues-cv \
    --output results/covid_blues_cv_summary.json

# COVID-BLUES randinit
uv run python scripts/aggregate_cv.py \
    --cv_dir experiments/eval/covid_blues_cv_randinit \
    --tag randinit-vitl-covid-blues-cv \
    --output results/covid_blues_cv_randinit_summary.json
```

#### 4. Comprehensive Metrics

```bash
uv run python scripts/evaluate_metrics.py
```

Outputs `results/cv_metrics.json` with AUC-ROC/PR, balanced accuracy, Cohen's kappa, MCC, sensitivity/specificity, 95% CI, paired t-test, and Cohen's d for pretrained vs random-init.

#### 5. Generate Figures

```bash
uv run python scripts/generate_paper_figures.py
```

## Project Structure

```
configs/                  # YAML configs for pretraining and evaluation
data/                     # Data preparation scripts and CSV splits
  csv/                    # Train/val/test split CSVs and 5-fold splits
evals/                    # Evaluation framework (frozen probe)
figures/                  # Generated figures
results/                  # JSON summaries and prediction CSVs
  cv_predictions/         # Per-fold softmax predictions (saved at best epoch)
  cv_metrics.json   # Full metrics: AUC, kappa, MCC, CI, significance
  paper_results.json      # Authoritative results store
scripts/                  # Pipeline scripts
  run_full_eval.sh        # One-command full pipeline
  run_pocus_cv.sh         # POCUS 5-fold CV (pretrained)
  run_pocus_cv_randinit.sh
  run_covid_blues_cv.sh   # COVID-BLUES 5-fold CV (pretrained)
  run_covid_blues_cv_randinit.sh
  aggregate_cv.py         # Aggregate per-fold accuracy → JSON
  evaluate_metrics.py     # Comprehensive metrics suite
  generate_paper_figures.py
src/                      # Core model and training code
```

## Checkpoints

Pretrained checkpoints are available on [HuggingFace](https://huggingface.co/Dimios45/echolung).

## Citation

If you use this work, please cite:

- [V-JEPA 2](https://arxiv.org/abs/2412.04468) (Bardes et al., 2024)
- [EchoNet-Dynamic](https://echonet.github.io/dynamic/) (Ouyang et al., 2020)
- [POCUS Dataset](https://github.com/jannisborn/covid19_ultrasound) (Born et al., 2020)
- [COVID-BLUES](https://zenodo.org/records/6373057) (Roshankhah et al., 2022)
