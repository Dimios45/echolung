#!/usr/bin/env python3
"""Comprehensive evaluation metrics for EchoLung.

Per-fold metrics (from CV prediction CSVs):
  - Accuracy, Balanced accuracy
  - AUC-ROC (macro OvR), AUC-PR (macro)
  - Cohen's Kappa, MCC
  - Sensitivity & Specificity per class
  - ECE (Expected Calibration Error)

Aggregated (across 5 folds):
  - Mean ± std for every metric above
  - 95% CI using t-distribution
  - Bootstrap 95% CI on mean accuracy

Statistical comparison (pretrained vs random-init):
  - Paired t-test
  - Wilcoxon signed-rank test
  - Cohen's d effect size

Usage:
    uv run python scripts/evaluate_metrics.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CV_PREDS = RESULTS / "cv_predictions"

TASKS = {
    "pocus": {
        "class_names": ["COVID", "Pneumonia", "Regular"],
        "n_classes": 3,
        "pretrained_dir": CV_PREDS / "pocus" / "pretrained",
        "randinit_dir": CV_PREDS / "pocus" / "randinit",
        "cv_summary_pretrained": RESULTS / "pocus_cv_summary.json",
        "cv_summary_randinit": RESULTS / "pocus_cv_randinit_summary.json",
    },
    "covid_blues": {
        "class_names": ["COVID-", "COVID+"],
        "n_classes": 2,
        "pretrained_dir": CV_PREDS / "covid_blues" / "pretrained",
        "randinit_dir": CV_PREDS / "covid_blues" / "randinit",
        "cv_summary_pretrained": RESULTS / "covid_blues_cv_summary.json",
        "cv_summary_randinit": RESULTS / "covid_blues_cv_randinit_summary.json",
    },
}
N_FOLDS = 5
N_BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ci95_t(values):
    """95% CI half-width via t-distribution (appropriate for n=5)."""
    n = len(values)
    se = stats.sem(values)
    h = se * stats.t.ppf(0.975, df=n - 1)
    return float(np.mean(values)), float(h)


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=0.95):
    """Bootstrap percentile 95% CI on the mean."""
    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return lo, hi


def cohens_d(a, b):
    """Cohen's d for paired samples."""
    diff = np.array(a) - np.array(b)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-9))


def sensitivity_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    result = {}
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        result[i] = {
            "sensitivity": round(tp / (tp + fn + 1e-9), 4),
            "specificity": round(tn / (tn + fp + 1e-9), 4),
        }
    return result


def expected_calibration_error(y_true, prob_matrix, n_bins=10):
    """ECE: weighted average bin-level |confidence - accuracy|."""
    confidences = prob_matrix.max(axis=1)
    predictions = prob_matrix.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = correct[mask].mean()
        ece += mask.sum() / n * abs(avg_conf - avg_acc)
    return float(ece)


def get_prob_matrix(df, n_classes):
    prob_cols = [f"prob_class_{c}" for c in range(n_classes)]
    if all(c in df.columns for c in prob_cols):
        return df[prob_cols].values.astype(float)
    # Binary fallback from confidence
    if n_classes == 2 and "prediction_confidence" in df.columns:
        conf = df["prediction_confidence"].values.astype(float)
        pred = df["predicted_class"].values.astype(int)
        prob1 = np.where(pred == 1, conf, 1.0 - conf)
        return np.column_stack([1.0 - prob1, prob1])
    return None


def compute_fold_metrics(df, class_names, n_classes):
    y_true = df["true_label"].values.astype(int)
    y_pred = df["predicted_class"].values.astype(int)
    prob_matrix = get_prob_matrix(df, n_classes)

    acc = float((y_true == y_pred).mean() * 100)
    bal_acc = float(balanced_accuracy_score(y_true, y_pred) * 100)
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    ss = sensitivity_specificity(y_true, y_pred, n_classes)

    m = {"accuracy": acc, "balanced_accuracy": bal_acc, "kappa": kappa, "mcc": mcc,
         "sensitivity": {class_names[i]: ss[i]["sensitivity"] for i in range(n_classes)},
         "specificity": {class_names[i]: ss[i]["specificity"] for i in range(n_classes)}}

    if prob_matrix is not None:
        try:
            if n_classes == 2:
                m["auc_roc"] = float(roc_auc_score(y_true, prob_matrix[:, 1]))
                m["auc_pr"] = float(average_precision_score(y_true, prob_matrix[:, 1]))
            else:
                m["auc_roc"] = float(roc_auc_score(y_true, prob_matrix, multi_class="ovr", average="macro"))
                per_ap = [average_precision_score((y_true == c).astype(int), prob_matrix[:, c])
                          for c in range(n_classes)]
                m["auc_pr"] = float(np.mean(per_ap))
                m["auc_pr_per_class"] = {class_names[c]: round(float(per_ap[c]), 4)
                                          for c in range(n_classes)}
            m["ece"] = expected_calibration_error(y_true, prob_matrix)
        except Exception as e:
            m["auc_roc_error"] = str(e)
    return m


def aggregate_metric(fold_values, key):
    vals = [f[key] for f in fold_values if key in f]
    if not vals:
        return {}
    mean, h = ci95_t(vals)
    lo_boot, hi_boot = bootstrap_ci(vals)
    return {
        "per_fold": [round(v, 4) for v in vals],
        "mean": round(mean, 4),
        "std": round(float(np.std(vals, ddof=1)), 4),
        "ci95_half_t": round(h, 4),
        "ci95_t": f"{mean:.2f} ± {h:.2f}",
        "ci95_boot": f"[{lo_boot:.2f}, {hi_boot:.2f}]",
    }


def load_fold_predictions(pred_dir, n_folds=N_FOLDS):
    folds = []
    for k in range(n_folds):
        p = pred_dir / f"fold{k}_predictions.csv"
        if p.exists():
            folds.append(pd.read_csv(p))
        else:
            folds.append(None)
    return folds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

all_results = {}

for task_name, cfg in TASKS.items():
    class_names = cfg["class_names"]
    n_classes = cfg["n_classes"]
    print(f"\n{'='*70}")
    print(f"  {task_name.upper()}")
    print(f"{'='*70}")

    task_results = {}

    for variant_label, pred_dir in [("pretrained", cfg["pretrained_dir"]),
                                     ("randinit", cfg["randinit_dir"])]:
        folds = load_fold_predictions(pred_dir)
        available = [i for i, f in enumerate(folds) if f is not None]

        if not available:
            print(f"\n  [{variant_label}] No per-fold predictions found at {pred_dir}")
            print(f"  Falling back to CV summary JSON for accuracy only.")
            summary_path = cfg[f"cv_summary_{variant_label}"]
            if summary_path.exists():
                with open(summary_path) as f:
                    cv = json.load(f)
                fold_accs = cv["per_fold_best_val_acc"]
                mean, h = ci95_t(fold_accs)
                lo_b, hi_b = bootstrap_ci(fold_accs)
                task_results[variant_label] = {
                    "accuracy": {
                        "per_fold": [round(a, 4) for a in fold_accs],
                        "mean": round(mean, 4),
                        "std": round(float(np.std(fold_accs, ddof=1)), 4),
                        "ci95_half_t": round(h, 4),
                        "ci95_t": f"{mean:.2f} ± {h:.2f}",
                        "ci95_boot": f"[{lo_b:.2f}, {hi_b:.2f}]",
                    }
                }
                print(f"  Accuracy: {mean:.2f}% ± {h:.2f}% (95% CI t)  |  boot [{lo_b:.2f}, {hi_b:.2f}]")
            continue

        print(f"\n  --- {variant_label} (folds: {available}) ---")
        fold_metrics = []
        for k in available:
            fm = compute_fold_metrics(folds[k], class_names, n_classes)
            fold_metrics.append(fm)
            auc_str = f"  AUC-ROC={fm['auc_roc']:.4f}" if "auc_roc" in fm else ""
            print(f"  Fold {k}: Acc={fm['accuracy']:.2f}%  BalAcc={fm['balanced_accuracy']:.2f}%"
                  f"  Kappa={fm['kappa']:.3f}  MCC={fm['mcc']:.3f}{auc_str}")

        # Aggregate all scalar metrics
        scalar_keys = ["accuracy", "balanced_accuracy", "kappa", "mcc", "auc_roc", "auc_pr", "ece"]
        variant_agg = {k: aggregate_metric(fold_metrics, k) for k in scalar_keys if any(k in f for f in fold_metrics)}

        # Sensitivity/specificity per class
        for cn in class_names:
            sens_vals = [f["sensitivity"].get(cn) for f in fold_metrics if cn in f.get("sensitivity", {})]
            spec_vals = [f["specificity"].get(cn) for f in fold_metrics if cn in f.get("specificity", {})]
            if sens_vals:
                variant_agg[f"sensitivity_{cn}"] = aggregate_metric(
                    [{f"sensitivity_{cn}": v} for v in sens_vals], f"sensitivity_{cn}")
            if spec_vals:
                variant_agg[f"specificity_{cn}"] = aggregate_metric(
                    [{f"specificity_{cn}": v} for v in spec_vals], f"specificity_{cn}")

        task_results[variant_label] = variant_agg

        # Print aggregated summary
        print(f"\n  AGGREGATED ({variant_label}):")
        for k in ["accuracy", "balanced_accuracy", "auc_roc", "auc_pr", "kappa", "mcc", "ece"]:
            if k in variant_agg:
                v = variant_agg[k]
                print(f"    {k:<22} {v['mean']:.4f} ± {v['std']:.4f}  95%CI_t={v['ci95_t']}  boot={v['ci95_boot']}")

    # Statistical comparison pretrained vs randinit
    pre_dir = cfg["pretrained_dir"]
    rnd_dir = cfg["randinit_dir"]
    pre_folds = load_fold_predictions(pre_dir)
    rnd_folds = load_fold_predictions(rnd_dir)
    paired_available = [k for k in range(N_FOLDS)
                        if pre_folds[k] is not None and rnd_folds[k] is not None]

    if len(paired_available) >= 2:
        pre_accs = [compute_fold_metrics(pre_folds[k], class_names, n_classes)["accuracy"]
                    for k in paired_available]
        rnd_accs = [compute_fold_metrics(rnd_folds[k], class_names, n_classes)["accuracy"]
                    for k in paired_available]
        t_stat, t_p = stats.ttest_rel(pre_accs, rnd_accs)
        if len(paired_available) >= 5:
            w_stat, w_p = stats.wilcoxon(pre_accs, rnd_accs)
        else:
            w_stat, w_p = float("nan"), float("nan")
        d = cohens_d(pre_accs, rnd_accs)
        task_results["significance"] = {
            "paired_t_stat": round(float(t_stat), 4),
            "paired_t_p": round(float(t_p), 4),
            "wilcoxon_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
            "wilcoxon_p": round(float(w_p), 4) if not np.isnan(w_p) else None,
            "cohens_d": round(d, 4),
            "mean_gain_%": round(float(np.mean(pre_accs) - np.mean(rnd_accs)), 2),
        }
        print(f"\n  SIGNIFICANCE (pretrained vs randinit, n={len(paired_available)} paired folds):")
        print(f"    Paired t-test:  t={t_stat:.3f}  p={t_p:.4f}  {'*' if t_p < 0.05 else 'n.s.'}")
        if not np.isnan(w_p):
            print(f"    Wilcoxon:       W={w_stat:.1f}   p={w_p:.4f}  {'*' if w_p < 0.05 else 'n.s.'}")
        print(f"    Cohen's d:      {d:.3f}  ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'})")
        print(f"    Mean gain:      +{np.mean(pre_accs) - np.mean(rnd_accs):.2f}%")
    elif len(paired_available) == 0:
        # Fall back to summary JSONs for significance
        pre_path = cfg["cv_summary_pretrained"]
        rnd_path = cfg["cv_summary_randinit"]
        if pre_path.exists() and rnd_path.exists():
            with open(pre_path) as f:
                pre_cv = json.load(f)
            with open(rnd_path) as f:
                rnd_cv = json.load(f)
            pre_accs = pre_cv["per_fold_best_val_acc"]
            rnd_accs = rnd_cv["per_fold_best_val_acc"]
            t_stat, t_p = stats.ttest_rel(pre_accs, rnd_accs)
            if len(pre_accs) >= 5:
                w_stat, w_p = stats.wilcoxon(pre_accs, rnd_accs)
            else:
                w_stat, w_p = float("nan"), float("nan")
            d = cohens_d(pre_accs, rnd_accs)
            task_results["significance"] = {
                "note": "from summary JSON (no per-fold prediction CSVs yet)",
                "paired_t_stat": round(float(t_stat), 4),
                "paired_t_p": round(float(t_p), 4),
                "wilcoxon_stat": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
                "wilcoxon_p": round(float(w_p), 4) if not np.isnan(w_p) else None,
                "cohens_d": round(d, 4),
                "mean_gain_%": round(float(np.mean(pre_accs) - np.mean(rnd_accs)), 2),
            }
            print(f"\n  SIGNIFICANCE (from JSON, no per-fold CSVs yet):")
            print(f"    Paired t-test:  t={t_stat:.3f}  p={t_p:.4f}  {'*' if t_p < 0.05 else 'n.s.'}")
            if not np.isnan(w_p):
                print(f"    Wilcoxon:       W={w_stat:.1f}   p={w_p:.4f}  {'*' if w_p < 0.05 else 'n.s.'}")
            print(f"    Cohen's d:      {d:.3f}  ({'large' if abs(d) > 0.8 else 'medium' if abs(d) > 0.5 else 'small'})")
            print(f"    Mean gain:      +{np.mean(pre_accs) - np.mean(rnd_accs):.2f}%")

    all_results[task_name] = task_results

# Save
out_path = RESULTS / "cv_metrics.json"
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n\nSaved to {out_path}")
