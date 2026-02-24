#!/usr/bin/env python3
"""Generate paper figures: bar chart comparing pretrained vs random-init, normalized confusion matrices."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_cv_summary(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_cv_comparison():
    """Bar chart: pretrained vs random-init across POCUS and COVID-BLUES."""
    datasets = []
    pretrained_means, pretrained_stds = [], []
    randinit_means, randinit_stds = [], []

    for label, pre_path, rand_path in [
        ("POCUS 3-Class", "pocus_cv_summary.json", "pocus_cv_randinit_summary.json"),
        ("COVID-BLUES Binary", "covid_blues_cv_summary.json", "covid_blues_cv_randinit_summary.json"),
    ]:
        pre = load_cv_summary(os.path.join(RESULTS_DIR, pre_path))
        rand = load_cv_summary(os.path.join(RESULTS_DIR, rand_path))
        if pre is None and rand is None:
            print(f"Skipping {label}: no results found")
            continue
        datasets.append(label)
        pretrained_means.append(pre["mean"] if pre else 0)
        pretrained_stds.append(pre["std"] if pre else 0)
        randinit_means.append(rand["mean"] if rand else 0)
        randinit_stds.append(rand["std"] if rand else 0)

    if not datasets:
        print("No CV results to plot.")
        return

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, pretrained_means, width, yerr=pretrained_stds,
                   label="EchoJEPA (pretrained)", color="#4C72B0", capsize=5)
    bars2 = ax.bar(x + width / 2, randinit_means, width, yerr=randinit_stds,
                   label="Random init", color="#DD8452", capsize=5)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("5-Fold CV: Pretrained vs Random Init")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 105)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "cv_pretrained_vs_randinit.pdf")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_normalized_confusion_matrices():
    """Normalized confusion matrices (percentages) for test predictions."""
    configs = [
        {
            "name": "POCUS 3-Class",
            "csv": "pocus_test_predictions.csv",
            "labels": ["COVID", "Pneumonia", "Regular"],
            "filename": "cm_pocus_3class_norm.pdf",
        },
        {
            "name": "COVID-BLUES Binary",
            "csv": "covid_blues_binary_test_predictions.csv",
            "labels": ["COVID-", "COVID+"],
            "filename": "cm_covid_blues_binary_norm.pdf",
        },
    ]

    for cfg in configs:
        csv_path = os.path.join(RESULTS_DIR, cfg["csv"])
        if not os.path.exists(csv_path):
            print(f"Skipping {cfg['name']}: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        y_true = df["true_label"].astype(int).values
        y_pred = df["predicted_class"].astype(int).values
        labels = list(range(len(cfg["labels"])))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        acc = np.trace(cm) / cm.sum() * 100

        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm_norm, display_labels=cfg["labels"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".1f")
        ax.set_title(f"{cfg['name']}\nAccuracy: {acc:.1f}%")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()

        out_path = os.path.join(FIGURES_DIR, cfg["filename"])
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path} ({cfg['name']}: {acc:.1f}%)")


def compile_paper_results():
    """Compile all results into a single JSON."""
    results = {}
    for key, path in [
        ("pocus_cv_pretrained", "pocus_cv_summary.json"),
        ("pocus_cv_randinit", "pocus_cv_randinit_summary.json"),
        ("covid_blues_cv_pretrained", "covid_blues_cv_summary.json"),
        ("covid_blues_cv_randinit", "covid_blues_cv_randinit_summary.json"),
        ("test_results", "test_summary.json"),
    ]:
        data = load_cv_summary(os.path.join(RESULTS_DIR, path))
        if data:
            results[key] = data

    out_path = os.path.join(RESULTS_DIR, "paper_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


def main():
    plot_cv_comparison()
    plot_normalized_confusion_matrices()
    compile_paper_results()


if __name__ == "__main__":
    main()
