#!/usr/bin/env python3
"""Generate confusion matrices for all test set predictions."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

CONFIGS = [
    {
        "name": "POCUS 3-Class",
        "csv": "pocus_test_predictions.csv",
        "labels": ["COVID", "Pneumonia", "Regular"],
        "filename": "cm_pocus_3class.pdf",
    },
    {
        "name": "COVID-BLUES Binary",
        "csv": "covid_blues_binary_test_predictions.csv",
        "labels": ["COVID-", "COVID+"],
        "filename": "cm_covid_blues_binary.pdf",
    },
    {
        "name": "COVID-BLUES Severity",
        "csv": "covid_blues_severity_test_predictions.csv",
        "labels": ["Score 0", "Score 1", "Score 2", "Score 3"],
        "filename": "cm_covid_blues_severity.pdf",
    },
]


def main():
    for cfg in CONFIGS:
        csv_path = os.path.join(RESULTS_DIR, cfg["csv"])
        if not os.path.exists(csv_path):
            print(f"Skipping {cfg['name']}: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        y_true = df["true_label"].astype(int).values
        y_pred = df["predicted_class"].astype(int).values
        labels = list(range(len(cfg["labels"])))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        acc = np.trace(cm) / cm.sum() * 100

        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm, display_labels=cfg["labels"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{cfg['name']}\nAccuracy: {acc:.1f}%")
        plt.tight_layout()

        out_path = os.path.join(FIGURES_DIR, cfg["filename"])
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path} ({cfg['name']}: {acc:.1f}%)")


if __name__ == "__main__":
    main()
