#!/usr/bin/env python3
"""Post-process test predictions: confusion matrices, classification reports, summary JSON."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

matplotlib.rcParams.update({"font.size": 12, "figure.dpi": 150})

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

TASKS = {
    "pocus_3class": {
        "pred_csv": RESULTS / "pocus_test_predictions.csv",
        "class_names": ["COVID", "Pneumonia", "Regular"],
        "title": "POCUS 3-Class Test Set",
    },
    "covid_blues_binary": {
        "pred_csv": RESULTS / "covid_blues_binary_test_predictions.csv",
        "class_names": ["COVID-", "COVID+"],
        "title": "COVID-BLUES Binary Test Set",
    },
    "covid_blues_severity": {
        "pred_csv": RESULTS / "covid_blues_severity_test_predictions.csv",
        "class_names": ["Score 0", "Score 1", "Score 2", "Score 3"],
        "title": "COVID-BLUES Severity Test Set",
    },
}

summary = {}

for task_name, cfg in TASKS.items():
    csv_path = cfg["pred_csv"]
    if not csv_path.exists():
        print(f"[SKIP] {task_name}: {csv_path} not found")
        continue

    df = pd.read_csv(csv_path)
    y_true = df["true_label"].values
    y_pred = df["predicted_class"].values
    class_names = cfg["class_names"]

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(f"\n{'='*60}")
    print(f"  {cfg['title']}")
    print(f"{'='*60}")
    print(report)

    # Metrics dict
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    acc = report_dict["accuracy"]
    summary[task_name] = {
        "accuracy": round(acc * 100, 1),
        "n_samples": len(df),
        "per_class": {
            cn: {
                "precision": round(report_dict[cn]["precision"], 3),
                "recall": round(report_dict[cn]["recall"], 3),
                "f1": round(report_dict[cn]["f1-score"], 3),
                "support": int(report_dict[cn]["support"]),
            }
            for cn in class_names
        },
    }

    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(cfg["title"])
    fig.tight_layout()
    out_path = FIGURES / f"cm_{task_name}.png"
    fig.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

# Save summary JSON
if summary:
    summary_path = RESULTS / "test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

print("\nDone.")
