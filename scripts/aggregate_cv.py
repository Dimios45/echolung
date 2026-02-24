#!/usr/bin/env python3
"""Aggregate K-fold CV results for classification experiments."""

import argparse
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Aggregate K-fold CV results")
    parser.add_argument("--cv_dir", type=str, default="experiments/eval/pocus_cv",
                        help="Root directory containing fold0/, fold1/, ... subdirs")
    parser.add_argument("--tag", type=str, default="echojepa-vitl-pocus-cv",
                        help="Experiment tag (subfolder name under video_classification_frozen/)")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/pocus_cv_summary.json")
    args = parser.parse_args()

    best_accs = []
    for k in range(args.num_folds):
        log_path = os.path.join(args.cv_dir, f"fold{k}", "video_classification_frozen", args.tag, "log_r0.csv")
        if not os.path.exists(log_path):
            print(f"WARNING: {log_path} not found, skipping fold {k}")
            continue
        df = pd.read_csv(log_path)
        df = df[pd.to_numeric(df["epoch"], errors="coerce").notna()].apply(pd.to_numeric, errors="coerce")
        acc_col = [c for c in df.columns if "val" in c.lower() and "acc" in c.lower()]
        if not acc_col:
            acc_col = [df.columns[-1]]
        best_acc = df[acc_col[0]].max()
        best_accs.append(best_acc)
        print(f"Fold {k}: best val acc = {best_acc:.4f}")

    if not best_accs:
        print("No fold results found.")
        return

    mean_acc = sum(best_accs) / len(best_accs)
    std_acc = pd.Series(best_accs).std()
    print(f"\n{args.num_folds}-Fold CV: {mean_acc:.4f} ± {std_acc:.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary = {
        "num_folds": len(best_accs),
        "per_fold_best_val_acc": best_accs,
        "mean": round(mean_acc, 4),
        "std": round(std_acc, 4),
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
