#!/usr/bin/env python3
"""
Prepare POCUS convex lung ultrasound dataset for classification.

Parses video filenames to assign labels:
    Cov_* / Cov-* → 0 (COVID)
    Pneu_* / Pneu-* / pneu-* → 1 (Pneumonia)
    Reg_* / Reg-* → 2 (Regular/Healthy)
    Vir_* → dropped (only 2 samples)

Handles mixed formats: .avi, .gif, .mov, .mp4, .mpeg

Usage:
    python data/prepare_pocus.py \
        --pocus_dir /home/sra/EchoJEPA/covid19_ultrasound/data/pocus_videos/convex \
        --output_dir data/csv
"""

import argparse
import os
import re

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


LABEL_MAP = {
    "cov": 0,
    "pneu": 1,
    "reg": 2,
}

VIDEO_EXTENSIONS = {".avi", ".mov", ".mp4", ".mpeg"}  # GIFs pre-converted to mp4


def classify_filename(fname: str) -> int | None:
    """Return integer label from filename prefix, or None to skip."""
    lower = fname.lower()
    for prefix, label in LABEL_MAP.items():
        if lower.startswith(prefix):
            return label
    return None  # e.g. Vir_*


def main():
    parser = argparse.ArgumentParser(description="Prepare POCUS convex for classification")
    parser.add_argument("--pocus_dir", type=str, required=True,
                        help="Path to pocus_videos/convex/")
    parser.add_argument("--output_dir", type=str, default="data/csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kfold", type=int, default=0,
                        help="If >0, generate K-fold stratified CV splits instead of train/val/test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    skipped = []
    for fname in sorted(os.listdir(args.pocus_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            continue
        label = classify_filename(fname)
        if label is None:
            skipped.append(fname)
            continue
        rows.append({
            "path": os.path.join(os.path.abspath(args.pocus_dir), fname),
            "label": label,
        })

    df = pd.DataFrame(rows)
    print(f"Total videos: {len(df)} (skipped {len(skipped)}: {skipped})")
    print(f"Class distribution:\n{df['label'].value_counts().sort_index()}")

    if args.kfold > 0:
        # K-fold stratified cross-validation splits
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        for k, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            for split_name, split_df in [("train", train_df), ("val", val_df)]:
                out_path = os.path.join(args.output_dir, f"pocus_fold{k}_{split_name}.csv")
                split_df[["path", "label"]].to_csv(out_path, sep=" ", header=False, index=False)
                print(f"Fold {k} {split_name}: {len(split_df)} rows → {out_path}")
                print(f"  {split_df['label'].value_counts().sort_index().to_dict()}")
        print(f"\nGenerated {args.kfold} folds. Label mapping: Cov=0, Pneu=1, Reg=2")
    else:
        # Stratified train/val/test split: 60/20/20
        train_df, temp_df = train_test_split(
            df, test_size=0.4, random_state=args.seed, stratify=df["label"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df["label"]
        )

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            out_path = os.path.join(args.output_dir, f"pocus_{split_name}.csv")
            split_df[["path", "label"]].to_csv(out_path, sep=" ", header=False, index=False)
            print(f"{split_name}: {len(split_df)} rows → {out_path}")
            print(f"  {split_df['label'].value_counts().sort_index().to_dict()}")

        print("\nLabel mapping: Cov=0, Pneu=1, Reg=2")


if __name__ == "__main__":
    main()
