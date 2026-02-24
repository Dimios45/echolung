#!/usr/bin/env python3
"""
Prepare EchoNet-Dynamic dataset for EchoJEPA probe evaluation.

Downloads (if needed) and converts EchoNet-Dynamic into the space-delimited
CSV format required by the training configs:
    <video_path> <z_score_normalized_lvef>

Usage:
    python data/prepare_echonet_dynamic.py \
        --echonet_dir /path/to/EchoNet-Dynamic \
        --output_dir data/csv

The script will:
    1. Read the FileList.csv from EchoNet-Dynamic
    2. Split into train/val/test using their official splits
    3. Z-score normalize LVEF (fit on train only)
    4. Write space-delimited CSVs for each split
    5. Save the scaler for later inference
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description="Prepare EchoNet-Dynamic for EchoJEPA")
    parser.add_argument("--echonet_dir", type=str, required=True,
                        help="Path to EchoNet-Dynamic root (contains Videos/ and FileList.csv)")
    parser.add_argument("--output_dir", type=str, default="data/csv",
                        help="Where to write the output CSVs")
    parser.add_argument("--video_subdir", type=str, default="Videos",
                        help="Subdirectory containing the .avi files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load the master file list
    filelist_path = os.path.join(args.echonet_dir, "FileList.csv")
    if not os.path.exists(filelist_path):
        raise FileNotFoundError(
            f"FileList.csv not found at {filelist_path}. "
            "Make sure --echonet_dir points to the EchoNet-Dynamic root."
        )

    df = pd.read_csv(filelist_path)
    print(f"Loaded {len(df)} entries from FileList.csv")

    # 2. Build video paths -- EchoNet-Dynamic uses .avi format
    video_dir = os.path.join(args.echonet_dir, args.video_subdir)
    df["video_path"] = df["FileName"].apply(
        lambda f: os.path.join(video_dir, f + ".avi") if not f.endswith(".avi") else os.path.join(video_dir, f)
    )

    # Use EF column for LVEF
    ef_col = "EF"
    if ef_col not in df.columns:
        # Try alternative column names
        for candidate in ["EF", "ef", "LVEF", "lvef", "Ejection Fraction"]:
            if candidate in df.columns:
                ef_col = candidate
                break
        else:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError("Cannot find EF/LVEF column in FileList.csv")

    # 3. Filter valid entries
    df = df.dropna(subset=[ef_col, "Split"])
    df = df[(df[ef_col] >= 0) & (df[ef_col] <= 100)].copy()
    print(f"After filtering: {len(df)} valid entries")

    # 4. Split using official splits
    train_df = df[df["Split"] == "TRAIN"].copy()
    val_df = df[df["Split"] == "VAL"].copy()
    test_df = df[df["Split"] == "TEST"].copy()

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 5. Z-score normalization (fit on train ONLY)
    scaler = StandardScaler()
    train_values = train_df[ef_col].values.reshape(-1, 1)
    scaler.fit(train_values)

    print(f"Scaler fitted -- Mean: {scaler.mean_[0]:.4f}, Std: {scaler.scale_[0]:.4f}")

    train_df["norm_ef"] = scaler.transform(train_df[ef_col].values.reshape(-1, 1))
    val_df["norm_ef"] = scaler.transform(val_df[ef_col].values.reshape(-1, 1))
    test_df["norm_ef"] = scaler.transform(test_df[ef_col].values.reshape(-1, 1))

    # 6. Write space-delimited CSVs (no header, no index)
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        output_path = os.path.join(args.output_dir, f"echonet_dynamic_{split_name}.csv")
        export = split_df[["video_path", "norm_ef"]].copy()
        export.to_csv(output_path, sep=" ", header=False, index=False)
        print(f"Wrote {len(export)} rows to {output_path}")

    # 7. Save the scaler
    scaler_path = os.path.join(args.output_dir, "echonet_dynamic_lvef_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    # 8. Print config values for reference
    print("\n" + "=" * 50)
    print("Add these to your eval config:")
    print(f"    target_mean: {scaler.mean_[0]:.4f}")
    print(f"    target_std:  {scaler.scale_[0]:.4f}")
    print("=" * 50)

    # 9. Verify a few lines
    sample_path = os.path.join(args.output_dir, "echonet_dynamic_train.csv")
    print(f"\nSample from {sample_path}:")
    with open(sample_path) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  {line.strip()}")


if __name__ == "__main__":
    main()
