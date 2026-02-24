#!/usr/bin/env python3
"""
Prepare COVID-BLUES lung ultrasound dataset for classification.

Two task modes:
    - Binary: COVID+ (cov_test=1) vs COVID- (cov_test=0) — patient-level labels
    - Severity: 4-class (scores 0, 1, 2, 3) — video-level labels

Patient-level stratified splitting to prevent data leakage (6 videos per patient).

Usage:
    python data/prepare_covid_blues.py \
        --blues_dir /home/sra/EchoJEPA/COVID-BLUES \
        --output_dir data/csv
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def main():
    parser = argparse.ArgumentParser(description="Prepare COVID-BLUES for classification")
    parser.add_argument("--blues_dir", type=str, required=True,
                        help="Path to COVID-BLUES root (contains severity.csv, lus_videos/, clinical_variables.csv)")
    parser.add_argument("--output_dir", type=str, default="data/csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=0,
                        help="If >0, generate K-fold CV splits for binary task (patient-level stratified)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_dir = os.path.join(args.blues_dir, "lus_videos")

    # Load severity labels (video-level)
    severity_df = pd.read_csv(os.path.join(args.blues_dir, "severity.csv"))
    severity_df["Severity Score"] = severity_df["Severity Score"].astype(int)
    severity_df["patient_id"] = severity_df["video_file"].str.extract(r"(patient_\d+)")
    severity_df["path"] = severity_df["video_file"].apply(
        lambda f: os.path.join(os.path.abspath(video_dir), f + ".mp4")
    )

    # Verify video files exist
    missing = severity_df[~severity_df["path"].apply(os.path.exists)]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} videos not found, dropping them")
        severity_df = severity_df[severity_df["path"].apply(os.path.exists)].copy()

    print(f"Total videos: {len(severity_df)}")
    print(f"Patients: {severity_df['patient_id'].nunique()}")
    print(f"Severity distribution:\n{severity_df['Severity Score'].value_counts().sort_index()}")

    # Load clinical variables for COVID status (patient-level)
    clinical_df = pd.read_csv(os.path.join(args.blues_dir, "clinical_variables.csv"))
    clinical_df["patient_id"] = "patient_" + clinical_df["patient_id"].astype(str)
    clinical_df = clinical_df[["patient_id", "cov_test"]].copy()
    clinical_df["cov_test"] = clinical_df["cov_test"].astype(int)

    print(f"\nCOVID status: {clinical_df['cov_test'].value_counts().sort_index().to_dict()}")

    # Merge COVID status into severity df
    severity_df = severity_df.merge(clinical_df, on="patient_id", how="left")

    # --- Patient-level split (60/20/20) ---
    patients = severity_df[["patient_id"]].drop_duplicates()

    # For binary split, stratify by COVID status
    patient_covid = severity_df[["patient_id", "cov_test"]].drop_duplicates()
    train_patients, temp_patients = train_test_split(
        patient_covid, test_size=0.4, random_state=args.seed,
        stratify=patient_covid["cov_test"]
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5, random_state=args.seed,
        stratify=temp_patients["cov_test"]
    )

    def assign_split(df, train_p, val_p, test_p):
        train = df[df["patient_id"].isin(train_p["patient_id"])]
        val = df[df["patient_id"].isin(val_p["patient_id"])]
        test = df[df["patient_id"].isin(test_p["patient_id"])]
        return train, val, test

    # --- Task 1: Severity (4-class) ---
    print("\n=== Severity (4-class) ===")
    for split_name, split_patients in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
        split_df = severity_df[severity_df["patient_id"].isin(split_patients["patient_id"])]
        out_path = os.path.join(args.output_dir, f"covid_blues_severity_{split_name}.csv")
        split_df[["path", "Severity Score"]].to_csv(out_path, sep=" ", header=False, index=False)
        print(f"{split_name}: {len(split_df)} videos, {split_df['patient_id'].nunique()} patients → {out_path}")

    # --- Task 2: Binary COVID (patient-level label applied to all videos) ---
    print("\n=== Binary COVID ===")
    for split_name, split_patients in [("train", train_patients), ("val", val_patients), ("test", test_patients)]:
        split_df = severity_df[severity_df["patient_id"].isin(split_patients["patient_id"])]
        out_path = os.path.join(args.output_dir, f"covid_blues_binary_{split_name}.csv")
        split_df[["path", "cov_test"]].to_csv(out_path, sep=" ", header=False, index=False)
        print(f"{split_name}: {len(split_df)} videos, {split_df['patient_id'].nunique()} patients → {out_path}")

    # --- Optional: K-Fold CV for binary task ---
    if args.cv_folds > 0:
        print(f"\n=== {args.cv_folds}-Fold CV (Binary COVID) ===")
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(patient_covid, patient_covid["cov_test"])):
            train_p = patient_covid.iloc[train_idx]
            val_p = patient_covid.iloc[val_idx]
            train_df = severity_df[severity_df["patient_id"].isin(train_p["patient_id"])]
            val_df = severity_df[severity_df["patient_id"].isin(val_p["patient_id"])]
            for split_name, split_df in [("train", train_df), ("val", val_df)]:
                out_path = os.path.join(args.output_dir, f"covid_blues_binary_fold{fold}_{split_name}.csv")
                split_df[["path", "cov_test"]].to_csv(out_path, sep=" ", header=False, index=False)
            print(f"Fold {fold}: train={len(train_df)} ({train_p['patient_id'].nunique()}p), "
                  f"val={len(val_df)} ({val_p['patient_id'].nunique()}p)")

    print("\nLabel mapping — Severity: 0/1/2/3, Binary: 0=COVID-, 1=COVID+")


if __name__ == "__main__":
    main()
