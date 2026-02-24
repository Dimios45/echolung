#!/usr/bin/env python3
"""
Prepare combined cardiac echo pretraining dataset (SSL, no labels needed).

Merges:
    - EchoNet-Dynamic (10,030 videos)
    - EchoNet-Pediatric A4C (3,284 videos)
    - EchoNet-Pediatric PSAX (4,526 videos)

All videos get label=0 (unused in SSL pretraining, but required by VideoDataset format).

Usage:
    python data/prepare_pretrain_combined.py \
        --dynamic_dir /home/sra/EchoJEPA/EchoNet/EchoNet-Dynamic \
        --pediatric_dir /home/sra/EchoJEPA/echonetpediatric/echonetpediatric/pediatric_echo_avi/pediatric_echo_avi \
        --output_dir data/csv
"""

import argparse
import os

import pandas as pd


def collect_echonet_dynamic(echonet_dir: str) -> list[str]:
    """Collect all .avi paths from EchoNet-Dynamic/Videos/."""
    video_dir = os.path.join(echonet_dir, "Videos")
    paths = []
    for fname in sorted(os.listdir(video_dir)):
        if fname.endswith(".avi"):
            paths.append(os.path.join(os.path.abspath(video_dir), fname))
    return paths


def collect_echonet_pediatric(pediatric_dir: str, view: str) -> list[str]:
    """Collect all .avi paths from EchoNet-Pediatric/{view}/Videos/."""
    video_dir = os.path.join(pediatric_dir, view, "Videos")
    paths = []
    for fname in sorted(os.listdir(video_dir)):
        if fname.endswith(".avi"):
            paths.append(os.path.join(os.path.abspath(video_dir), fname))
    return paths


def write_csv(paths: list[str], output_path: str):
    """Write space-delimited CSV: <path> <label=0>."""
    df = pd.DataFrame({"path": paths, "label": 0})
    df.to_csv(output_path, sep=" ", header=False, index=False)
    print(f"Wrote {len(df)} rows → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare combined pretrain dataset")
    parser.add_argument("--dynamic_dir", type=str, required=True,
                        help="Path to EchoNet-Dynamic root")
    parser.add_argument("--pediatric_dir", type=str, required=True,
                        help="Path to EchoNet-Pediatric root (containing A4C/ and PSAX/)")
    parser.add_argument("--output_dir", type=str, default="data/csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect all paths
    dynamic_paths = collect_echonet_dynamic(args.dynamic_dir)
    a4c_paths = collect_echonet_pediatric(args.pediatric_dir, "A4C")
    psax_paths = collect_echonet_pediatric(args.pediatric_dir, "PSAX")

    print(f"EchoNet-Dynamic: {len(dynamic_paths)}")
    print(f"Pediatric A4C: {len(a4c_paths)}")
    print(f"Pediatric PSAX: {len(psax_paths)}")

    pediatric_all = a4c_paths + psax_paths
    combined_all = dynamic_paths + pediatric_all

    print(f"Combined total: {len(combined_all)}")

    # Write CSVs
    write_csv(combined_all, os.path.join(args.output_dir, "pretrain_combined.csv"))
    write_csv(dynamic_paths, os.path.join(args.output_dir, "pretrain_dynamic_only.csv"))
    write_csv(pediatric_all, os.path.join(args.output_dir, "pretrain_pediatric_only.csv"))


if __name__ == "__main__":
    main()
