#!/usr/bin/env python3
"""Generate training curves and result figures for the EchoLung paper."""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({"font.size": 12, "figure.dpi": 150})

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"
OUT.mkdir(exist_ok=True)


def load_eval_csv(path):
    epochs, train, val = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                epochs.append(int(r["epoch"]))
                train.append(float(r["train_acc"]))
                val.append(float(r["val_acc"]))
            except ValueError:
                continue  # skip duplicate header rows
    return epochs, train, val


def load_pretrain_csv(path):
    epoch_losses = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            epoch_losses[int(r["epoch"])].append(float(r["loss"]))
    epochs = sorted(epoch_losses)
    avg = [sum(epoch_losses[e]) / len(epoch_losses[e]) for e in epochs]
    return epochs, avg


# --- Pretrain loss curve ---
pretrain_csv = ROOT / "experiments/pretrain/echonet_combined_vitl_224px_16f/log_r0.csv"
if pretrain_csv.exists():
    epochs, losses = load_pretrain_csv(pretrain_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSL Loss")
    ax.set_title("EchoJEPA SSL Pretraining (17.8K cardiac echo)")
    ax.annotate(f"{losses[0]:.3f}", (epochs[0], losses[0]), fontsize=10, ha="left")
    ax.annotate(f"{losses[-1]:.3f}", (epochs[-1], losses[-1]), fontsize=10, ha="right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pretrain_loss.png")
    print(f"Saved {OUT / 'pretrain_loss.png'}")
    plt.close()

# --- POCUS accuracy curve ---
pocus_csv = ROOT / "experiments/eval/pocus_classification/video_classification_frozen/echojepa-vitl-pocus-3class/log_r0.csv"
if pocus_csv.exists():
    epochs, train, val = load_eval_csv(pocus_csv)
    best_ep = epochs[val.index(max(val))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train, label="Train", color="#2563eb", linewidth=1.5)
    ax.plot(epochs, val, label="Val", color="#dc2626", linewidth=1.5)
    ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.5, label=f"Best val (ep {best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("POCUS 3-Class (COVID / Pneumonia / Regular)")
    ax.set_ylim(40, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pocus_accuracy.png")
    print(f"Saved {OUT / 'pocus_accuracy.png'}")
    plt.close()

# --- COVID-BLUES accuracy curve ---
blues_csv = ROOT / "experiments/eval/covid_blues_classification/video_classification_frozen/echojepa-vitl-covid-blues-binary/log_r0.csv"
if blues_csv.exists():
    epochs, train, val = load_eval_csv(blues_csv)
    best_ep = epochs[val.index(max(val))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train, label="Train", color="#2563eb", linewidth=1.5)
    ax.plot(epochs, val, label="Val", color="#dc2626", linewidth=1.5)
    ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.5, label=f"Best val (ep {best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("COVID-BLUES Binary (COVID+ / COVID−)")
    ax.set_ylim(40, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "covid_blues_accuracy.png")
    print(f"Saved {OUT / 'covid_blues_accuracy.png'}")
    plt.close()

# --- COVID-BLUES severity accuracy curve ---
severity_csv = ROOT / "experiments/eval/covid_blues_severity/video_classification_frozen/echojepa-vitl-covid-blues-severity/log_r0.csv"
if severity_csv.exists():
    epochs, train, val = load_eval_csv(severity_csv)
    best_ep = epochs[val.index(max(val))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train, label="Train", color="#2563eb", linewidth=1.5)
    ax.plot(epochs, val, label="Val", color="#dc2626", linewidth=1.5)
    ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.5, label=f"Best val (ep {best_ep})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("COVID-BLUES 4-Class Severity")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "covid_blues_severity_accuracy.png")
    print(f"Saved {OUT / 'covid_blues_severity_accuracy.png'}")
    plt.close()

# --- Combined summary bar chart (val) ---
fig, ax = plt.subplots(figsize=(5, 4))
tasks = ["POCUS 3-class\n(n=122)", "COVID-BLUES\nbinary (n=362)"]
accs = [100.0, 76.4]
colors = ["#2563eb", "#dc2626"]
bars = ax.bar(tasks, accs, color=colors, width=0.5, edgecolor="white")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{acc:.1f}%", ha="center", fontweight="bold", fontsize=13)
ax.set_ylabel("Best Val Accuracy (%)")
ax.set_title("Cross-Anatomy Transfer: Cardiac Echo → Lung US")
ax.set_ylim(0, 115)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "results_summary.png")
print(f"Saved {OUT / 'results_summary.png'}")
plt.close()

# --- Test accuracy bar chart (from test_summary.json) ---
test_summary_path = ROOT / "results" / "test_summary.json"
if test_summary_path.exists():
    with open(test_summary_path) as f:
        test_summary = json.load(f)
    task_labels = []
    test_accs = []
    bar_colors = ["#2563eb", "#dc2626", "#16a34a"]
    label_map = {
        "pocus_3class": "POCUS\n3-class",
        "covid_blues_binary": "COVID-BLUES\nbinary",
        "covid_blues_severity": "COVID-BLUES\nseverity",
    }
    for key in ["pocus_3class", "covid_blues_binary", "covid_blues_severity"]:
        if key in test_summary:
            task_labels.append(f"{label_map[key]}\n(n={test_summary[key]['n_samples']})")
            test_accs.append(test_summary[key]["accuracy"])
    if test_accs:
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(task_labels, test_accs, color=bar_colors[:len(test_accs)],
                       width=0.5, edgecolor="white")
        for bar, acc in zip(bars, test_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{acc:.1f}%", ha="center", fontweight="bold", fontsize=13)
        ax.set_ylabel("Test Accuracy (%)")
        ax.set_title("Test Set Results")
        ax.set_ylim(0, 115)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "test_results_summary.png")
        print(f"Saved {OUT / 'test_results_summary.png'}")
        plt.close()

print("\nAll figures saved to docs/assets/")
