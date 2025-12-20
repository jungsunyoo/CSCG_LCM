#!/usr/bin/env python3
"""
compare_runs.py

This script aggregates metrics from multiple training runs (baseline and CoDA),
computes mean ± SE of evaluation returns, and plots the mean ± SE bands of
Markovization metrics ("frac_deterministic" and "1-norm_entropy") across seeds.

Outputs:
  - mean ± SE of returns printed to console
  - comparison plot saved to runs/compare_markovization.png
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mean_se(values):
    """Return mean ± standard error for a list of numeric values."""
    values = np.asarray(values, dtype=float)
    return np.mean(values), np.std(values, ddof=1) / np.sqrt(len(values))


def read_eval_returns(run_dir):
    """Read evaluation return values from text files in a given run directory."""
    eval_files = glob.glob(os.path.join(run_dir, "*eval*.txt"))
    vals = []
    for f in eval_files:
        try:
            with open(f, "r") as fh:
                for line in fh:
                    for token in line.replace(",", " ").split():
                        try:
                            vals.append(float(token))
                        except ValueError:
                            continue
        except Exception:
            continue
    return vals


def load_metrics(pattern):
    """Load all metrics.csv files matching a pattern into a dict of {seed: DataFrame}."""
    runs = {}
    for path in sorted(glob.glob(pattern)):
        seed = os.path.basename(os.path.dirname(path))
        try:
            runs[seed] = pd.read_csv(path)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return runs


def aggregate_metric(runs, metric_name):
    """Compute mean ± SE arrays over aligned training steps for a given metric."""
    arrays = []
    for df in runs.values():
        if metric_name in df.columns:
            arrays.append(df[metric_name].to_numpy())
    if not arrays:
        return None, None, None
    min_len = min(map(len, arrays))
    arrays = np.stack([a[:min_len] for a in arrays])
    mean = arrays.mean(axis=0)
    se = arrays.std(axis=0, ddof=1) / np.sqrt(arrays.shape[0])
    x = np.arange(min_len)
    return x, mean, se


def plot_with_band(ax, x, mean, se, label, color):
    """Plot a mean curve with ±SE shaded region."""
    ax.plot(x, mean, color=color, label=label)
    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2)


def main():
    os.makedirs("runs", exist_ok=True)

    # Load runs
    baseline_runs = load_metrics("runs/ppo_baseline_*/metrics.csv")
    coda_runs = load_metrics("runs/ppo_coda_*/metrics.csv")

    # Compute return stats
    baseline_returns = []
    for d in glob.glob("runs/ppo_baseline_*"):
        baseline_returns.extend(read_eval_returns(d))
    coda_returns = []
    for d in glob.glob("runs/ppo_coda_*"):
        coda_returns.extend(read_eval_returns(d))

    b_mean, b_se = mean_se(baseline_returns) if baseline_returns else (np.nan, np.nan)
    c_mean, c_se = mean_se(coda_returns) if coda_returns else (np.nan, np.nan)

    print(f"Baseline returns: {b_mean:.3f} ± {b_se:.3f}")
    print(f"CoDA returns:     {c_mean:.3f} ± {c_se:.3f}")

    # Plot metrics
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    metrics = ["frac_deterministic", "1-norm_entropy"]

    for ax, metric in zip(axes, metrics):
        for label, runs, color in [
            ("Baseline", baseline_runs, "tab:blue"),
            ("CoDA", coda_runs, "tab:orange"),
        ]:
            x, mean, se = aggregate_metric(runs, metric)
            if x is not None:
                plot_with_band(ax, x, mean, se, label, color)
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Training steps")
    plt.tight_layout()

    out_path = "runs/compare_markovization.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()