#!/usr/bin/env python3
"""
Learning curves + AUC (sample-efficiency) for Baseline vs CoDA.

Expected input layout (simple & flexible):
  runs/curves/
    baseline_seed0.txt   # one float (return) per line, one line per episode
    baseline_seed1.txt
    ...
    coda_seed0.txt
    coda_seed1.txt
    ...

Usage:
  python learning_curves_auc.py --curves_dir runs/curves --out runs/curves

Outputs:
  runs/curves/curve_summary.json    # means, SEs, AUCs, CIs
  runs/curves/mean_curves.png       # mean ± SEM curves
  runs/curves/auc_scatter.png       # per-seed AUC scatter (paired if seed ids match)
"""
import argparse, glob, json, os, re
from pathlib import Path

import numpy as np

# Matplotlib is optional; script still prints/outputs JSON if not available.
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

def _load_curves(glob_pat):
    curves = []
    seeds = []
    for fp in sorted(glob.glob(glob_pat)):
        try:
            xs = np.loadtxt(fp, dtype=float).reshape(-1)
            # Filter NaNs safely; keep alignment
            xs = xs[np.isfinite(xs)]
            curves.append(xs)
            m = re.search(r"seed(\d+)", os.path.basename(fp))
            seeds.append(int(m.group(1)) if m else len(seeds))
        except Exception:
            continue
    return seeds, curves

def _align(curves_a, curves_b, truncate_to_min=True):
    """
    Returns two arrays [n, T] aligned in time.
    truncate_to_min=True: trims all seeds to min length across both groups (fair comparison).
    """
    if not curves_a or not curves_b:
        return None, None
    lengths = [len(x) for x in curves_a] + [len(y) for y in curves_b]
    T = min(lengths) if truncate_to_min else max(lengths)
    def pad_to_T(x):
        z = np.full(T, np.nan, dtype=float)
        z[:min(T, len(x))] = x[:min(T, len(x))]
        return z
    A = np.stack([pad_to_T(x) for x in curves_a], axis=0)
    B = np.stack([pad_to_T(y) for y in curves_b], axis=0)
    if truncate_to_min:
        A = A[:, :T]
        B = B[:, :T]
    return A, B

def _auc(x):
    # AUC via trapezoid (ignores NaNs at tail safely if present)
    x = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(x)): return np.nan
    valid = np.isfinite(x)
    if valid.sum() < 2: return np.nan
    idx = np.where(valid)[0]
    return float(np.trapz(x[idx], x=idx))

def _bootstrap_mean(vals, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    vals = vals[ok]
    if len(vals) == 0: return np.nan, (np.nan, np.nan)
    boot = []
    for _ in range(B):
        idx = rng.integers(0, len(vals), size=len(vals))
        boot.append(np.mean(vals[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.mean(vals)), (float(lo), float(hi))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves_dir", type=str, default="runs/curves")
    ap.add_argument("--baseline_glob", type=str, default="baseline_seed*.txt")
    ap.add_argument("--coda_glob", type=str, default="coda_seed*.txt")
    ap.add_argument("--truncate_to_min", action="store_true", default=True)
    ap.add_argument("--out", type=str, default="runs/curves")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    seeds_b, curves_b = _load_curves(str(Path(args.curves_dir, args.baseline_glob)))
    seeds_c, curves_c = _load_curves(str(Path(args.curves_dir, args.coda_glob)))

    if not curves_b or not curves_c:
        raise SystemExit("No curve files found. See docstring for expected layout.")

    # Pair by seed id if possible
    seedset = sorted(set(seeds_b).intersection(seeds_c))
    if seedset:
        b_use = [curves_b[seeds_b.index(s)] for s in seedset]
        c_use = [curves_c[seeds_c.index(s)] for s in seedset]
        pairing = "paired"
    else:
        b_use, c_use = curves_b, curves_c
        pairing = "unpaired"

    A, C = _align(b_use, c_use, truncate_to_min=args.truncate_to_min)
    # [n, T]
    mean_b = np.nanmean(A, axis=0); se_b = np.nanstd(A, axis=0, ddof=1)/np.sqrt(A.shape[0])
    mean_c = np.nanmean(C, axis=0); se_c = np.nanstd(C, axis=0, ddof=1)/np.sqrt(C.shape[0])

    auc_b = np.array([_auc(x) for x in A])
    auc_c = np.array([_auc(x) for x in C])
    auc_b_mean, auc_b_ci = _bootstrap_mean(auc_b)
    auc_c_mean, auc_c_ci = _bootstrap_mean(auc_c)

    # Paired difference if seeds paired
    if pairing == "paired":
        diffs = auc_c - auc_b
        auc_diff_mean, auc_diff_ci = _bootstrap_mean(diffs)
    else:
        # Welch-style bootstrap on means
        auc_diff_mean = auc_c_mean - auc_b_mean
        auc_diff_ci = (np.nan, np.nan)

    summary = dict(
        pairing=pairing,
        n_baseline=int(A.shape[0]),
        n_coda=int(C.shape[0]),
        horizon=int(A.shape[1]),
        auc_baseline=dict(mean=auc_b_mean, ci95=auc_b_ci),
        auc_coda=dict(mean=auc_c_mean, ci95=auc_c_ci),
        auc_diff=dict(mean=auc_diff_mean, ci95=auc_diff_ci),
    )
    Path(args.out, "curve_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if HAVE_PLT:
        # mean curves ± SEM
        plt.figure()
        x = np.arange(len(mean_b))
        plt.plot(x, mean_b, label="Baseline")
        plt.fill_between(x, mean_b-se_b, mean_b+se_b, alpha=0.25)
        plt.plot(x, mean_c, label="CoDA")
        plt.fill_between(x, mean_c-se_c, mean_c+se_c, alpha=0.25)
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.title("Learning curves (mean ± SEM)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.out, "mean_curves.png"), dpi=150)
        plt.close()

        # AUC scatter (paired if seed ids match)
        plt.figure()
        if pairing == "paired":
            plt.scatter(auc_b, auc_c)
            L = [np.nanmin([auc_b.min(), auc_c.min()]), np.nanmax([auc_b.max(), auc_c.max()])]
            plt.plot(L, L, linestyle="--")
            plt.xlabel("Baseline AUC per seed")
            plt.ylabel("CoDA AUC per seed")
            plt.title("AUC (paired by seed)")
        else:
            # jitter for display
            jitter = 0.05*(np.nanstd(np.concatenate([auc_b, auc_c]))+1e-9)
            plt.scatter(np.zeros_like(auc_b)+0.0+jitter*np.random.randn(len(auc_b)), auc_b, label="Baseline")
            plt.scatter(np.zeros_like(auc_c)+1.0+jitter*np.random.randn(len(auc_c)), auc_c, label="CoDA")
            plt.xticks([0,1], ["Baseline","CoDA"])
            plt.ylabel("AUC")
            plt.title("AUC (unpaired)")
            plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.out, "auc_scatter.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
