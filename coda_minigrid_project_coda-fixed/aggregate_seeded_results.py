#!/usr/bin/env python3
# Usage:
#   python aggregate_seeded_results.py
#
# Reads all 'runs/seeded/compare_returns_seed*.txt' files and extracts
# 'Baseline mean ± SE: X ± Y' and 'CoDA mean ± SE: X ± Y' lines per seed.
# Writes a consolidated 'runs/compare_returns.txt' in the format:
#   seed K baseline B coda C
# and prints the aggregated mean ± SE across seeds.
#
import re, glob, numpy as np
from pathlib import Path

seed_files = sorted(glob.glob('runs/seeded/compare_returns_seed*.txt'))
if not seed_files:
    raise SystemExit('No per-seed files found in runs/seeded/. Did you run run_many_seeds.py?')

baseline_vals, coda_vals = [], []
Path('runs').mkdir(exist_ok=True, parents=True)
with open('runs/compare_returns.txt', 'w') as out:
    for fp in seed_files:
        text = Path(fp).read_text()
        mB = re.search(r'Baseline\s+mean\s*[±+/-]\s*SE:\s*([-+]?\d*\.\d+|\d+)', text, flags=re.I)
        mC = re.search(r'CoDA\s+mean\s*[±+/-]\s*SE:\s*([-+]?\d*\.\d+|\d+)', text, flags=re.I)
        seed_match = re.search(r'(\d+)', Path(fp).stem)  # extract from filename
        seed = int(seed_match.group(1)) if seed_match else len(baseline_vals)
        if not (mB and mC):
            print(f'WARNING: Could not parse baseline/coda means from {fp}')
            continue
        b = float(mB.group(1)); c = float(mC.group(1))
        baseline_vals.append(b); coda_vals.append(c)
        out.write(f'seed {seed} baseline {b:.6f} coda {c:.6f}\n')

def mean_se(xs):
    xs = np.asarray(xs, float)
    n = np.isfinite(xs).sum()
    m = float(np.nanmean(xs))
    se = float(np.nanstd(xs, ddof=1)/np.sqrt(n)) if n > 1 else float('nan')
    return m, se

mb, seb = mean_se(baseline_vals)
mc, sec = mean_se(coda_vals)
print(f'Aggregated over {len(baseline_vals)} seeds.')
print(f'Baseline mean ± SE: {mb:.3f} ± {seb:.3f}')
print(f'CoDA mean ± SE: {mc:.3f} ± {sec:.3f}')
print('Saved consolidated lines to runs/compare_returns.txt')
