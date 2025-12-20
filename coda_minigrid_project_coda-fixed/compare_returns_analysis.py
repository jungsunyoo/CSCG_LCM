#!/usr/bin/env python3

import argparse, re, json, math, os
from pathlib import Path
import numpy as np

# Optional deps
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False

def _try_parse_pairs(lines):
    # Try to read per-run pairs (baseline, coda) from lines that mention both
    baselines, codas = [], []
    for ln in lines:
        # Example patterns:
        # "seed=3 baseline=0.62 coda=0.68"
        # "baseline 0.612 | CoDA 0.691"
        m1 = re.search(r'baseline[^0-9\-+]*([-+]?\d*\.\d+|\d+)', ln, flags=re.I)
        m2 = re.search(r'(?:coda)\b[^0-9\-+]*([-+]?\d*\.\d+|\d+)', ln, flags=re.I)
        if m1 and m2:
            baselines.append(float(m1.group(1)))
            codas.append(float(m2.group(1)))
    return baselines, codas

def _try_parse_separate(lines):
    # If pairs weren't on the same line, try to parse two separate lists that appear in order
    b_vals, c_vals = [], []
    section = None
    for ln in lines:
        if re.search(r'\bbaseline\b', ln, flags=re.I): section = 'b'
        if re.search(r'\bcoda\b', ln, flags=re.I):     section = 'c'
        # pick the first float on each line if we are in a section
        m = re.search(r'([-+]?\d*\.\d+|\d+)', ln)
        if m and section == 'b': b_vals.append(float(m.group(1)))
        if m and section == 'c': c_vals.append(float(m.group(1)))
    return b_vals, c_vals

def _parse_summary(lines):
    # Parse the mean ± SE lines into a dict (as a fallback)
    out = {}
    for ln in lines:
        m = re.search(r'Baseline\s+mean\s*[±+/-]\s*SE:\s*([-+]?\d*\.\d+|\d+)\s*[±+/-]\s*([-+]?\d*\.\d+|\d+)', ln, flags=re.I)
        if m:
            out['baseline'] = {'mean': float(m.group(1)), 'se': float(m.group(2))}
        m = re.search(r'CoDA\s+mean\s*[±+/-]\s*SE:\s*([-+]?\d*\.\d+|\d+)\s*[±+/-]\s*([-+]?\d*\.\d+|\d+)', ln, flags=re.I)
        if m:
            out['coda'] = {'mean': float(m.group(1)), 'se': float(m.group(2))}
    return out

def paired_stats(b, c, B=10000, seed=0):
    # Return paired mean diff, SE, bootstrap CI, and simple t/z approximations.
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    assert len(b) == len(c) and len(b) > 0
    d = c - b
    mean_diff = float(d.mean())
    se_diff   = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan')
    # bootstrap
    rng = np.random.default_rng(seed)
    boot = []
    n = len(d)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        boot.append(d[idx].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # t-approx (fallback if scipy not available)
    t_stat = mean_diff / (d.std(ddof=1) / np.sqrt(n)) if n > 1 and d.std(ddof=1) > 0 else float('nan')
    return dict(
        n=n, mean_baseline=float(b.mean()), mean_coda=float(c.mean()),
        se_baseline=float(b.std(ddof=1)/np.sqrt(len(b))) if len(b) > 1 else float('nan'),
        se_coda=float(c.std(ddof=1)/np.sqrt(len(c))) if len(c) > 1 else float('nan'),
        mean_diff=mean_diff, se_diff=se_diff, boot_ci=[float(lo), float(hi)],
        t_stat=float(t_stat)
    )

def welch_stats(b, c):
    b = np.asarray(b, dtype=float); c = np.asarray(c, dtype=float)
    nb, nc = len(b), len(c)
    mb, mc = b.mean(), c.mean()
    vb, vc = b.var(ddof=1), c.var(ddof=1)
    se = math.sqrt(vb/nb + vc/nc) if nb>0 and nc>0 else float('nan')
    z = (mc-mb)/se if se>0 else float('nan')
    return dict(n_baseline=nb, n_coda=nc, mean_baseline=float(mb), mean_coda=float(mc), se_diff=float(se), z=float(z))

def main():
    ap = argparse.ArgumentParser(description='Analyze baseline vs CoDA returns, with paired/Welch tests and bootstrap CI.')
    ap.add_argument('--path', default='runs/compare_returns.txt', help='Path to compare_returns.txt')
    ap.add_argument('--out',  default='runs', help='Directory to save plots/summaries')
    ap.add_argument('--no-plot', action='store_true', help='Skip plotting even if matplotlib is available.')
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f'File not found: {p}')

    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    # 1) Try per-run paired parse
    b, c = _try_parse_pairs(lines)
    paired = len(b) == len(c) and len(b) > 0

    # 2) If not paired on one line, try separate sections
    if not paired:
        b2, c2 = _try_parse_separate(lines)
        paired = len(b2) == len(c2) and len(b2) > 0
        if paired:
            b, c = b2, c2

    # 3) If still nothing, fall back to summary-only parse
    summary = None
    if not paired:
        summary = _parse_summary(lines)

    os.makedirs(args.out, exist_ok=True)

    # Report
    if paired:
        stats = paired_stats(b, c, B=20000, seed=0)
        out = {
            'mode': 'paired',
            'stats': stats,
            'relative_lift': float((stats['mean_diff']) / (stats['mean_baseline'] if stats['mean_baseline'] != 0 else np.nan))
        }
        Path(args.out, 'compare_returns_summary.json').write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))

        # Plot
        if HAVE_PLT and not args.no_plot:
            import matplotlib.pyplot as plt
            xs = np.array(b); ys = np.array(c)
            plt.figure()
            plt.scatter(xs, ys, alpha=0.8)
            lims = [min(xs.min(), ys.min()), max(xs.max(), ys.max())]
            pad = 0.05*(lims[1]-lims[0] if lims[1]>lims[0] else 1.0)
            plt.plot([lims[0]-pad, lims[1]+pad], [lims[0]-pad, lims[1]+pad], linestyle='--')
            plt.xlabel('Baseline per-run return')
            plt.ylabel('CoDA per-run return')
            plt.title('Baseline vs CoDA (paired per seed)')
            plt.tight_layout()
            plt.savefig(Path(args.out, 'baseline_vs_coda_scatter.png'), dpi=150)
            plt.close()
    else:
        if summary is None:
            raise SystemExit('Could not parse paired runs or summary stats from the file. Please inspect the format.')
        # compute quick overlap diagnostic
        if 'baseline' in summary and 'coda' in summary:
            mb, seb = summary['baseline']['mean'], summary['baseline']['se']
            mc, sec = summary['coda']['mean'], summary['coda']['se']
            diff = mc - mb
            se_diff = math.sqrt(seb**2 + sec**2)
            z = diff / se_diff if se_diff > 0 else float('nan')
            out = dict(mode='summary_only', baseline=summary['baseline'], coda=summary['coda'],
                       diff=diff, se_diff=se_diff, z=z,
                       relative_lift=(diff / mb if mb != 0 else float('nan')))
            Path(args.out, 'compare_returns_summary.json').write_text(json.dumps(out, indent=2))
            print(json.dumps(out, indent=2))
        else:
            raise SystemExit('Found no usable data in the summary.')
if __name__ == '__main__':
    main()
