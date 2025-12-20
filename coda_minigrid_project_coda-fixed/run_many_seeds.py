#!/usr/bin/env python3
# Usage:
#   python run_many_seeds.py --n 30
#
# This script loops over seeds, sets seeds for Python/numpy/(torch if present),
# and runs your existing 'evaluate_and_compare.py' in a fresh process each time.
# After each run, if 'runs/compare_returns.txt' is created by your script, it will
# be copied to 'runs/seeded/compare_returns_seed{K}.txt' to avoid overwriting.
#
# IMPORTANT: To guarantee that each run actually *honors* the per-run seed,
# add the tiny patch shown below to evaluate_and_compare.py (near the top):
#   import os, random, numpy as np
#   def seed_everything(seed):
#       os.environ['PYTHONHASHSEED'] = str(seed)
#       random.seed(seed); np.random.seed(seed)
#       try:
#           import torch
#           torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
#           torch.use_deterministic_algorithms(True, warn_only=True)
#       except Exception:
#           pass
#   seed_everything(int(os.getenv('SEED', '0')))
#
# That reseeds AFTER modules are imported (overriding any module-level fixed seeds).
#
import argparse, os, subprocess, shutil, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30, help='number of seeds')
    ap.add_argument('--start', type=int, default=0, help='starting seed')
    ap.add_argument('--python', type=str, default=sys.executable, help='Python binary to use')
    ap.add_argument('--script', type=str, default='evaluate_and_compare.py', help='Script to run per seed')
    args = ap.parse_args()

    outdir = Path('runs/seeded')
    outdir.mkdir(parents=True, exist_ok=True)

    for s in range(args.start, args.start + args.n):
        env = os.environ.copy()
        env['SEED'] = str(s)
        print(f'=== Running seed {s} ===')
        # Run the user script in a fresh process with SEED exported.
        p = subprocess.run([args.python, args.script], env=env)
        if p.returncode != 0:
            print(f'[seed {s}] script returned {p.returncode}; stopping.')
            sys.exit(p.returncode)

        src = Path('runs/compare_returns.txt')
        if src.exists():
            dst = outdir / f'compare_returns_seed{s}.txt'
            shutil.copy(src, dst)
            print(f'  saved: {dst}')
        else:
            print('  WARNING: runs/compare_returns.txt not found after run; nothing copied.')
    print('Done.')

if __name__ == '__main__':
    main()
