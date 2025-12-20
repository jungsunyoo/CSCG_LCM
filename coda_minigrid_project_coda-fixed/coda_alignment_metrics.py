#!/usr/bin/env python3

import os, sys, math, json, argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

# Ensure your modules are importable
if '/mnt/data' not in sys.path:
    sys.path.append('/mnt/data')

# --- igraph shim so util.py can import even if igraph isn't installed ---
try:
    import igraph  # type: ignore
except Exception:
    import types
    sys.modules['igraph'] = types.ModuleType('igraph')

# Import your code
from spatial_environments import GridEnvRightDownNoSelf, GridEnvRightDownNoCue, GridEnv, ContinuousTMaze  # type: ignore
from util import transition_matrix_action, row_normalize, generate_dataset  # type: ignore
from coda_metrics import kl_over_time, entropy_over_time, markovization_score  # type: ignore

# -------------------------------
# Ground-truth augmented builder
# -------------------------------

def build_gt_augmented_T_for_grid(env, action_order: List[int] = None) -> np.ndarray:
    # Build an augmented-state ground-truth transition tensor T_gt (S_aug x A x S_aug)
    # by adding a 'cue-visited' bit m \in {0,1} to each base state.
    # Terminals collapse to their true outcome states using m.

    # Identify base (non-terminal) states
    all_base = sorted(env.pos_to_state.values())
    terminals_R = set(env.rewarded_terminals)
    terminals_NR = set(env.unrewarded_terminals)
    terminals = terminals_R | terminals_NR

    base_states = [s for s in all_base if s not in terminals]
    S_base = len(base_states)

    # Map base state to compact index
    base_to_idx = {s:i for i,s in enumerate(base_states)}

    # Augmented indices: (s_idx, m) -> aug_idx; append terminal ids as absorbing nodes
    aug_map = {}
    idx = 0
    for s in base_states:
        for m in (0,1):
            aug_map[(s,m)] = idx
            idx += 1

    term_ids = list(sorted(terminals))  # original ids
    term_aug_idx = {}
    for t in term_ids:
        term_aug_idx[t] = idx
        idx += 1

    S_aug = idx

    # Action space (detected from env.get_valid_actions over base states)
    if action_order is None:
        act_set = set()
        for s in base_states:
            act_set.update(env.get_valid_actions(s))
        action_order = sorted(list(act_set))
    A = max(action_order)+1 if action_order else 1

    T = np.zeros((S_aug, A, S_aug), dtype=float)

    # Helper to compute deterministic next base state from s,a
    def next_base_state(s, a):
        if s in terminals:
            return s
        if a not in env.get_valid_actions(s):
            return s
        r, c = env.state_to_pos[s]
        dr, dc = env.base_actions[a]
        nr, nc = r + dr, c + dc
        return env.pos_to_state[(nr, nc)]

    cue_set = set(getattr(env, 'cue_states', []) or [])
    pair_map = {}
    for i, r_t in enumerate(env.rewarded_terminals):
        if i < len(env.unrewarded_terminals):
            pair_map[r_t] = env.unrewarded_terminals[i]

    for s in base_states:
        for m in (0,1):
            i_aug = aug_map[(s,m)]
            for a in action_order:
                sp = next_base_state(s, a)
                m_next = 1 if (m==1 or sp in cue_set) else 0

                if sp in terminals_R:
                    # choose terminal depending on memory bit
                    t_id = sp if m_next==1 else pair_map.get(sp, sp)
                    j_aug = term_aug_idx[t_id]
                elif sp in terminals_NR:
                    j_aug = term_aug_idx[sp]
                else:
                    j_aug = aug_map[(sp, m_next)]

                T[i_aug, a, j_aug] += 1.0

    # Make terminals absorbing (self-loops)
    for t in term_ids:
        j = term_aug_idx[t]
        for a in action_order:
            T[j, a, j] = 1.0

    # Row-normalize
    denom = T.sum(axis=2, keepdims=True)
    denom[denom==0] = 1.0
    T = T / denom
    return T

# -----------------
# Utilities/metrics
# -----------------

def dataset_to_T(dataset):
    # states/actions -> counts -> probabilities using your util functions.
    counts = transition_matrix_action(dataset)
    denom = counts.sum(axis=2, keepdims=True).astype(float)
    denom[denom == 0] = 1.0
    return counts / denom

def kl_to_ground_truth_over_time(T_series: List[np.ndarray], env, use_js: bool=False) -> np.ndarray:
    # Compute KL (or JS) between each empirical T in T_series and the environment's
    # augmented ground-truth T_gt (pads shapes as needed).
    T_gt = build_gt_augmented_T_for_grid(env)
    ref_fn = lambda T: T_gt
    return kl_over_time(T_series, ref_fn, use_js=use_js)

def summarize_alignment(T_series: List[np.ndarray], env) -> Dict[str, float]:
    ent = float(np.mean(entropy_over_time(T_series)))
    kl  = float(np.mean(kl_to_ground_truth_over_time(T_series, env, use_js=False)))
    js  = float(np.mean(kl_to_ground_truth_over_time(T_series, env, use_js=True)))
    markov = float(np.mean([markovization_score(T) for T in T_series]))
    return dict(mean_entropy=ent, mean_kl_to_gt=kl, mean_js_to_gt=js, mean_markovization=markov)

# --------------
# Script entry
# --------------

def main():
    ap = argparse.ArgumentParser(description='CoDA alignment to ground-truth augmented MDP (KL/JS/entropy).')
    ap.add_argument('--env', default='GridEnvRightDownNoSelf', choices=[
        'GridEnvRightDownNoSelf', 'GridEnvRightDownNoCue', 'GridEnv', 'ContinuousTMaze'
    ])
    ap.add_argument('--episodes', type=str, default='', help="Optional .npz with saved dataset {'states': [...], 'actions': [...]} list; if empty, will generate random dataset via util.generate_dataset.")
    ap.add_argument('--n_episodes', type=int, default=1000, help='If generating, number of episodes.')
    ap.add_argument('--max_steps', type=int, default=20, help='If generating, max steps per episode.')
    ap.add_argument('--out', type=str, default='runs/alignment', help='Output directory for a small JSON summary.')
    args = ap.parse_args()

    # Instantiate env
    if args.env == 'GridEnvRightDownNoSelf':
        env = GridEnvRightDownNoSelf()
    elif args.env == 'GridEnvRightDownNoCue':
        env = GridEnvRightDownNoCue()
    elif args.env == 'GridEnv':
        env = GridEnv()
    else:
        env = ContinuousTMaze()

    # Load or generate episodes
    dataset = None
    if args.episodes:
        data = np.load(args.episodes, allow_pickle=True)
        if 'states' in data and 'actions' in data:
            states_list = data['states']
            actions_list = data['actions']
            dataset = [(list(states_list[i]), list(actions_list[i])) for i in range(len(states_list))]
        else:
            raise SystemExit("Expected keys 'states' and 'actions' in the .npz file.")
    else:
        dataset = generate_dataset(env, n_episodes=args.n_episodes, max_steps=args.max_steps)

    # Build empirical T and summarize
    T_emp = dataset_to_T(dataset)
    T_series = [T_emp]

    metrics = summarize_alignment(T_series, env)
    os.makedirs(args.out, exist_ok=True)
    Path(args.out, 'alignment_summary.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
