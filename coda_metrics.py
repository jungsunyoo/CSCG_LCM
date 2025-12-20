
import numpy as np
from typing import Callable, Optional, Tuple, Dict, List

EPS = 1e-12

def _safe_row_norm(x: np.ndarray, axis: int = -1, eps: float = EPS) -> np.ndarray:
    y = x.astype(float, copy=True)
    s = y.sum(axis=axis, keepdims=True)
    s[s < eps] = 1.0
    y /= s
    return y

def _pad_to_shape(A: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    S, A_, S2 = A.shape
    Sg, Ag, S2g = shape
    out = np.zeros(shape, dtype=float)
    out[:min(S,Sg), :min(A_,Ag), :min(S2,S2g)] = A[:min(S,Sg), :min(A_,Ag), :min(S2,S2g)]
    return out

def _aggregate_actions(T: np.ndarray) -> np.ndarray:
    return _safe_row_norm(T.sum(axis=1), axis=1)

def _kl_row(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p = np.clip(p, eps, 1.0); p /= p.sum()
    q = np.clip(q, eps, 1.0); q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))

def _js_row(p: np.ndarray, q: np.ndarray, eps: float = EPS) -> float:
    p = np.clip(p, eps, 1.0); p /= p.sum()
    q = np.clip(q, eps, 1.0); q /= q.sum()
    m = 0.5*(p+q)
    return 0.5*_kl_row(p, m, eps) + 0.5*_kl_row(q, m, eps)

def _entropy_row(p: np.ndarray, eps: float = EPS) -> float:
    p = np.clip(p, eps, 1.0); p /= p.sum()
    return float(-np.sum(p * np.log(p)))

def kl_over_time(T_series: List[np.ndarray],
                 T_ref_fn,
                 weights: Optional[np.ndarray] = None,
                 use_js: bool = False) -> np.ndarray:
    scores = []
    for T in T_series:
        T_ref = T_ref_fn(T)
        shape = tuple(max(x,y) for x,y in zip(T.shape, T_ref.shape))
        T_ = _pad_to_shape(T, shape)
        R_ = _pad_to_shape(T_ref, shape)
        P = _aggregate_actions(T_)
        Q = _aggregate_actions(R_)
        S = P.shape[0]
        if weights is None:
            w = np.ones(S, dtype=float)/S
        else:
            w = np.zeros(S, dtype=float)
            w[:min(S, weights.shape[0])] = weights[:min(S, weights.shape[0])]
            w = w / max(w.sum(), EPS)
        if use_js:
            row_scores = np.array([_js_row(P[i], Q[i]) for i in range(S)])
        else:
            row_scores = np.array([_kl_row(P[i], Q[i]) for i in range(S)])
        scores.append(float(np.sum(w * row_scores)))
    return np.array(scores)

def entropy_over_time(T_series: List[np.ndarray]) -> np.ndarray:
    out = []
    for T in T_series:
        P = _aggregate_actions(T)
        H = np.array([_entropy_row(P[i]) for i in range(P.shape[0])])
        out.append(float(np.mean(H)))
    return np.array(out)

def markovization_score(T: np.ndarray, eps: float = EPS) -> float:
    P = _aggregate_actions(T)
    H = np.array([_entropy_row(P[i], eps) for i in range(P.shape[0])])
    Hmax = np.log(max(2, P.shape[1]))
    return float(1.0 - np.mean(H)/Hmax)

def ref_empirical_from_rollouts(env, policy_fn, n_episodes=200, max_steps=20) -> np.ndarray:
    S_guess = getattr(env, "num_unique_states", None)
    if S_guess is None:
        S_guess = max(getattr(env, "state_to_pos", {0:(0,0)}).keys()) + 16
    A_guess = 4
    counts = np.zeros((S_guess, A_guess, S_guess), dtype=float)
    for _ in range(n_episodes):
        s = env.reset()
        for _t in range(max_steps):
            acts = env.get_valid_actions(s if s not in getattr(env, "clone_dict", {}) else env.clone_dict[s])
            if not acts: break
            a = policy_fn(env, s, acts)
            s_next, r, done = env.step(a)
            if s >= counts.shape[0] or a >= counts.shape[1] or s_next >= counts.shape[2]:
                newS = max(counts.shape[0], s+1, s_next+1)
                newA = max(counts.shape[1], a+1)
                new_counts = np.zeros((newS, newA, newS), dtype=float)
                new_counts[:counts.shape[0], :counts.shape[1], :counts.shape[2]] = counts
                counts = new_counts
            counts[s, a, s_next] += 1.0
            s = s_next
            if done:
                break
    denom = counts.sum(axis=2, keepdims=True)
    denom[denom < EPS] = 1.0
    return counts / denom

def greedy_right_down_policy(env, s, acts):
    if 0 in acts: return 0
    if 1 in acts: return 1
    return np.random.choice(acts)
