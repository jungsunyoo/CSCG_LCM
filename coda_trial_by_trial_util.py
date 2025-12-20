from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import numpy as np

from util import compute_eligibility_traces, accumulate_conditioned_eligibility_traces
from spatial_environments import GridEnvRightDownNoSelf, GridEnvRightDownNoCue

# ------------------- Uncertainty helpers -------------------
try:
    from mpmath import betainc, erfcinv
    _HAS_MPMATH = True
except Exception:
    _HAS_MPMATH = False

def posterior_prob_p_greater_than(theta: float, success: float, failure: float, alpha0: float=0.5, beta0: float=0.5) -> float:
    if not _HAS_MPMATH:
        return 0.0
    a = alpha0 + max(0.0, float(success))
    b = beta0 + max(0.0, float(failure))
    cdf = betainc(a, b, 0, theta, regularized=True)
    return float(1.0 - cdf)

def wilson_lower_bound(phat: float, n: float, confidence: float=0.95) -> float:
    if n <= 0:
        return 0.0
    if _HAS_MPMATH:
        z = float((2.0**0.5) * erfcinv(2*(1.0-confidence)))
    else:
        z = 1.6448536269514722
    denom = 1.0 + (z*z)/n
    center = phat + (z*z)/(2.0*n)
    adj = z * ((phat*(1.0-phat) + (z*z)/(4.0*n))/n)**0.5
    return (center - adj)/denom

# ------------------- Core -------------------
def _ensure_shape_3d(arr: np.ndarray, new_shape: Tuple[int,int,int]) -> np.ndarray:
    if arr is None:
        return np.zeros(new_shape, dtype=float)
    S_old, A_old, S2_old = arr.shape
    S, A, S2 = new_shape
    out = np.zeros(new_shape, dtype=float)
    out[:S_old, :A_old, :S2_old] = arr
    return out

def normalize_transition_counts(counts: np.ndarray, eps: float=1e-12) -> np.ndarray:
    probs = counts.astype(float, copy=True)
    denom = probs.sum(axis=2, keepdims=True)
    denom[denom < eps] = 1.0
    probs /= denom
    return probs

def get_action_successors_from_counts(counts: np.ndarray, s: int, a: int):
    if counts.size == 0:
        return []
    succ_mask = counts[s, a] > 0
    return list(np.where(succ_mask)[0])

@dataclass
class CoDAConfig:
    gamma: float = 0.9
    lam: float = 0.8
    theta_split: float = 0.9
    theta_merge: float = 0.5
    n_threshold: int = 10
    eps: float = 1e-9
    min_presence_episodes: int = 5
    min_effective_exposure: float = 20.0
    confidence: float = 0.95
    alpha0: float = 0.5
    beta0: float = 0.5
    # forgetting knobs
    count_decay: float = 1.0   # e.g., 0.995 to fade old transitions
    trace_decay: float = 1.0   # decays E_r, E_nr, C (e.g., 0.98–0.995)
    retro_decay: float = 1.0   # decays cs_us_presence and US-episode EMA

@dataclass
class CoDAAgent:
    env: GridEnvRightDownNoSelf
    cfg: CoDAConfig = field(default_factory=CoDAConfig)

    # dynamics
    n_actions: int = field(init=False)
    n_states: int = field(init=False)
    transition_counts: np.ndarray = field(init=False)
    transition_probs: np.ndarray = field(init=False)

    # prospective accumulators
    E_r: np.ndarray = field(init=False)
    E_nr: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)

    # retrospective accumulators (EMA)
    cs_us_presence: np.ndarray = field(init=False)
    us_episode_ema: float = field(default=0.0, init=False)

    # presence & bookkeeping
    presence_episodes: np.ndarray = field(init=False)
    salient_cues: Set[int] = field(default_factory=set, init=False)
    cue_to_clones: Dict[int, List[int]] = field(default_factory=dict, init=False)
    clone_parent: Dict[int, int] = field(default_factory=dict, init=False)
    created_by_cue: Dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        all_actions = set()
        for s, acts in self.env.valid_actions.items():
            for a in acts:
                all_actions.add(a)
        self.n_actions = max(all_actions) + 1 if all_actions else 0

        self.n_states = int(self.env.num_unique_states)

        self.transition_counts = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=float)
        self.transition_probs = normalize_transition_counts(self.transition_counts)

        self.E_r  = np.zeros((1, self.n_states))
        self.E_nr = np.zeros((1, self.n_states))
        self.C    = np.zeros((1, self.n_states))
        self.cs_us_presence = np.zeros((1, self.n_states))
        self.presence_episodes = np.zeros((1, self.n_states))

    def _maybe_grow(self, needed_state_index: int):
        if needed_state_index < self.n_states:
            return
        new_n = needed_state_index + 1
        self.transition_counts = _ensure_shape_3d(self.transition_counts, (new_n, self.n_actions, new_n))
        self.transition_probs = normalize_transition_counts(self.transition_counts)
        pad = ((0,0),(0,new_n - self.E_r.shape[1]))
        self.E_r  = np.pad(self.E_r,  pad, mode='constant')
        self.E_nr = np.pad(self.E_nr, pad, mode='constant')
        self.C    = np.pad(self.C,    pad, mode='constant')
        self.cs_us_presence = np.pad(self.cs_us_presence, pad, mode='constant')
        self.presence_episodes = np.pad(self.presence_episodes, pad, mode='constant')
        self.n_states = new_n

    def update_with_episode(self, states: List[int], actions: List[int]):
        # A) fade old transitions
        if self.cfg.count_decay < 1.0:
            self.transition_counts *= self.cfg.count_decay

        # B) decay prospective accumulators
        if self.cfg.trace_decay < 1.0:
            self.E_r  *= self.cfg.trace_decay
            self.E_nr *= self.cfg.trace_decay
            self.C    *= self.cfg.trace_decay

        # C) decay retrospective accumulators + denominator
        if self.cfg.retro_decay < 1.0:
            self.cs_us_presence *= self.cfg.retro_decay
            self.us_episode_ema *= self.cfg.retro_decay

        max_id = max(states) if states else 0
        self._maybe_grow(max_id)

        # transition counts
        for t in range(len(actions)):
            s  = states[t]
            a  = actions[t]
            sp = states[t+1]
            self._maybe_grow(max(s, sp))
            self.transition_counts[s, a, sp] += 1.0

        # prospective traces (context-conditioned)
        rewarded_terminal  = self.env.rewarded_terminals[0]  if len(self.env.rewarded_terminals)>0   else None
        unrewarded_terminal= self.env.unrewarded_terminals[0] if len(self.env.unrewarded_terminals)>0 else None
        self.E_r, self.E_nr, self.C = accumulate_conditioned_eligibility_traces(
            self.E_r, self.E_nr, self.C,
            states,
            sprime=rewarded_terminal,
            sprime2=unrewarded_terminal,
            n_states=self.n_states,
            lam=self.cfg.lam,
            gamma=self.cfg.gamma
        )

        # presence for retrospective/visit gates
        _, ep_counts = compute_eligibility_traces(states, self.n_states, gamma=self.cfg.gamma, lam=self.cfg.lam)
        presence = (ep_counts[-1,:] > 0).astype(float)[None, :]

        # EMA denominator + retrospective numerator
        if states[-1] == rewarded_terminal:
            self.us_episode_ema += 1.0
            self.cs_us_presence[:, :presence.shape[1]] += presence

        self.presence_episodes[:, :presence.shape[1]] += presence
        self.transition_probs = normalize_transition_counts(self.transition_counts)

    def prospective(self) -> np.ndarray:
        num = self.E_r.copy()
        den = self.E_r + self.E_nr + self.cfg.eps
        return (num / den).reshape(-1)

    def retrospective(self) -> np.ndarray:
        if self.us_episode_ema <= self.cfg.eps:
            return np.zeros(self.n_states)
        return (self.cs_us_presence / float(self.us_episode_ema)).reshape(-1)

    def maybe_split(self) -> List[int]:
        pc = self.prospective()
        new_cues = []
        exposure = (self.E_r + self.E_nr).reshape(-1)

        for s in range(self.n_states):
            if s in self.env.rewarded_terminals or s in self.env.unrewarded_terminals:
                continue
            if self.C[0, s] < self.cfg.n_threshold:
                continue
            if self.presence_episodes[0, s] < self.cfg.min_presence_episodes:
                continue
            if exposure[s] < self.cfg.min_effective_exposure:
                continue
            if s in self.salient_cues:
                continue

            post_prob = posterior_prob_p_greater_than(self.cfg.theta_split, self.E_r[0, s], self.E_nr[0, s],
                                                      self.cfg.alpha0, self.cfg.beta0)
            if post_prob == 0.0:
                lb = wilson_lower_bound(pc[s], max(exposure[s], 1e-9), confidence=self.cfg.confidence)
                pass_test = (lb > self.cfg.theta_split)
            else:
                pass_test = (post_prob >= self.cfg.confidence)

            if pass_test:
                clones = self._split_state(s)
                if clones:
                    self.salient_cues.add(s)
                    self.cue_to_clones[s] = list(clones)
                    new_cues.append(s)
        return new_cues

    def _split_state(self, s: int) -> List[int]:
        """Clone successors of state s and re-route s->s' to s->clone(s').
           Do not clone terminals; never clone a clone; reuse existing clones."""
        clones: List[int] = []
        terminals = set(self.env.rewarded_terminals) | set(self.env.unrewarded_terminals)
        seen_successors = set()

        for a in range(self.n_actions):
            s_primes = get_action_successors_from_counts(self.transition_counts, s, a)
            for sprime in s_primes:
                if sprime in seen_successors:
                    continue
                seen_successors.add(sprime)

                if sprime in terminals:
                    continue

                if sprime in self.clone_parent:
                    target = sprime
                else:
                    target = self.env.reverse_clone_dict.get(sprime, None)
                    if target is None:
                        clone_id = self.n_states
                        self._maybe_grow(clone_id)

                        self.env.add_clone_dict(clone_id, successor=sprime)
                        self.env.add_reverse_clone_dict(new_clone=clone_id, successor=sprime)

                        self.clone_parent[clone_id] = sprime
                        self.created_by_cue[clone_id] = s
                        clones.append(clone_id)

                        self.transition_counts[clone_id, :, :] = self.transition_counts[sprime, :, :]
                        target = clone_id

                cnt = self.transition_counts[s, a, sprime]
                if cnt > 0:
                    self.transition_counts[s, a, sprime] = 0.0
                    self.transition_counts[s, a, target] += cnt

        self.transition_probs = normalize_transition_counts(self.transition_counts)
        return clones

    def maybe_merge(self) -> List[int]:
        """Merge when informativeness falls or clone edges are unused."""
        pc = self.prospective()
        rc = self.retrospective()
        merged: List[int] = []
        edge_eps = 1e-6

        for cue in list(self.salient_cues):
            info = pc[cue] * rc[cue]

            clone_ids = self.cue_to_clones.get(cue, [])
            edge_mass = 0.0
            if clone_ids:
                for a in range(self.n_actions):
                    for cl in clone_ids:
                        if cl < self.transition_probs.shape[2]:
                            edge_mass += float(self.transition_probs[cue, a, cl])

            if (info < self.cfg.theta_merge) or (edge_mass < edge_eps):
                self._merge_cue(cue)
                self.salient_cues.remove(cue)
                self.cue_to_clones.pop(cue, None)
                merged.append(cue)

        self.transition_probs = normalize_transition_counts(self.transition_counts)
        return merged

    def _merge_cue(self, cue: int):
        clones = self.cue_to_clones.get(cue, [])
        for clone_id in clones:
            if clone_id >= self.n_states or clone_id not in self.clone_parent:
                continue
            sprime = self.clone_parent[clone_id]
            for a in range(self.n_actions):
                cnt = self.transition_counts[cue, a, clone_id]
                if cnt > 0:
                    self.transition_counts[cue, a, clone_id] = 0.0
                    self.transition_counts[cue, a, sprime] += cnt
            inbound = self.transition_counts[:, :, clone_id]
            if inbound.sum() > 0:
                self.transition_counts[:, :, sprime] += inbound
                self.transition_counts[:, :, clone_id] = 0.0
            self.transition_counts[clone_id, :, :] = 0.0
            if clone_id in self.env.clone_dict:
                self.env.clone_dict.pop(clone_id, None)
        new_rev = {}
        for cl, par in self.env.clone_dict.items():
            new_rev[par] = cl
        self.env.reverse_clone_dict = new_rev

    def get_T(self) -> np.ndarray:
        return self.transition_probs

    def get_PC_RC(self):
        return self.prospective(), self.retrospective()
