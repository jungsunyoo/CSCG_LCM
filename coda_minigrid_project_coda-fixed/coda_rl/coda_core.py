
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set

# -------------- utilities --------------
def _ensure_shape_3d(arr: np.ndarray, new_shape: Tuple[int,int,int]) -> np.ndarray:
    if arr is None:
        return np.zeros(new_shape, dtype=float)
    S_old, A_old, S2_old = arr.shape
    S, A, S2 = new_shape
    out = np.zeros(new_shape, dtype=float)
    out[:min(S_old,S), :min(A_old,A), :min(S2_old,S2)] = arr[:min(S_old,S), :min(A_old,A), :min(S2_old,S2)]
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

# --- uncertainty helpers ---
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
        z = 1.6448536269514722  # ~95%
    denom = 1.0 + (z*z)/n
    center = phat + (z*z)/(2.0*n)
    adj = z * ((phat*(1.0-phat) + (z*z)/(4.0*n))/n)**0.5
    return (center - adj)/denom

# -------------- CoDA --------------
@dataclass
class CoDAConfig:
    gamma: float = 0.95
    lam: float = 0.90
    theta_split: float = 0.60
    theta_merge: float = 0.50
    n_threshold: int = 8
    eps: float = 1e-9
    min_presence_episodes: int = 3
    min_effective_exposure: float = 5.0
    confidence: float = 0.80
    alpha0: float = 0.5
    beta0: float = 0.5
    count_decay: float = 0.90
    trace_decay: float = 0.90
    retro_decay: float = 0.90

@dataclass
class CoDAAgent:
    n_actions: int
    rewarded_flag: float = 1.0  # reward > 0 signals US
    cfg: CoDAConfig = field(default_factory=CoDAConfig)

    # dynamic state
    n_states: int = field(default=0, init=False)
    transition_counts: np.ndarray = field(default=None, init=False)
    transition_probs: np.ndarray = field(default=None, init=False)

    E_r: np.ndarray = field(default=None, init=False)
    E_nr: np.ndarray = field(default=None, init=False)
    C: np.ndarray = field(default=None, init=False)

    cs_us_presence: np.ndarray = field(default=None, init=False)
    us_episode_ema: float = field(default=0.0, init=False)
    presence_episodes: np.ndarray = field(default=None, init=False)

    salient_cues: Set[int] = field(default_factory=set, init=False)
    cue_to_clones: Dict[int, List[int]] = field(default_factory=dict, init=False)
    clone_parent: Dict[int, int] = field(default_factory=dict, init=False)
    created_by_cue: Dict[int, int] = field(default_factory=dict, init=False)

    # book-keeping (we hash raw observations to int state ids)
    _obs_to_sid: Dict[bytes, int] = field(default_factory=dict, init=False)
    _sid_to_obs: Dict[int, bytes] = field(default_factory=dict, init=False)

    def _sid(self, obs_hash: bytes) -> int:
        sid = self._obs_to_sid.get(obs_hash, None)
        if sid is None:
            sid = len(self._obs_to_sid)
            self._obs_to_sid[obs_hash] = sid
            self._sid_to_obs[sid] = obs_hash
            self._maybe_grow(sid)
        return sid

    def _maybe_grow(self, needed_state_index: int):
        if self.transition_counts is None:
            S = needed_state_index + 1
            self.n_states = S
            self.transition_counts = np.zeros((S, self.n_actions, S), dtype=float)
            self.transition_probs = normalize_transition_counts(self.transition_counts)
            self.E_r  = np.zeros((1, S)); self.E_nr = np.zeros((1, S)); self.C = np.zeros((1, S))
            self.cs_us_presence = np.zeros((1, S)); self.presence_episodes = np.zeros((1, S))
            return
        if needed_state_index < self.n_states:
            return
        new_n = needed_state_index + 1
        self.transition_counts = _ensure_shape_3d(self.transition_counts, (new_n, self.n_actions, new_n))
        self.transition_probs  = normalize_transition_counts(self.transition_counts)
        pad = ((0,0),(0,new_n - self.E_r.shape[1]))
        self.E_r  = np.pad(self.E_r,  pad, mode='constant')
        self.E_nr = np.pad(self.E_nr, pad, mode='constant')
        self.C    = np.pad(self.C,    pad, mode='constant')
        self.cs_us_presence = np.pad(self.cs_us_presence, pad, mode='constant')
        self.presence_episodes = np.pad(self.presence_episodes, pad, mode='constant')
        self.n_states = new_n

    # ---- per-episode update ----
    def update_with_episode(self, obs_hashes: List[bytes], actions: List[int], rewards: List[float]):
        if len(obs_hashes) < 2:  # no transitions
            return

        # Ensure arrays exist before applying decay
        if self.transition_counts is None:
            # Initialize minimal structure
            self._maybe_grow(0)

        # Materialize state ids first to ensure internal structures are created
        states = [self._sid(h) for h in obs_hashes]
        self._maybe_grow(max(states))

        # Safe to apply decays now
        if self.cfg.count_decay < 1.0:
            self.transition_counts *= self.cfg.count_decay
        if self.cfg.trace_decay < 1.0:
            self.E_r  *= self.cfg.trace_decay
            self.E_nr *= self.cfg.trace_decay
            self.C    *= self.cfg.trace_decay
        if self.cfg.retro_decay < 1.0:
            self.cs_us_presence *= self.cfg.retro_decay
            self.us_episode_ema *= self.cfg.retro_decay

        for t, a in enumerate(actions):
            s, sp = states[t], states[t+1]
            self._maybe_grow(max(s, sp))
            self.transition_counts[s, a, sp] += 1.0

        # eligibility-style accumulators wrt US (any reward > 0)
        S = self.n_states
        gamma, lam = self.cfg.gamma, self.cfg.lam
        e = np.zeros(S); c = np.zeros(S); e_nr = np.zeros(S)
        us_seen = False
        for t in range(len(actions)-1, -1, -1):
            s = states[t]
            # update traces
            e *= gamma * lam
            e[s] += 1.0
            c[s] += 1.0
            if rewards[t] > 0:
                self.E_r[0,:]  += e
                self.C[0,:]    += c
                us_seen = True
            else:
                self.E_nr[0,:] += e
        # retrospective: presence during US episodes using EMA denominator
        presence = np.zeros((1,S))
        for s in set(states):
            presence[0,s] = 1.0
        if us_seen:
            self.us_episode_ema += 1.0
            self.cs_us_presence += presence
        self.presence_episodes += presence

        self.transition_probs = normalize_transition_counts(self.transition_counts)

    # ---- PC & RC ----
    def prospective(self) -> np.ndarray:
        num = self.E_r.copy()
        den = self.E_r + self.E_nr + self.cfg.eps
        return (num / den).reshape(-1)

    def retrospective(self) -> np.ndarray:
        if self.us_episode_ema <= self.cfg.eps: return np.zeros(self.n_states)
        return (self.cs_us_presence / float(self.us_episode_ema)).reshape(-1)

    # ---- split/merge ----
    def maybe_split(self) -> List[int]:
        pc = self.prospective()
        new_cues = []
        exposure = (self.E_r + self.E_nr).reshape(-1)

        for s in range(self.n_states):
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
                # fallback to Wilson lower bound on pc[s]
                n = max(exposure[s], 1e-9)
                lb = wilson_lower_bound(pc[s], n, confidence=self.cfg.confidence)
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
        clones: List[int] = []
        seen_successors = set()
        for a in range(self.n_actions):
            s_primes = get_action_successors_from_counts(self.transition_counts, s, a)
            for sprime in s_primes:
                if sprime in seen_successors:
                    continue
                seen_successors.add(sprime)

                if sprime in self.clone_parent:
                    target = sprime  # already a clone somewhere
                else:
                    # create new clone
                    clone_id = self.n_states
                    self._maybe_grow(clone_id)

                    self.clone_parent[clone_id] = sprime
                    self.created_by_cue[clone_id] = s
                    clones.append(clone_id)

                    # initialize clone's outgoing like parent successor
                    self.transition_counts[clone_id, :, :] = self.transition_counts[sprime, :, :]
                    target = clone_id

                # re-route s --a--> sprime to s --a--> target
                cnt = self.transition_counts[s, a, sprime]
                if cnt > 0:
                    self.transition_counts[s, a, sprime] = 0.0
                    self.transition_counts[s, a, target] += cnt

        self.transition_probs = normalize_transition_counts(self.transition_counts)
        return clones

    def maybe_merge(self, edge_eps: float = 1e-6) -> List[int]:
        pc, rc = self.prospective(), self.retrospective()
        merged: List[int] = []

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

    # ----- basic stats for logging -----
    def stats(self, thresh: float = 0.9) -> Dict[str, float]:
        if self.transition_counts is None: 
            return {"frac_deterministic": 0.0, "norm_entropy": 1.0, "num_states": 0, "num_salient": 0}
        T = self.transition_probs
        rows = 0; frac = 0.0; ent_sum = 0.0
        for s in range(T.shape[0]):
            for a in range(T.shape[1]):
                p = T[s, a, :]
                if p.sum() <= 1e-12: 
                    continue
                rows += 1
                # deterministic?
                if p.max() >= thresh: frac += 1
                # entropy
                pp = p / max(p.sum(), 1e-12)
                ent_sum += float(-(pp * np.log(np.clip(pp,1e-12,1.0))).sum())
        avgK = T.shape[0]  # crude
        norm_ent = (ent_sum / max(rows,1)) / (np.log(max(2.0, avgK)))
        return {"frac_deterministic": frac/max(rows,1), "norm_entropy": norm_ent, "num_states": T.shape[0], "num_salient": len(self.salient_cues)}
