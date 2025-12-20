
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import numpy as np

# -------------------- Minimal environments (self-contained) --------------------

class _BaseEnv:
    def __init__(self):
        self.clone_dict = {}
        self.reverse_clone_dict = {}
        self.cue_states = []
        self.rewarded_terminals = []
        self.unrewarded_terminals = []
        self.valid_actions = {}
        self.base_actions = {}
        self.env_size = (4,4)
        self.pos_to_state = {}
        self.state_to_pos = {}
        self.start_state = 0
        self.current_state = 0
        self.num_unique_states = 0
        self.visited_cue = False

    def add_clone_dict(self, new_clone, successor):
        self.clone_dict[new_clone] = successor

    def add_reverse_clone_dict(self, new_clone, successor):
        self.reverse_clone_dict[successor] = new_clone

    def reset(self):
        self.current_state = self.start_state
        self.visited_cue = False
        return self.current_state

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.current_state
        return self.valid_actions[state]

    def is_terminal(self, s):
        return s in self.rewarded_terminals or s in self.unrewarded_terminals

# Deterministic right+down grid, reward only if cue visited (otherwise -1 and go to unrewarded terminal label)
class GridEnvRightDownNoSelf(_BaseEnv):
    def __init__(self, cue_states=[5], env_size=(4,4), rewarded_terminal=[15]):
        super().__init__()
        self.env_size = env_size
        self.cue_states = cue_states
        self.rewarded_terminals = rewarded_terminal

        # Build mapping 0..(rows*cols-1)
        state = -1
        for i in range(self.env_size[0]):
            for j in range(self.env_size[1]):
                state += 1
                self.pos_to_state[(i,j)] = state
        self.state_to_pos = {s: rc for rc, s in self.pos_to_state.items()}

        # allocate one unrewarded terminal for each rewarded terminal after the grid ids
        last_grid_state = state
        self.unrewarded_terminals = [s + last_grid_state for s in np.arange(1, len(rewarded_terminal)+1)]
        self.num_unique_states = last_grid_state + 1 + len(self.unrewarded_terminals)

        self.start_state = 0
        self.current_state = self.start_state

        # two actions: right(0), down(1)
        self.base_actions = {0:(0,1), 1:(1,0)}

        # valid actions per grid state; terminals have none
        self.valid_actions = {}
        for i in range(self.env_size[0]):
            for j in range(self.env_size[1]):
                s = self.pos_to_state[(i,j)]
                acts = []
                for a, (di, dj) in self.base_actions.items():
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.env_size[0] and 0 <= nj < self.env_size[1]:
                        if self.pos_to_state[(ni,nj)] != s:
                            acts.append(a)
                self.valid_actions[s] = acts
        for t in self.rewarded_terminals + self.unrewarded_terminals:
            self.valid_actions[t] = []

    def step(self, action: int):
        if action not in self.get_valid_actions(self.current_state):
            raise ValueError("Invalid action")
        if self.is_terminal(self.current_state):
            return self.current_state, 0, True

        i,j = self.state_to_pos[self.current_state]
        di, dj = self.base_actions[action]
        ni, nj = i+di, j+dj
        next_state = self.pos_to_state[(ni,nj)]

        if next_state in self.cue_states:
            self.visited_cue = True

        reward, done = 0, False
        if next_state in self.rewarded_terminals:
            done = True
            if self.visited_cue:
                reward = +1
            else:
                reward = -1
                idx = self.rewarded_terminals.index(next_state)
                next_state = self.unrewarded_terminals[idx]

        self.current_state = next_state
        return next_state, reward, done


# Same geometry; outcome is not exclusively tied to cue (used to degrade informativeness / extinction)
class GridEnvRightDownNoCue(GridEnvRightDownNoSelf):
    def step(self, action: int):
        if action not in self.get_valid_actions(self.current_state):
            raise ValueError("Invalid action")
        if self.is_terminal(self.current_state):
            return self.current_state, 0, True

        i,j = self.state_to_pos[self.current_state]
        di, dj = self.base_actions[action]
        ni, nj = i+di, j+dj
        next_state = self.pos_to_state[(ni,nj)]

        if next_state in self.cue_states:
            self.visited_cue = True

        reward, done = 0, False
        if next_state in self.rewarded_terminals:
            done = True
            # outcome does NOT depend on cue here (randomized for demo)
            if np.random.rand() < 0.5:
                reward = +1
            else:
                reward = -1
                idx = self.rewarded_terminals.index(next_state)
                next_state = self.unrewarded_terminals[idx]

        self.current_state = next_state
        return next_state, reward, done


# -------------------- CoDA helpers (inline) --------------------

def compute_eligibility_traces(states, n_states, gamma=0.9, lam=0.8):
    """
    Return (E, C) where:
      E[t, s] = eligibility of state s after step t
      C[t, s] = raw visit counts for state s after step t
    """
    E = np.zeros((len(states), n_states))
    C = np.zeros((len(states), n_states))
    e = np.zeros(n_states)
    c = np.zeros(n_states)
    for t, s in enumerate(states):
        e *= gamma * lam
        e[s] += 1.0
        c[s] += 1.0
        E[t] = e.copy()
        C[t] = c.copy()
    return E, C

def accumulate_conditioned_eligibility_traces(E_r, E_nr, C, state_seq, sprime, sprime2, n_states, lam=0.8, gamma=0.9):
    E, count = compute_eligibility_traces(state_seq, n_states, lam=lam, gamma=gamma)
    if state_seq[-1] == sprime:
        E_r += E[-1, :]
    elif state_seq[-1] == sprime2:
        E_nr += E[-1, :]
    C += count[-1, :]
    return E_r, E_nr, C

# ---------------- utility helpers ----------------

def _ensure_shape_3d(arr: np.ndarray, new_shape: Tuple[int,int,int]) -> np.ndarray:
    """
    Grow (or initialize) a [S, A, S] array to new_shape, preserving existing data.
    """
    if arr is None:
        return np.zeros(new_shape, dtype=float)
    S_old, A_old, S2_old = arr.shape
    S, A, S2 = new_shape
    out = np.zeros(new_shape, dtype=float)
    out[:S_old, :A_old, :S2_old] = arr
    return out

def normalize_transition_counts(counts: np.ndarray, eps: float=1e-12) -> np.ndarray:
    """Row-normalize a transition count tensor counts[s,a,s'] -> probs[s,a,s']"""
    probs = counts.astype(float, copy=True)
    denom = probs.sum(axis=2, keepdims=True)
    denom[denom < eps] = 1.0
    probs /= denom
    return probs

def get_successors_from_counts(counts: np.ndarray, s: int) -> List[int]:
    """All s' with any count from state s (across actions)."""
    if counts.size == 0:
        return []
    succ_mask = counts[s].sum(axis=0) > 0
    return list(np.where(succ_mask)[0])

def get_action_successors_from_counts(counts: np.ndarray, s: int, a: int) -> List[int]:
    if counts.size == 0:
        return []
    succ_mask = counts[s, a] > 0
    return list(np.where(succ_mask)[0])

def _presence_vector(count_vec: np.ndarray) -> np.ndarray:
    """Convert counts per state in an episode to a 0/1 'was it visited' vector."""
    pres = (count_vec > 0).astype(float)
    return pres

# ------------------- Uncertainty helpers -------------------
try:
    # mpmath is widely available; used for Beta CDF
    from mpmath import betainc, erfcinv
    _HAS_MPMATH = True
except Exception:
    _HAS_MPMATH = False

def posterior_prob_p_greater_than(theta: float, success: float, failure: float, alpha0: float=0.5, beta0: float=0.5) -> float:
    """
    P(p > theta | data) under Beta(alpha0+success, beta0+failure).
    Works with fractional 'success'/'failure' from eligibility traces.
    """
    if not _HAS_MPMATH:
        # Fallback: conservative 0.0 if mpmath isn't available
        return 0.0
    a = alpha0 + max(0.0, float(success))
    b = beta0 + max(0.0, float(failure))
    # regularized incomplete beta from 0..theta is the CDF; tail is 1-CDF
    cdf = betainc(a, b, 0, theta, regularized=True)
    try:
        return float(1.0 - cdf)
    except Exception:
        return 0.0

def wilson_lower_bound(phat: float, n: float, confidence: float=0.95) -> float:
    """One-sided Wilson lower bound (approx) allowing non-integer n."""
    if n <= 0:
        return 0.0
    # z for one-sided lower bound
    if _HAS_MPMATH:
        z = float((2.0**0.5) * erfcinv(2*(1.0-confidence)))
    else:
        z = 1.6448536269514722  # approx for 95% one-sided
    denom = 1.0 + (z*z)/n
    center = phat + (z*z)/(2.0*n)
    adj = z * ((phat*(1.0-phat) + (z*z)/(4.0*n))/n)**0.5
    return (center - adj)/denom

# ---------------- CoDA core ----------------

@dataclass
class CoDAConfig:
    gamma: float = 0.9
    lam: float = 0.8
    theta_split: float = 0.9
    theta_merge: float = 0.5
    n_threshold: int = 10        # minimum evidence per state to act on it
    eps: float = 1e-9
    # --- uncertainty control for splitting ---
    min_presence_episodes: int = 5
    min_effective_exposure: float = 20.0
    confidence: float = 0.95     # require P(p>theta_split | data) >= confidence
    alpha0: float = 0.5          # Jeffreys prior by default
    beta0: float = 0.5
    count_decay: float = 1.0  # e.g., 0.995 to slowly forget, 1.0 to disable
    
@dataclass
class CoDAAgent:
    env: _BaseEnv
    cfg: CoDAConfig = field(default_factory=CoDAConfig)

    # dynamic state (initialized in __post_init__)
    n_actions: int = field(init=False)
    n_states: int = field(init=False)  # grows as clones are added
    transition_counts: np.ndarray = field(init=False)   # [S,A,S]
    transition_probs: np.ndarray = field(init=False)    # [S,A,S]

    # contextual eligibility accumulators (trial-by-trial)
    E_r: np.ndarray = field(init=False)  # shape [1,S]
    E_nr: np.ndarray = field(init=False) # shape [1,S]
    C: np.ndarray = field(init=False)    # shape [1,S]  (visit counts)

    # retrospective counters for merging
    us_episode_count: int = field(default=0, init=False)      # # episodes that ended in rewarded terminal
    cs_us_presence: np.ndarray = field(init=False)            # per-state # of US-episodes where state was present

    # presence counts across *all* episodes (for min_presence_episodes gating)
    presence_episodes: np.ndarray = field(init=False)

    # bookkeeping for splits
    salient_cues: Set[int] = field(default_factory=set, init=False)
    cue_to_clones: Dict[int, List[int]] = field(default_factory=dict, init=False)
    clone_parent: Dict[int, int] = field(default_factory=dict, init=False)    # clone_id -> original successor
    created_by_cue: Dict[int, int] = field(default_factory=dict, init=False)  # clone_id -> cue state

    def __post_init__(self):
        # infer number of actions
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

    # ------------------- growth helpers -------------------
    def _maybe_grow(self, needed_state_index: int):
        """Ensure internal arrays include [0..needed_state_index]."""
        if needed_state_index < self.n_states:
            return
        new_n = needed_state_index + 1
        # grow transitions
        self.transition_counts = _ensure_shape_3d(self.transition_counts, (new_n, self.n_actions, new_n))
        self.transition_probs = normalize_transition_counts(self.transition_counts)
        # grow contextual eligibility and presence
        pad = ((0,0),(0,new_n - self.E_r.shape[1]))
        self.E_r  = np.pad(self.E_r,  pad, mode='constant')
        self.E_nr = np.pad(self.E_nr, pad, mode='constant')
        self.C    = np.pad(self.C,    pad, mode='constant')
        self.cs_us_presence = np.pad(self.cs_us_presence, pad, mode='constant')
        self.presence_episodes = np.pad(self.presence_episodes, pad, mode='constant')
        self.n_states = new_n

    # ------------------- episode update -------------------
    def update_with_episode(self, states: List[int], actions: List[int]):
        """
        Trial-by-trial update:
        • transition counts
        • eligibility trace accumulators (E_r, E_nr, C)
        • retrospective counters for P(CS|US)
        """
        if self.cfg.count_decay < 1.0:
            self.transition_counts *= self.cfg.count_decay

        # Grow if we saw clone IDs
        max_id = max(states) if states else 0
        self._maybe_grow(max_id)

        # 1) transition counts
        for t in range(len(actions)):
            s  = states[t]
            a  = actions[t]
            sp = states[t+1]
            self._maybe_grow(max(s, sp))
            self.transition_counts[s, a, sp] += 1.0

        # 2) contextual eligibility snapshots
        rewarded_terminal = self.env.rewarded_terminals[0] if len(self.env.rewarded_terminals)>0 else None
        unrewarded_terminal = self.env.unrewarded_terminals[0] if len(self.env.unrewarded_terminals)>0 else None
        self.E_r, self.E_nr, self.C = accumulate_conditioned_eligibility_traces(
            self.E_r, self.E_nr, self.C,
            states,
            sprime=rewarded_terminal,
            sprime2=unrewarded_terminal,
            n_states=self.n_states,
            lam=self.cfg.lam,
            gamma=self.cfg.gamma
        )

        # 3) retrospective counters for P(CS|US) using presence-only statistics
        _, ep_counts = compute_eligibility_traces(states, self.n_states, gamma=self.cfg.gamma, lam=self.cfg.lam)
        presence = (ep_counts[-1,:] > 0).astype(float)[None, :]
        if states[-1] == rewarded_terminal:
            self.us_episode_count += 1
            self.cs_us_presence[:, :presence.shape[1]] += presence

        # global presence across all episodes
        self.presence_episodes[:, :presence.shape[1]] += presence

        # 4) update probs
        self.transition_probs = normalize_transition_counts(self.transition_counts)

    # ------------------- contingency estimates -------------------
    def prospective(self) -> np.ndarray:
        """P(US|CS) via contextual eligibility traces E_r/(E_r+E_nr)."""
        num = self.E_r.copy()
        den = self.E_r + self.E_nr + self.cfg.eps
        return (num / den).reshape(-1)  # shape [S]

    def retrospective(self) -> np.ndarray:
        """P(CS|US) using presence among US episodes (episode-level counts)."""
        if self.us_episode_count == 0:
            return np.zeros(self.n_states)
        rc = (self.cs_us_presence / float(self.us_episode_count)).reshape(-1)
        return rc

    # ------------------- split / merge rules -------------------
    def maybe_split(self) -> List[int]:
        """Return list of newly‑split cue states (could be empty). Uses uncertainty-aware gating."""
        pc = self.prospective()
        new_cues = []

        # effective exposure per state from contextual traces
        exposure = (self.E_r + self.E_nr).reshape(-1)

        for s in range(self.n_states):
            if s in self.env.rewarded_terminals or s in self.env.unrewarded_terminals:
                continue

            # basic evidence gate (visits)
            if self.C[0, s] < self.cfg.n_threshold:
                continue

            # min presence across episodes (avoid '3 good trials' issue)
            if self.presence_episodes[0, s] < self.cfg.min_presence_episodes:
                continue

            # min effective exposure from traces (controls posterior sharpness)
            if exposure[s] < self.cfg.min_effective_exposure:
                continue

            # if already a cue, skip
            if s in self.salient_cues:
                continue

            # Uncertainty-aware test: require posterior P(p > theta_split) >= confidence
            post_prob = posterior_prob_p_greater_than(self.cfg.theta_split, self.E_r[0, s], self.E_nr[0, s], self.cfg.alpha0, self.cfg.beta0)

            # As a backup when mpmath isn't available, use Wilson lower bound
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
        seen_successors = set()  # avoid duplicating work within this call

        for a in range(self.n_actions):
            s_primes = get_action_successors_from_counts(self.transition_counts, s, a)
            for sprime in s_primes:
                if sprime in seen_successors:
                    # If we already handled this successor, just re-route below
                    pass
                seen_successors.add(sprime)

                # --- 0) Do NOT clone terminals ---
                if sprime in terminals:
                    # leave s->terminal as is; do NOT create/reuse any clone
                    continue

                # --- 1) Never clone a clone ---
                if sprime in self.clone_parent:
                    target = sprime  # it's already a clone; re-route to it
                else:
                    # --- 2) Reuse existing clone for this original successor if any ---
                    target = self.env.reverse_clone_dict.get(sprime, None)
                    if target is None:
                        # --- 3) Create a new clone ONCE per original successor ---
                        clone_id = self.n_states
                        self._maybe_grow(clone_id)

                        self.env.add_clone_dict(clone_id, successor=sprime)
                        self.env.add_reverse_clone_dict(new_clone=clone_id, successor=sprime)

                        self.clone_parent[clone_id] = sprime
                        self.created_by_cue[clone_id] = s
                        clones.append(clone_id)

                        # initialize clone's outgoing counts like parent
                        self.transition_counts[clone_id, :, :] = self.transition_counts[sprime, :, :]
                        target = clone_id

                # --- 4) Re-route edges s --a--> sprime to s --a--> target (existing or new clone) ---
                cnt = self.transition_counts[s, a, sprime]
                if cnt > 0:
                    self.transition_counts[s, a, sprime] = 0.0
                    self.transition_counts[s, a, target] += cnt

        self.transition_probs = normalize_transition_counts(self.transition_counts)
        return clones



    def maybe_merge(self) -> List[int]:
        """Check merge condition for existing cues, merge and return list of merged cues."""
        pc = self.prospective()
        rc = self.retrospective()
        merged: List[int] = []

        for cue in list(self.salient_cues):
            cue_info = pc[cue] * rc[cue]
            if cue_info < self.cfg.theta_merge:
                self._merge_cue(cue)
                self.salient_cues.remove(cue)
                self.cue_to_clones.pop(cue, None)
                merged.append(cue)

        self.transition_probs = normalize_transition_counts(self.transition_counts)
        return merged

    def _merge_cue(self, cue: int):
        """Merge: re-route s->clone back into s->original successor, zero-out clone inbound counts."""
        clones = self.cue_to_clones.get(cue, [])
        for clone_id in clones:
            if clone_id >= self.n_states:
                continue
            if clone_id not in self.clone_parent:
                continue
            sprime = self.clone_parent[clone_id]
            # re-route from cue
            for a in range(self.n_actions):
                cnt = self.transition_counts[cue, a, clone_id]
                if cnt > 0:
                    self.transition_counts[cue, a, clone_id] = 0.0
                    self.transition_counts[cue, a, sprime] += cnt
            # clear clone inbound counts from everyone else
            inbound = self.transition_counts[:, :, clone_id]
            if inbound.sum() > 0:
                self.transition_counts[:, :, sprime] += inbound
                self.transition_counts[:, :, clone_id] = 0.0

            # clear outgoing (clone remains dormant)
            self.transition_counts[clone_id, :, :] = 0.0

            # clean env bookkeeping (optional)
            if clone_id in self.env.clone_dict:
                self.env.clone_dict.pop(clone_id, None)
        # rebuild reverse clone dict to map successor->latest clone if any remain
        new_rev = {}
        for cl, par in self.env.clone_dict.items():
            new_rev[par] = cl
        self.env.reverse_clone_dict = new_rev

    # ------------------- convenience -------------------
    def get_T(self) -> np.ndarray:
        """Return current transition probabilities [S,A,S]."""
        return self.transition_probs

    def get_PC_RC(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.prospective(), self.retrospective()


# ------------------- dataset helpers (self-contained) -------------------

def generate_episode(env: _BaseEnv, T: Optional[np.ndarray]=None, max_steps: int=20):
    """
    Generate ONE episode as (states, actions).
    If T is provided (post-augmentation), we map original successors to clone IDs
    when the probability in T for the original edge is near zero (as in your notebook).
    """
    states, actions = [], []
    s = env.reset()
    for _ in range(max_steps):
        states.append(s)
        valid = env.get_valid_actions(s if s < (env.env_size[0]*env.env_size[1]) else env.clone_dict.get(s, 0))
        if not valid:
            break
        a = np.random.choice(valid)
        actions.append(a)
        snext, r, done = env.step(a)

        # If T is provided, redirect to clone when original edge becomes invalid
        if T is not None:
            # defensively ensure shapes
            if snext < T.shape[2] and s < T.shape[0] and a < T.shape[1]:
                if T[s, a, snext] < 1e-6 and snext in env.reverse_clone_dict:
                    snext = env.reverse_clone_dict[snext]

        s = snext
        if done:
            states.append(s)
            break
    return states, actions


# ------------------- demonstration -------------------

def run_demo(seed: int=0, n_acq: int=300, n_ext: int=400):
    """
    Simple end-to-end demo (with uncertainty-aware split gates):
    1) Acquisition on GridEnvRightDownNoSelf (one deterministic cue task).
    2) Extinction on a variant env in which reward is not exclusively tied to the cue.
    Returns a small log dictionary with split/merge episode indices.
    """
    np.random.seed(seed)

    env = GridEnvRightDownNoSelf(cue_states=[5], env_size=(4,4), rewarded_terminal=[15])
    agent = CoDAAgent(env)

    split_events = []
    merge_events = []

    with_clones = False
    for ep in range(1, n_acq+1):
        states, actions = generate_episode(env, T=agent.get_T() if with_clones else None, max_steps=20)
        agent.update_with_episode(states, actions)
        new_cues = agent.maybe_split()
        if new_cues:
            with_clones = True
            split_events.extend([ep]*len(new_cues))

    # extinction-like phase (outcome independent of cue)
    env_ext = GridEnvRightDownNoCue(cue_states=[5], env_size=(4,4), rewarded_terminal=[15])
    env_ext.clone_dict = dict(agent.env.clone_dict)
    env_ext.reverse_clone_dict = dict(agent.env.reverse_clone_dict)
    agent.env = env_ext

    for ep in range(n_acq+1, n_acq+n_ext+1):
        states, actions = generate_episode(env_ext, T=agent.get_T(), max_steps=20)
        agent.update_with_episode(states, actions)
        merged = agent.maybe_merge()
        if merged:
            merge_events.extend([ep]*len(merged))

    return {"split_events": split_events, "merge_events": merge_events, "final_num_states": agent.n_states}
