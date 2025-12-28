# coda_zipper.py

from __future__ import annotations
from dataclasses import dataclass, field

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Iterable
import math
import random

Action = int  # 0=Right, 1=Down

# -----------------------------
# Eligibility traces (configurable)
# -----------------------------
@dataclass
class EligibilityConfig:
    mode: str = "first"   # "first", "replacing", "accumulating", "decay"
    gamma: float = 1.0    # used for "decay"
    lam: float = 0.9      # used for "decay"

def compute_eligibility(states: List[int], cfg: EligibilityConfig) -> Dict[int, float]:
    """
    Returns eligibility weights over visited latent states in an episode prefix.
    Default "first": binary indicator for first-visit only.
    """
    e: Dict[int, float] = defaultdict(float)

    if cfg.mode == "first":
        seen = set()
        for s in states:
            if s not in seen:
                e[s] = 1.0
                seen.add(s)
        return dict(e)

    if cfg.mode == "replacing":
        # binary, but refreshed (still 1 if visited)
        for s in states:
            e[s] = 1.0
        return dict(e)

    if cfg.mode == "accumulating":
        # counts visits
        for s in states:
            e[s] += 1.0
        return dict(e)

    if cfg.mode == "decay":
        # TD(λ)-like backward decay: later states higher eligibility
        gamma, lam = cfg.gamma, cfg.lam
        elig = 1.0
        for s in reversed(states):
            e[s] += elig
            elig *= gamma * lam
        return dict(e)

    raise ValueError(f"Unknown eligibility mode: {cfg.mode}")

# -----------------------------
# Simple ID allocator (optional demo schedule)
# -----------------------------
class IdAllocator:
    """
    Default: sequential ids.
    Optional: provide a fixed list of ids (to reproduce a specific animation labeling).
    """
    def __init__(self, start: int, forced: Optional[List[int]] = None):
        self._next = start
        self._forced = list(forced) if forced else []

    def new(self) -> int:
        if self._forced:
            return self._forced.pop(0)
        nid = self._next
        self._next += 1
        return nid

# -----------------------------
# Environment: cued gridworld with aliased terminal
# -----------------------------
class CuedGridWorld:
    """
    4x4 grid, obs are 0..15 (row-major).
    Hidden context bit cue_seen flips to 1 upon visiting cue_obs.
    Terminal observation is always terminal_obs (aliased across cue_seen contexts),
    but reward depends on cue_seen -> creates reward-entropy at terminal obs.
    """
    def __init__(self, size: int = 4, cue_obs: int = 5, terminal_obs: int = 15):
        self.size = size
        self.cue_obs = cue_obs
        self.terminal_obs = terminal_obs
        self.reset()

    def reset(self) -> int:
        self.r = 0
        self.c = 0
        self.cue_seen = 0
        self.done = False
        return self._obs()

    def _obs(self) -> int:
        return self.r * self.size + self.c

    def available_actions(self) -> List[Action]:
        if self.done:
            return []
        acts = []
        if self.c < self.size - 1:
            acts.append(0)  # Right
        if self.r < self.size - 1:
            acts.append(1)  # Down
        return acts

    def step(self, a: Action) -> Tuple[int, float, bool, Dict[str, Any]]:
        if self.done:
            return self._obs(), 0.0, True, {}

        if a == 0 and self.c < self.size - 1:
            self.c += 1
        elif a == 1 and self.r < self.size - 1:
            self.r += 1

        obs = self._obs()

        if obs == self.cue_obs:
            self.cue_seen = 1

        rew = 0.0
        if obs == self.terminal_obs:
            self.done = True
            rew = 1.0 if self.cue_seen == 1 else 0.0

        return obs, rew, self.done, {"cue_seen": self.cue_seen}

# -----------------------------
# Minimal CoDA graph/agent
# -----------------------------
@dataclass
class CoDAConfig:
    # discovery
    reward_entropy_threshold: float = 0.65
    min_terminal_visits: int = 6

    # cue inference
    cue_score_threshold: float = 0.25

    # splitting control
    one_split_per_episode: bool = True

    # eligibility trace
    elig: EligibilityConfig = field(default_factory=EligibilityConfig)  # <-- FIX


class CoDAAgent:
    """
    Minimal agent:
    - Maintains latent states (clones) for each observation.
    - Learns transition probabilities via Dirichlet-counts per (latent, action).
    - Learns reward probabilities via Beta-counts per latent.
    - Discovers "interest obs" by high reward-entropy.
    - When interest triggers, uses eligibility-trace cue inference to pick a cue latent.
    - Splits at cue by cloning ALL experienced one-step successors of that cue.
    """

    def __init__(
        self,
        n_obs: int,
        cfg: Optional[CoDAConfig] = None,
        id_allocator: Optional[IdAllocator] = None,
        seed: int = 0,
    ):
        self.cfg = cfg or CoDAConfig()
        random.seed(seed)

        self.n_obs = n_obs
        self.id_alloc = id_allocator or IdAllocator(start=n_obs)

        # mapping: obs -> list of latent ids
        self.obs_to_latents: Dict[int, List[int]] = {o: [o] for o in range(n_obs)}
        # latent -> obs
        self.latent_to_obs: Dict[int, int] = {o: o for o in range(n_obs)}

        # transitions: counts[latent][action][latent_next] = count
        self.counts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        # reward Beta counts per latent: (alpha, beta)
        self.r_alpha = defaultdict(lambda: 1.0)
        self.r_beta  = defaultdict(lambda: 1.0)

        # presence/outcome stats for cue inference (over EPISODES)
        # pres_counts[latent] = (present_rew, present_total, absent_rew, absent_total)
        self.pres_counts = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        # discovered interest observations (by reward entropy)
        self.interest_obs: set[int] = set()

        # bookkeeping for “don’t keep cloning forever”
        self.split_history = set()  # (cue_latent, action, succ_obs) to avoid repeats

    # ---------- probabilities ----------
    def trans_probs(self, s: int, a: Action) -> Dict[int, float]:
        d = self.counts[s][a]
        tot = sum(d.values())
        if tot <= 0:
            return {}
        return {sp: c / tot for sp, c in d.items()}

    def reward_prob(self, s: int) -> float:
        return self.r_alpha[s] / (self.r_alpha[s] + self.r_beta[s])

    # ---------- discovery ----------
    @staticmethod
    def bernoulli_entropy(p: float) -> float:
        p = min(max(p, 1e-9), 1 - 1e-9)
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))

    def _obs_reward_entropy(self, obs: int) -> Tuple[float, int]:
        # aggregate across all latents for this obs
        alphas = 0.0
        betas = 0.0
        for s in self.obs_to_latents.get(obs, []):
            alphas += self.r_alpha[s] - 1.0
            betas  += self.r_beta[s] - 1.0
        n = int(alphas + betas)
        if n <= 0:
            return 0.0, 0
        p = alphas / (alphas + betas + 1e-9)
        return self.bernoulli_entropy(p), n

    def discover_interest_obs(self):
        # Discover any obs with stochastic reward (high entropy)
        for obs in range(self.n_obs):
            H, n = self._obs_reward_entropy(obs)
            if n >= self.cfg.min_terminal_visits and H >= self.cfg.reward_entropy_threshold:
                self.interest_obs.add(obs)

    # ---------- latent assignment ----------
    def choose_latent_for_obs(
        self,
        obs: int,
        prev_latent: Optional[int],
        prev_action: Optional[Action],
        reward_if_terminal: Optional[float] = None,
    ) -> int:
        """
        Choose which latent state to use for an observation.
        - If multiple latents exist, choose by transition likelihood from prev,
          and (if terminal reward known) choose latent whose reward model best matches.
        """
        candidates = self.obs_to_latents.get(obs, [obs])

        if len(candidates) == 1:
            return candidates[0]

        # If terminal reward is known, choose the latent with reward prob closest to outcome
        if reward_if_terminal is not None:
            target = float(reward_if_terminal)
            best = min(candidates, key=lambda s: abs(self.reward_prob(s) - target))
            return best

        # Else choose by transition probability from prev_latent, if available
        if prev_latent is not None and prev_action is not None:
            probs = self.trans_probs(prev_latent, prev_action)
            if probs:
                best = max(candidates, key=lambda s: probs.get(s, 0.0))
                if probs.get(best, 0.0) > 0:
                    return best

        return candidates[0]

    # ---------- splitting ----------
    def clone_latent(self, obs: int) -> int:
        new_id = self.id_alloc.new()
        self.obs_to_latents.setdefault(obs, []).append(new_id)
        self.latent_to_obs[new_id] = obs

        # initialize reward priors (copy weakly from first latent of this obs)
        base = self.obs_to_latents[obs][0]
        self.r_alpha[new_id] = self.r_alpha[base]
        self.r_beta[new_id] = self.r_beta[base]

        # transitions start empty (learned)
        return new_id

    def split_terminal_if_needed(self, terminal_obs: int) -> Optional[int]:
        """
        If terminal obs is interesting and only has 1 latent, create 1 clone.
        Returns new clone latent id if created.
        """
        if terminal_obs not in self.interest_obs:
            return None
        latents = self.obs_to_latents.get(terminal_obs, [terminal_obs])
        if len(latents) >= 2:
            return None
        new_lat = self.clone_latent(terminal_obs)
        return new_lat

    def cue_score(self, latent: int) -> float:
        pr, pt, ar, at = self.pres_counts[latent]
        if pt < 1 or at < 1:
            return 0.0
        p1 = pr / pt
        p0 = ar / at
        return abs(p1 - p0)

    def infer_cue_latent(self, elig: Dict[int, float]) -> Optional[int]:
        """
        Choose cue among eligible latents by highest contingency score.
        Eligibility weights can be used as tie-breakers / gating.
        """
        best_s = None
        best_score = 0.0
        for s, w in elig.items():
            if w <= 0:
                continue
            score = self.cue_score(s)
            if score > best_score:
                best_score = score
                best_s = s
        if best_s is None or best_score < self.cfg.cue_score_threshold:
            return None
        return best_s

    def split_at_cue(
        self,
        cue_latent: int,
        frames: Optional[List[Dict[str, Any]]] = None,
        highlight: bool = True,
    ) -> List[int]:
        """
        One split operation:
        Clone *all experienced one-step successors* of cue_latent for each action.
        Adds transitions cue_latent --a--> clone(succ_obs).
        Returns list of newly created clone latents.
        Optionally appends visualization frame-specs to `frames`.
        """
        new_clones: List[int] = []

        for a in [0, 1]:
            succ_counts = self.counts[cue_latent][a]
            if not succ_counts:
                continue

            # determine which successor OBS have been experienced
            succ_latents = list(succ_counts.keys())
            succ_obs_set = {self.latent_to_obs[sp] for sp in succ_latents}

            for succ_obs in sorted(succ_obs_set):
                key = (cue_latent, a, succ_obs)
                if key in self.split_history:
                    continue

                # create clone for that successor observation
                clone = self.clone_latent(succ_obs)
                new_clones.append(clone)
                self.split_history.add(key)

                # "branch": add count mass to cue->clone so it can be selected
                # (we do not delete old edge; CoDA will gradually favor the right branch)
                self.counts[cue_latent][a][clone] += 1.0

                if frames is not None and highlight:
                    frames.append({
                        "cue": cue_latent,
                        "new": clone,
                        "title": None,
                    })

        return new_clones

    # ---------- learning loop ----------
    def run_episode(
        self,
        env: CuedGridWorld,
        policy_fn,
        frames: Optional[List[Dict[str, Any]]] = None,
        terminal_obs: int = 15,
    ) -> Dict[str, Any]:
        """
        Runs one episode with an external policy_fn(env)->action.
        Updates transitions + rewards.
        Then: discover interest, maybe split terminal, maybe infer cue and split.
        Enforces: one split operation per episode (but can clone multiple one-step successors).
        """
        obs = env.reset()
        done = False

        episode_latents: List[int] = []
        episode_obs: List[int] = [obs]
        episode_actions: List[int] = []
        reward = 0.0

        prev_lat = None
        prev_a = None

        while not done:
            # choose latent for current obs (non-terminal phase)
            lat = self.choose_latent_for_obs(obs, prev_lat, prev_a)
            episode_latents.append(lat)

            a = policy_fn(env)
            obs2, r, done, info = env.step(a)

            # choose latent for next obs; if terminal and reward known, use reward-based assignment
            rew_if_term = r if (done and obs2 == terminal_obs) else None
            lat2 = self.choose_latent_for_obs(obs2, lat, a, reward_if_terminal=rew_if_term)

            # learn transition
            self.counts[lat][a][lat2] += 1.0

            episode_actions.append(a)
            episode_obs.append(obs2)

            obs = obs2
            prev_lat = lat
            prev_a = a
            reward = r

        # terminal reward update
        terminal_lat = episode_latents[-1] if episode_latents else self.choose_latent_for_obs(obs, None, None)
        # Note: at terminal, last appended latent corresponds to pre-terminal;
        # the terminal latent is chosen by assignment above as lat2 on last step.
        # We can retrieve it from last transition:
        if len(episode_actions) >= 1:
            pre_lat = episode_latents[-1]
            last_a = episode_actions[-1]
            # pick terminal latent as argmax of last transition counts just recorded
            # (since we incremented exactly once on last step)
            term_probs = self.trans_probs(pre_lat, last_a)
            if term_probs:
                terminal_lat = max(term_probs, key=term_probs.get)

        if reward > 0:
            self.r_alpha[terminal_lat] += 1.0
        else:
            self.r_beta[terminal_lat] += 1.0

        # update interest discovery
        self.discover_interest_obs()

        # split terminal first if needed
        new_terminal_clone = self.split_terminal_if_needed(terminal_obs)
        if new_terminal_clone is not None and frames is not None:
            # show terminal split as a frame (no cue highlight)
            frames.append({"cue": None, "new": new_terminal_clone, "title": None})

        # update presence/outcome statistics for cue inference (only if terminal is interesting)
        if terminal_obs in self.interest_obs:
            # define outcome as reward (1/0)
            outcome = 1.0 if reward > 0 else 0.0
            present = set(compute_eligibility(episode_latents, self.cfg.elig).keys())

            # update for all latents encountered so far in agent
            # (cheap way: update only those in present plus a sampled set; but we keep simple)
            all_latents = list(self.latent_to_obs.keys())
            for s in all_latents:
                if s in present:
                    self.pres_counts[s][0] += outcome
                    self.pres_counts[s][1] += 1.0
                else:
                    self.pres_counts[s][2] += outcome
                    self.pres_counts[s][3] += 1.0

        # If terminal is interesting and already has >=2 latents, try cue inference + split
        did_split = False
        if terminal_obs in self.interest_obs and len(self.obs_to_latents.get(terminal_obs, [terminal_obs])) >= 2:
            elig = compute_eligibility(episode_latents, self.cfg.elig)
            cue = self.infer_cue_latent(elig)
            if cue is not None:
                # one split op per episode
                self.split_at_cue(cue, frames=frames, highlight=True)
                did_split = True

        return {
            "reward": reward,
            "interest_obs": set(self.interest_obs),
            "did_split": did_split,
        }

# -----------------------------
# Simple scripted policies for the cued-grid demo
# -----------------------------
def policy_avoid_cue(env: CuedGridWorld) -> Action:
    """
    Deterministic-ish: go Right along top row, then Down.
    This avoids visiting cue at obs=5 (row=1,col=1) for size=4 if you stay on row=0.
    """
    # if can go right and we're still on top row -> go right
    if env.r == 0 and env.c < env.size - 1:
        return 0
    # otherwise go down if possible
    if env.r < env.size - 1:
        return 1
    return 0

def policy_seek_cue(env: CuedGridWorld) -> Action:
    """
    Go Down once, then Right once to hit obs=5 (row=1,col=1) for size=4.
    Then proceed Right/Down to terminal.
    """
    # target cue location
    cue_r, cue_c = divmod(env.cue_obs, env.size)

    # before reaching cue row, go down
    if env.r < cue_r:
        return 1
    # before reaching cue col, go right
    if env.c < cue_c:
        return 0
    # after cue reached, just move toward terminal: prefer right then down
    if env.c < env.size - 1:
        return 0
    if env.r < env.size - 1:
        return 1
    return 0
