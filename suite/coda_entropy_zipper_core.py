
"""
coda_entropy_zipper_core.py
===========================

CoDA-style cognitive graph learning with:

1) Reward-free DISCOVERY of "interest transitions" using transition entropy/surprisal
   (no assumption that reward is special).

2) Eligibility-trace based INFERENCE:
   - Maintain online eligibility vector e_t over visited latent states.
   - For each transition (s_t, a_t -> s_{t+1}), classify it as "interest" vs "non-interest".
   - Accumulate outcome-conditioned evidence:
        E_interest[s]   += e_t[s]
        E_noninterest[s]+= e_t[s]
   - Prospective contingency:
        PC(s) = P(interest | s) ≈ E_interest[s] / (E_interest[s] + E_noninterest[s])

3) SPLITTING and ZIPPERING (iterative, not global):
   - ONE split decision per episode: pick the single best new cue state.
   - Once a cue is chosen, perform a "zipper sweep" along the *experienced suffix* of that episode:
        for each state on the suffix, clone all one-step successors that have been experienced
        (i.e., have counts) and redirect edges to the clones.
   - This resolves downstream ambiguity gradually as experience reaches them.

4) MERGING (soft merge via deactivation):
   - Every episode, check all salient cues.
   - If utility(s) = PC(s)*RC(s) falls below theta_merge, undo redirects and deactivate descendants.
   - RC(s) is a retrospective contingency for *salient cues only*:
        RC(s)=P(s | interest) estimated by counts at interest transitions.

Eligibility trace modes:
- "first_visit" (default): only first visit to state in an episode can increment eligibility
- "accumulating": eligibility increments every visit
- "replacing": sets eligibility to 1 on visit

Dependencies:
  numpy
Optional for visualization:
  networkx, matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Hashable, Optional, List, Set, Any
import math
import numpy as np

Obs = Hashable
Action = Hashable
NodeId = int


@dataclass
class Node:
    sid: NodeId
    obs: Obs
    active: bool = True
    original_sid: Optional[NodeId] = None
    parent_cue: Optional[NodeId] = None
    created_at: int = 0


@dataclass
class GraphModel:
    """
    Transition counts:
        counts[(s,a)][s_next] = count
    """
    nodes: Dict[NodeId, Node] = field(default_factory=dict)
    counts: Dict[Tuple[NodeId, Action], Dict[NodeId, float]] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes[node.sid] = node

    def add_transition(self, s: NodeId, a: Action, s_next: NodeId, w: float = 1.0) -> None:
        d = self.counts.setdefault((s, a), {})
        d[s_next] = d.get(s_next, 0.0) + float(w)

    def successors(self, s: NodeId, a: Action) -> Dict[NodeId, float]:
        return self.counts.get((s, a), {})

    def total_sa(self, s: NodeId, a: Action) -> float:
        d = self.successors(s, a)
        return float(sum(d.values()))

    def redirect_all_successors(self, s: NodeId, a: Action, mapping: Dict[NodeId, NodeId]) -> None:
        d = self.counts.get((s, a))
        if not d:
            return
        newd: Dict[NodeId, float] = {}
        for old_next, c in d.items():
            new_next = mapping.get(old_next, old_next)
            newd[new_next] = newd.get(new_next, 0.0) + c
        self.counts[(s, a)] = newd


@dataclass
class EligibilityTraceConfig:
    gamma: float = 0.95
    lam: float = 0.9
    mode: str = "first_visit"  # "first_visit" | "accumulating" | "replacing"


@dataclass
class DiscoveryConfig:
    """
    Reward-free interest detection based on transition entropy/surprisal of NEXT OBS labels.
    """
    mode: str = "entropy"        # "entropy" | "surprisal" | "either"
    min_sa_count: float = 10.0   # wait for some data before declaring interest
    entropy_threshold: float = 0.45   # nats
    surprisal_threshold: float = 1.0  # -log p
    eps: float = 1e-9


@dataclass
class CoDAEntropyConfig:
    et: EligibilityTraceConfig = field(default_factory=EligibilityTraceConfig)
    disc: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    # split/merge
    n_threshold: float = 5.0
    theta_split: float = 0.8
    theta_merge: float = 0.4

    # behavior
    one_split_per_episode: bool = True
    zipper_sweep_only_when_new_cue: bool = True  # prevents repeated sweeping explosion

    max_nodes: int = 20000
    eps: float = 1e-9


@dataclass
class Snapshot:
    episode: int
    title: str
    nodes: Dict[int, Dict[str, Any]]
    edges: Dict[Tuple[int, Any], Dict[int, float]]
    salient: List[int]


def _entropy(ps: np.ndarray, eps: float = 1e-12) -> float:
    ps = ps[ps > eps]
    if ps.size == 0:
        return 0.0
    return float(-(ps * np.log(ps)).sum())


class CoDAEntropyZipperAgent:
    """
    Main agent: reward-free entropy discovery + eligibility inference + iterative zippering + merging.
    """

    def __init__(self, cfg: CoDAEntropyConfig):
        self.cfg = cfg
        self.G = GraphModel()

        self._next_sid: int = 1
        self.episode_idx: int = 0

        # cache: (cur, action, next_obs) -> next_sid
        self.obs_cache: Dict[Tuple[NodeId, Action, Obs], NodeId] = {}

        # PC evidence: interest vs non-interest
        self.e_interest: Dict[NodeId, float] = {}
        self.e_noninterest: Dict[NodeId, float] = {}

        # salient cues + RC counts (interest-conditioned)
        self.salient: Set[NodeId] = set()
        self.a_cue_and_interest: Dict[NodeId, float] = {}
        self.c_interest_without_cue: Dict[NodeId, float] = {}

        # split/merge bookkeeping under a root cue
        self.split_records: Dict[NodeId, List[Tuple[NodeId, Action, NodeId, NodeId]]] = {}
        self.descendants: Dict[NodeId, Set[NodeId]] = {}

        # keep sweep stable
        self._cue_swept_once: Set[NodeId] = set()

    # ------------------- Node helpers -------------------

    def new_node(self, obs: Obs, *, original_sid: Optional[NodeId] = None, parent_cue: Optional[NodeId] = None) -> NodeId:
        if len(self.G.nodes) >= self.cfg.max_nodes:
            raise RuntimeError(f"Exceeded max_nodes={self.cfg.max_nodes}.")
        sid = self._next_sid
        self._next_sid += 1
        self.G.add_node(Node(sid=sid, obs=obs, active=True, original_sid=original_sid, parent_cue=parent_cue, created_at=self.episode_idx))
        return sid

    def ensure_start(self, start_obs: Obs) -> NodeId:
        for n in self.G.nodes.values():
            if n.active and n.obs == start_obs and n.original_sid is None:
                return n.sid
        return self.new_node(start_obs)

    def _active_node_with_obs(self, sid: NodeId, obs: Obs) -> bool:
        n = self.G.nodes.get(sid)
        return (n is not None) and n.active and (n.obs == obs)

    # ------------------- Transition assignment -------------------

    def step_transition(self, cur: NodeId, a: Action, next_obs: Obs) -> NodeId:
        key = (cur, a, next_obs)
        if key in self.obs_cache:
            nxt = self.obs_cache[key]
            if self._active_node_with_obs(nxt, next_obs):
                self.G.add_transition(cur, a, nxt, 1.0)
                return nxt

        succ = self.G.successors(cur, a)
        for sid_next in succ.keys():
            if self._active_node_with_obs(sid_next, next_obs):
                self.obs_cache[key] = sid_next
                self.G.add_transition(cur, a, sid_next, 1.0)
                return sid_next

        new_sid = self.new_node(next_obs)
        self.obs_cache[key] = new_sid
        self.G.add_transition(cur, a, new_sid, 1.0)
        return new_sid

    # ------------------- Discovery: interest transitions -------------------

    def _sa_obs_distribution(self, s: NodeId, a: Action) -> Dict[Obs, float]:
        out: Dict[Obs, float] = {}
        for sid_next, c in self.G.successors(s, a).items():
            n = self.G.nodes.get(sid_next)
            if n is None or not n.active:
                continue
            out[n.obs] = out.get(n.obs, 0.0) + float(c)
        return out

    def transition_entropy(self, s: NodeId, a: Action) -> float:
        d = self._sa_obs_distribution(s, a)
        tot = sum(d.values())
        if tot <= 0:
            return 0.0
        ps = np.array([v / tot for v in d.values()], dtype=float)
        return _entropy(ps)

    def transition_surprisal(self, s: NodeId, a: Action, next_obs: Obs) -> float:
        d = self._sa_obs_distribution(s, a)
        tot = sum(d.values())
        if tot <= 0:
            return 0.0
        p = float(d.get(next_obs, 0.0)) / float(tot)
        p = max(p, self.cfg.disc.eps)
        return float(-math.log(p))

    def is_interest_transition(self, s: NodeId, a: Action, next_obs: Obs) -> bool:
        if self.G.total_sa(s, a) < self.cfg.disc.min_sa_count:
            return False

        H = self.transition_entropy(s, a)
        S = self.transition_surprisal(s, a, next_obs)

        mode = self.cfg.disc.mode
        if mode == "entropy":
            return H >= self.cfg.disc.entropy_threshold
        if mode == "surprisal":
            return S >= self.cfg.disc.surprisal_threshold
        if mode == "either":
            return (H >= self.cfg.disc.entropy_threshold) or (S >= self.cfg.disc.surprisal_threshold)
        raise ValueError(f"Unknown discovery mode: {mode}")

    # ------------------- Eligibility update (online) -------------------

    def _eligibility_step_update(self, e: Dict[NodeId, float], visited: Set[NodeId], s: NodeId) -> None:
        decay = self.cfg.et.gamma * self.cfg.et.lam
        for k in list(e.keys()):
            e[k] *= decay
            if e[k] < 1e-12:
                del e[k]

        mode = self.cfg.et.mode
        if mode == "accumulating":
            e[s] = e.get(s, 0.0) + 1.0
        elif mode == "replacing":
            e[s] = 1.0
        elif mode == "first_visit":
            if s not in visited:
                e[s] = e.get(s, 0.0) + 1.0
                visited.add(s)
        else:
            raise ValueError(f"Unknown eligibility mode: {mode}")

    # ------------------- PC / RC / utility -------------------

    def evidence(self, s: NodeId) -> float:
        return self.e_interest.get(s, 0.0) + self.e_noninterest.get(s, 0.0)

    def pc(self, s: NodeId) -> float:
        i = self.e_interest.get(s, 0.0)
        ni = self.e_noninterest.get(s, 0.0)
        denom = i + ni
        return 0.0 if denom <= self.cfg.eps else i / denom

    def rc(self, s: NodeId) -> float:
        a = self.a_cue_and_interest.get(s, 0.0)
        c = self.c_interest_without_cue.get(s, 0.0)
        denom = a + c
        return 0.0 if denom <= self.cfg.eps else a / denom

    def utility(self, s: NodeId) -> float:
        return self.pc(s) * self.rc(s)

    # ------------------- Splitting primitives -------------------

    def _clone_node(self, base_sid: NodeId, *, parent_cue: NodeId) -> NodeId:
        base = self.G.nodes[base_sid]
        return self.new_node(base.obs, original_sid=base_sid, parent_cue=parent_cue)

    def _clone_one_step_successors_of(self, s: NodeId, *, root_cue: NodeId) -> List[NodeId]:
        created: List[NodeId] = []
        for (ss, a), succ in list(self.G.counts.items()):
            if ss != s:
                continue
            mapping: Dict[NodeId, NodeId] = {}
            for old_next in list(succ.keys()):
                old_node = self.G.nodes.get(old_next)
                if old_node is None or not old_node.active:
                    continue
                clone = self._clone_node(old_next, parent_cue=root_cue)
                mapping[old_next] = clone
                created.append(clone)
                self.split_records.setdefault(root_cue, []).append((s, a, old_next, clone))
                self.descendants.setdefault(root_cue, set()).add(clone)
            if mapping:
                self.G.redirect_all_successors(s, a, mapping)
        return created

    def _zipper_sweep_suffix(self, latents: List[NodeId], actions: List[Action], cue_idx: int, *, root_cue: NodeId) -> None:
        for t in range(cue_idx, len(actions)):
            s = latents[t]
            node = self.G.nodes.get(s)
            if node is None or not node.active:
                continue
            self._clone_one_step_successors_of(s, root_cue=root_cue)

    # ------------------- Merge / deactivation -------------------

    def _merge_from_cue(self, cue: NodeId) -> None:
        recs = self.split_records.get(cue, [])

        # revert redirects grouped by (src, action): mapping clone->old
        by_source_action: Dict[Tuple[NodeId, Action], Dict[NodeId, NodeId]] = {}
        for src, a, old_next, clone_next in recs:
            by_source_action.setdefault((src, a), {})[clone_next] = old_next

        for (src, a), mapping in by_source_action.items():
            d = self.G.counts.get((src, a))
            if not d:
                continue
            newd: Dict[NodeId, float] = {}
            for nxt, c in d.items():
                old = mapping.get(nxt, nxt)
                newd[old] = newd.get(old, 0.0) + c
            self.G.counts[(src, a)] = newd

        # deactivate descendants (including ancestry)
        to_deactivate: Set[NodeId] = set(self.descendants.get(cue, set()))
        for sid, node in self.G.nodes.items():
            if not node.active or node.parent_cue is None:
                continue
            p = node.parent_cue
            while p is not None:
                if p == cue:
                    to_deactivate.add(sid)
                    break
                pnode = self.G.nodes.get(p)
                p = pnode.parent_cue if pnode is not None else None

        for sid in to_deactivate:
            n = self.G.nodes.get(sid)
            if n is not None:
                n.active = False
                self.salient.discard(sid)

        self.salient.discard(cue)
        self._cue_swept_once.discard(cue)

    # ------------------- Episode processing -------------------

    def process_episode(self, obs_seq: List[Obs], action_seq: Optional[List[Action]] = None) -> Dict[str, Any]:
        if action_seq is None:
            action_seq = ["next"] * (len(obs_seq) - 1)
        if len(action_seq) != len(obs_seq) - 1:
            raise ValueError("action_seq must have length len(obs_seq)-1 (or be None).")

        # build latent path
        cur = self.ensure_start(obs_seq[0])
        latents = [cur]
        for a, o_next in zip(action_seq, obs_seq[1:]):
            cur = self.step_transition(cur, a, o_next)
            latents.append(cur)

        # online eligibility + evidence updates
        e: Dict[NodeId, float] = {}
        visited: Set[NodeId] = set()
        n_interest_steps = 0

        for t in range(len(action_seq)):
            s = latents[t]
            a = action_seq[t]
            next_obs = obs_seq[t + 1]

            self._eligibility_step_update(e, visited, s)

            interest = self.is_interest_transition(s, a, next_obs)
            if interest:
                n_interest_steps += 1

            tgt = self.e_interest if interest else self.e_noninterest
            for sid, w in e.items():
                tgt[sid] = tgt.get(sid, 0.0) + w

            if interest and self.salient:
                for cue in list(self.salient):
                    ncue = self.G.nodes.get(cue)
                    if ncue is None or not ncue.active:
                        self.salient.discard(cue)
                        continue
                    if cue in e:
                        self.a_cue_and_interest[cue] = self.a_cue_and_interest.get(cue, 0.0) + 1.0
                    else:
                        self.c_interest_without_cue[cue] = self.c_interest_without_cue.get(cue, 0.0) + 1.0

        # ---- one split decision per episode ----
        split_cue: Optional[NodeId] = None
        best_score = -1.0

        for sid, node in self.G.nodes.items():
            if not node.active or sid in self.salient:
                continue
            if self.evidence(sid) < self.cfg.n_threshold:
                continue
            pc = self.pc(sid)
            if pc <= self.cfg.theta_split:
                continue
            score = pc * self.evidence(sid)
            if score > best_score:
                best_score = score
                split_cue = sid

        did_zipper = False
        if split_cue is not None:
            self.salient.add(split_cue)
            do_sweep = True
            if self.cfg.zipper_sweep_only_when_new_cue and (split_cue in self._cue_swept_once):
                do_sweep = False

            if do_sweep and (split_cue in latents):
                cue_idx = latents.index(split_cue)
                self._zipper_sweep_suffix(latents, action_seq, cue_idx, root_cue=split_cue)
                self._cue_swept_once.add(split_cue)
                did_zipper = True

        # ---- merge checks ----
        for cue in list(self.salient):
            node = self.G.nodes.get(cue)
            if node is None or not node.active:
                self.salient.discard(cue)
                continue
            u = self.utility(cue)
            if u > 0.0 and u < self.cfg.theta_merge:
                self._merge_from_cue(cue)

        diag = dict(
            episode=self.episode_idx,
            n_nodes=len(self.G.nodes),
            n_active=sum(1 for n in self.G.nodes.values() if n.active),
            n_salient=len(self.salient),
            n_interest_steps=n_interest_steps,
            split_cue=split_cue,
            did_zipper=did_zipper,
        )
        self.episode_idx += 1
        return diag

    # ------------------- Snapshot -------------------

    def snapshot(self, title: str = "") -> Snapshot:
        nodes = {sid: dict(obs=n.obs, active=n.active, original_sid=n.original_sid, parent_cue=n.parent_cue, created_at=n.created_at)
                 for sid, n in self.G.nodes.items()}
        edges = {(s, a): dict(succ) for (s, a), succ in self.G.counts.items()}
        return Snapshot(episode=self.episode_idx, title=title, nodes=nodes, edges=edges, salient=sorted(list(self.salient)))
