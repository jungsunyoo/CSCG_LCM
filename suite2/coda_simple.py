
"""
coda_simple.py
==============

An *intentionally minimal* CoDA-style agent that keeps the core ideas:

1) DISCOVERY (reward-free): "interest transitions" are discovered via *transition entropy*
   over (state, action) -> next_state distributions.

2) INFERENCE (eligibility traces): build online eligibility e_t and accumulate
   event-conditioned evidence:
      E_int[s]   += e_t[s]   if current step is interest
      E_nint[s]  += e_t[s]   otherwise
   Prospective contingency:
      PC(s) = P(interest | s) ≈ E_int[s] / (E_int[s] + E_nint[s])

3) SPLITTING + ZIPPERING (iterative):
   - ONE cue decision per episode (best new cue by score).
   - When a cue is chosen, zipper over the *experienced suffix* of that episode:
       for each visited state on the suffix, clone its *experienced* one-step successors
       (all actions with counts) and redirect to those clones.
   - To prevent "clone explosion", zipper clones are REUSED:
       (cue, src, action, old_next) -> clone_next
     and each (cue, src) is swept only once.

4) MERGING (paper-style, minimal):
   Maintain a retrospective contingency for salient cues:
      RC(cue) = P(cue present | interest)
   Utility = PC * RC.
   Each episode, if a salient cue's utility drops below theta_merge,
   we undo redirects made under that cue and deactivate its descendants.

Notes
-----
- This class is designed to work with the `spatial_environments.py` envs you uploaded,
  especially `GridEnvRightDownNoSelf`.
- It avoids extra modeling layers, context tags, particle filters, etc.

Dependencies: numpy
Optional (for your notebooks): networkx/matplotlib for plotting (in coda_viz_simple.py)

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Hashable, Optional, List, Set, Any, Iterable
import math
import numpy as np

Obs = Hashable
Action = Hashable
NodeId = int


@dataclass
class ETConfig:
    gamma: float = 0.95
    lam: float = 0.9
    mode: str = "first_visit"  # "first_visit" | "accumulating" | "replacing"


@dataclass
class CoDAConfig:
    # discovery (entropy)
    min_sa_count: int = 10
    entropy_threshold: float = 0.45  # nats
    eps: float = 1e-12

    # cue selection
    n_threshold: float = 6.0
    theta_split: float = 0.80

    # merge
    theta_merge: float = 0.40

    # zipper controls
    zipper_once_per_cue: bool = True
    max_nodes: int = 50000


@dataclass
class Node:
    sid: NodeId
    obs: Obs
    active: bool = True
    original_sid: Optional[NodeId] = None
    parent_cue: Optional[NodeId] = None


@dataclass
class Snapshot:
    title: str
    nodes: Dict[int, Dict[str, Any]]
    edges: Dict[Tuple[int, Any], Dict[int, float]]
    salient: List[int]


def entropy_from_counts(counts: Dict[Any, float], eps: float = 1e-12) -> float:
    tot = float(sum(counts.values()))
    if tot <= 0:
        return 0.0
    ps = np.array([c / tot for c in counts.values()], dtype=float)
    ps = ps[ps > eps]
    if ps.size == 0:
        return 0.0
    return float(-(ps * np.log(ps)).sum())


class CoDASimple:
    def __init__(self, cfg: CoDAConfig = CoDAConfig(), et: ETConfig = ETConfig()):
        self.cfg = cfg
        self.et = et

        self._next_sid: int = 1
        self.nodes: Dict[NodeId, Node] = {}

        # counts[(s,a)][s_next] = count
        self.counts: Dict[Tuple[NodeId, Action], Dict[NodeId, float]] = {}

        # cache mapping (s,a,next_obs)->sid to stabilize assignments
        self.cache: Dict[Tuple[NodeId, Action, Obs], NodeId] = {}

        # evidence for PC
        self.E_int: Dict[NodeId, float] = {}
        self.E_nint: Dict[NodeId, float] = {}

        # salient cues + RC evidence
        self.salient: Set[NodeId] = set()
        self.A_cue_and_int: Dict[NodeId, float] = {}
        self.C_int_without_cue: Dict[NodeId, float] = {}

        # split/merge bookkeeping under a cue
        # list of redirects: (src, action, old_next, clone_next)
        self.redirects: Dict[NodeId, List[Tuple[NodeId, Action, NodeId, NodeId]]] = {}
        self.descendants: Dict[NodeId, Set[NodeId]] = {}

        # reuse clones (prevents clone explosion)
        self.clone_map: Dict[Tuple[NodeId, NodeId, Action, NodeId], NodeId] = {}
        self.swept_under_cue: Dict[NodeId, Set[NodeId]] = {}
        self.cue_has_swept: Set[NodeId] = set()

    # ---------------- nodes ----------------
    def new_node(self, obs: Obs, *, original_sid: Optional[NodeId] = None, parent_cue: Optional[NodeId] = None) -> NodeId:
        if len(self.nodes) >= self.cfg.max_nodes:
            raise RuntimeError(f"Exceeded max_nodes={self.cfg.max_nodes}")
        sid = self._next_sid
        self._next_sid += 1
        self.nodes[sid] = Node(sid=sid, obs=obs, active=True, original_sid=original_sid, parent_cue=parent_cue)
        return sid

    def ensure_base_node(self, obs: Obs) -> NodeId:
        for sid, n in self.nodes.items():
            if n.active and n.obs == obs and n.original_sid is None and n.parent_cue is None:
                return sid
        return self.new_node(obs)

    def active(self, sid: NodeId) -> bool:
        n = self.nodes.get(sid)
        return (n is not None) and n.active

    # ---------------- transitions ----------------
    def add_transition(self, s: NodeId, a: Action, s_next: NodeId, w: float = 1.0) -> None:
        d = self.counts.setdefault((s, a), {})
        d[s_next] = d.get(s_next, 0.0) + float(w)

    def successors(self, s: NodeId, a: Action) -> Dict[NodeId, float]:
        return self.counts.get((s, a), {})

    def total_sa(self, s: NodeId, a: Action) -> float:
        return float(sum(self.successors(s, a).values()))

    def redirect_successors(self, s: NodeId, a: Action, mapping: Dict[NodeId, NodeId]) -> None:
        d = self.counts.get((s, a))
        if not d:
            return
        newd: Dict[NodeId, float] = {}
        for old_next, c in d.items():
            new_next = mapping.get(old_next, old_next)
            newd[new_next] = newd.get(new_next, 0.0) + c
        self.counts[(s, a)] = newd

    def step_assign(self, s: NodeId, a: Action, next_obs: Obs) -> NodeId:
        key = (s, a, next_obs)
        if key in self.cache:
            sid_next = self.cache[key]
            if self.active(sid_next) and self.nodes[sid_next].obs == next_obs:
                self.add_transition(s, a, sid_next, 1.0)
                return sid_next

        # try existing successors
        for sid_next in self.successors(s, a).keys():
            if self.active(sid_next) and self.nodes[sid_next].obs == next_obs:
                self.cache[key] = sid_next
                self.add_transition(s, a, sid_next, 1.0)
                return sid_next

        sid_new = self.new_node(next_obs)
        self.cache[key] = sid_new
        self.add_transition(s, a, sid_new, 1.0)
        return sid_new

    # ---------------- interest discovery ----------------
    def entropy_sa(self, s: NodeId, a: Action) -> float:
        # use next OBS distribution (simple, robust)
        d_obs: Dict[Obs, float] = {}
        for sid_next, c in self.successors(s, a).items():
            if not self.active(sid_next):
                continue
            o = self.nodes[sid_next].obs
            d_obs[o] = d_obs.get(o, 0.0) + float(c)
        return entropy_from_counts(d_obs, eps=self.cfg.eps)

    def is_interest(self, s: NodeId, a: Action) -> bool:
        if self.total_sa(s, a) < self.cfg.min_sa_count:
            return False
        return self.entropy_sa(s, a) >= self.cfg.entropy_threshold

    # ---------------- eligibility ----------------
    def _eligibility_update(self, e: Dict[NodeId, float], visited: Set[NodeId], s: NodeId) -> None:
        decay = self.et.gamma * self.et.lam
        for k in list(e.keys()):
            e[k] *= decay
            if e[k] < 1e-10:
                del e[k]

        if self.et.mode == "accumulating":
            e[s] = e.get(s, 0.0) + 1.0
        elif self.et.mode == "replacing":
            e[s] = 1.0
        elif self.et.mode == "first_visit":
            if s not in visited:
                e[s] = e.get(s, 0.0) + 1.0
                visited.add(s)
        else:
            raise ValueError(f"Unknown eligibility mode: {self.et.mode}")

    # ---------------- PC/RC/utility ----------------
    def evidence(self, s: NodeId) -> float:
        return self.E_int.get(s, 0.0) + self.E_nint.get(s, 0.0)

    def pc(self, s: NodeId) -> float:
        i = self.E_int.get(s, 0.0)
        ni = self.E_nint.get(s, 0.0)
        d = i + ni
        return 0.0 if d <= self.cfg.eps else i / d

    def rc(self, cue: NodeId) -> float:
        a = self.A_cue_and_int.get(cue, 0.0)
        c = self.C_int_without_cue.get(cue, 0.0)
        d = a + c
        return 0.0 if d <= self.cfg.eps else a / d

    def utility(self, cue: NodeId) -> float:
        return self.pc(cue) * self.rc(cue)

    # ---------------- zipper splitting ----------------
    def _clone(self, sid: NodeId, *, parent_cue: NodeId) -> NodeId:
        base = self.nodes[sid]
        return self.new_node(base.obs, original_sid=sid, parent_cue=parent_cue)

    def _split_state_one_step(self, src: NodeId, *, cue: NodeId) -> None:
        # do once per (cue, src)
        swept = self.swept_under_cue.setdefault(cue, set())
        if src in swept:
            return
        swept.add(src)

        # for every action leaving src, clone all experienced successors
        for (s, a), succ in list(self.counts.items()):
            if s != src:
                continue
            mapping: Dict[NodeId, NodeId] = {}
            for old_next in list(succ.keys()):
                if not self.active(old_next):
                    continue
                key = (cue, src, a, old_next)
                clone_next = self.clone_map.get(key)
                if clone_next is None or (not self.active(clone_next)):
                    clone_next = self._clone(old_next, parent_cue=cue)
                    self.clone_map[key] = clone_next
                    self.redirects.setdefault(cue, []).append((src, a, old_next, clone_next))
                    self.descendants.setdefault(cue, set()).add(clone_next)
                mapping[old_next] = clone_next
            if mapping:
                self.redirect_successors(src, a, mapping)

    def zipper_sweep_suffix(self, latents: List[NodeId], actions: List[Action], cue_sid: NodeId) -> None:
        if cue_sid not in latents:
            return
        idx = latents.index(cue_sid)
        for t in range(idx, len(actions)):
            self._split_state_one_step(latents[t], cue=cue_sid)

    # ---------------- merge ----------------
    def merge_cue(self, cue: NodeId) -> None:
        recs = self.redirects.get(cue, [])
        # revert redirects per (src,a): clone -> old
        by_sa: Dict[Tuple[NodeId, Action], Dict[NodeId, NodeId]] = {}
        for src, a, old_next, clone_next in recs:
            by_sa.setdefault((src, a), {})[clone_next] = old_next
        for (src, a), mapping in by_sa.items():
            d = self.counts.get((src, a))
            if not d:
                continue
            newd: Dict[NodeId, float] = {}
            for nxt, c in d.items():
                old = mapping.get(nxt, nxt)
                newd[old] = newd.get(old, 0.0) + c
            self.counts[(src, a)] = newd

        # deactivate descendants
        for sid in self.descendants.get(cue, set()):
            if sid in self.nodes:
                self.nodes[sid].active = False
        self.salient.discard(cue)

        # cleanup per-cue bookkeeping
        self.redirects.pop(cue, None)
        self.descendants.pop(cue, None)
        self.swept_under_cue.pop(cue, None)
        for k in list(self.clone_map.keys()):
            if k[0] == cue:
                del self.clone_map[k]
        self.cue_has_swept.discard(cue)

    # ---------------- candidate cue filtering ----------------
    @staticmethod
    def action_constrained_states(env) -> Set[int]:
        """
        Paper-matching heuristic: exclude states where action availability is constrained
        (edges/borders) because they can spuriously look predictive.

        Works with envs that implement get_valid_actions(state).
        """
        constrained = set()
        # try to infer state set
        if hasattr(env, "state_to_pos"):
            states = list(env.state_to_pos.keys())
        else:
            # fallback: assume states 0..(n_states-1)
            n_states = getattr(env, "n_states", 0)
            states = list(range(n_states))
        # determine max number of actions
        maxA = 0
        valid_map = {}
        for s in states:
            try:
                acts = env.get_valid_actions(s)
            except TypeError:
                # some envs use get_valid_actions() for current state only
                env.current_state = s
                acts = env.get_valid_actions()
            valid_map[s] = acts
            maxA = max(maxA, len(acts))
        for s, acts in valid_map.items():
            if len(acts) < maxA:
                constrained.add(s)
        return constrained

    # ---------------- main loop ----------------
    def run_episode(self,
                    obs_seq: List[Obs],
                    action_seq: Optional[List[Action]] = None,
                    exclude_states: Optional[Set[Obs]] = None) -> Dict[str, Any]:
        """
        Update the model on a single episode given an observation sequence and optional actions.
        One split decision per episode. Zipper sweep over suffix (experienced).
        """
        if action_seq is None:
            action_seq = ["next"] * (len(obs_seq) - 1)
        if len(action_seq) != len(obs_seq) - 1:
            raise ValueError("action_seq length must be len(obs_seq)-1")

        # build latent trajectory
        cur = self.ensure_base_node(obs_seq[0])
        latents = [cur]
        for a, o_next in zip(action_seq, obs_seq[1:]):
            cur = self.step_assign(cur, a, o_next)
            latents.append(cur)

        # online eligibility + evidence
        e: Dict[NodeId, float] = {}
        visited: Set[NodeId] = set()
        interest_steps = 0

        for t in range(len(action_seq)):
            s = latents[t]
            a = action_seq[t]
            self._eligibility_update(e, visited, s)

            interest = self.is_interest(s, a)
            if interest:
                interest_steps += 1

            tgt = self.E_int if interest else self.E_nint
            for sid, w in e.items():
                tgt[sid] = tgt.get(sid, 0.0) + w

            # RC for salient cues: count presence at interest moments
            if interest and self.salient:
                for cue in list(self.salient):
                    if not self.active(cue):
                        self.salient.discard(cue)
                        continue
                    if cue in e:
                        self.A_cue_and_int[cue] = self.A_cue_and_int.get(cue, 0.0) + 1.0
                    else:
                        self.C_int_without_cue[cue] = self.C_int_without_cue.get(cue, 0.0) + 1.0

        # choose ONE new cue
        split_cue: Optional[NodeId] = None
        best = -1.0

        for sid, n in self.nodes.items():
            if (not n.active) or (sid in self.salient):
                continue
            if self.evidence(sid) < self.cfg.n_threshold:
                continue
            if exclude_states is not None and (n.obs in exclude_states):
                continue
            pc = self.pc(sid)
            if pc <= self.cfg.theta_split:
                continue
            score = pc * self.evidence(sid)
            if score > best:
                best = score
                split_cue = sid

        did_zipper = False
        if split_cue is not None:
            self.salient.add(split_cue)
            if (not self.cfg.zipper_once_per_cue) or (split_cue not in self.cue_has_swept):
                self.zipper_sweep_suffix(latents, action_seq, split_cue)
                self.cue_has_swept.add(split_cue)
                did_zipper = True

        # merge checks
        merged = []
        for cue in list(self.salient):
            if not self.active(cue):
                self.salient.discard(cue)
                continue
            u = self.utility(cue)
            if u > 0.0 and u < self.cfg.theta_merge:
                self.merge_cue(cue)
                merged.append(cue)

        return dict(
            interest_steps=interest_steps,
            split_cue=split_cue,
            did_zipper=did_zipper,
            merged=merged,
            n_nodes=len(self.nodes),
            n_active=sum(1 for n in self.nodes.values() if n.active),
            n_salient=len(self.salient),
        )

    def snapshot(self, title: str = "") -> Snapshot:
        nodes = {sid: dict(obs=n.obs, active=n.active, original_sid=n.original_sid, parent_cue=n.parent_cue)
                 for sid, n in self.nodes.items()}
        edges = {(s, a): dict(succ) for (s, a), succ in self.counts.items()}
        return Snapshot(title=title, nodes=nodes, edges=edges, salient=sorted(self.salient))
