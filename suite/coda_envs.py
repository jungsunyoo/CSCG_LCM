
"""
coda_envs.py
============

Environments used in the demo notebooks.

1) Cued gridworld:
   - Random right/down walk.
   - Terminal observation is "T_R" vs "T_NR" depending on whether cue_cell was visited.
   This makes the last transition appear stochastic at the observation level.

2) Near/Far sequences:
   Uses your encoding exactly.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Hashable
import numpy as np

Obs = Hashable


@dataclass
class CuedGridworldEnv:
    rows: int = 5
    cols: int = 5
    cue_cell: int = 14
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.n_cells = self.rows * self.cols
        self.start = 1
        self.terminal_cell = self.n_cells

    def _neighbors_right_down(self, cell: int):
        r = (cell - 1) // self.cols
        c = (cell - 1) % self.cols
        out = []
        if c + 1 < self.cols:
            out.append(("R", cell + 1))
        if r + 1 < self.rows:
            out.append(("D", cell + self.cols))
        return out

    def sample_episode(self, max_steps: int = 200) -> Tuple[List[Obs], List[str]]:
        obs: List[Obs] = [self.start]
        acts: List[str] = []
        cur = self.start
        for _ in range(max_steps):
            if cur == self.terminal_cell:
                break
            neigh = self._neighbors_right_down(cur)
            a, nxt = neigh[self.rng.integers(0, len(neigh))]
            acts.append(a)
            cur = nxt
            obs.append(cur)
            if cur == self.terminal_cell:
                break

        has_cue = (self.cue_cell in obs)
        obs2 = obs[:-1] + (["T_R"] if has_cue else ["T_NR"])
        return obs2, acts


@dataclass
class NearFarSeqEnv:
    seed: int = 0
    p_near: float = 0.5

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.near = [1,1,1,1,1,1, 2,2,2,2, 1,1,1, 4,6, 1,1,1, 5,5, 1,1, 7, 0,0,0]
        self.far  = [1,1,1,1,1,1, 3,3,3,3, 1,1,1, 4,4, 1,1,1, 5,6, 1,1, 7, 0,0,0]

    def sample_episode(self) -> Tuple[List[int], Optional[List[str]], str]:
        is_near = (self.rng.random() < self.p_near)
        seq = self.near if is_near else self.far
        return list(seq), None, ("near" if is_near else "far")
