
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List
from .coda_core import CoDAAgent, CoDAConfig

def _hash_obs(obs) -> bytes:
    # robust hash for dict or ndarray obs
    if isinstance(obs, dict):
        # only include stable keys
        items = []
        for k in sorted(obs.keys()):
            v = obs[k]
            if isinstance(v, np.ndarray):
                items.append((k, v.tobytes()))
            elif np.isscalar(v):
                items.append((k, bytes(str(v), 'utf-8')))
            else:
                try:
                    items.append((k, np.asarray(v).tobytes()))
                except Exception:
                    items.append((k, bytes(repr(v), 'utf-8')))
        return b'|'.join([bytes(k,'utf-8')+b':'+vv for k,vv in items])
    elif isinstance(obs, np.ndarray):
        return obs.tobytes()
    else:
        return bytes(repr(obs), 'utf-8')

class CoDAWrapper(gym.Wrapper):
    """
    Wrap a MiniGrid env to compute CoDA online and optionally expose a compact
    K-dim 'coda' feature in the observation dict.
    """
    def __init__(self, env: gym.Env, coda_cfg: Dict = None, feature_dim: int = 8, expose_in_obs: bool = True):
        super().__init__(env)
        self.cfg = CoDAConfig(**(coda_cfg or {}))
        # Determine action space size
        if isinstance(self.env.action_space, spaces.Discrete):
            n_actions = int(self.env.action_space.n)
        else:
            raise ValueError("CoDAWrapper expects Discrete action space.")
        self.agent = CoDAAgent(n_actions=n_actions, cfg=self.cfg)
        self.expose_in_obs = expose_in_obs
        self.feature_dim = int(feature_dim)

        # extend observation space
        if expose_in_obs:
            base = self.env.observation_space
            self.observation_space = spaces.Dict({
                "obs": base if not isinstance(base, spaces.Dict) else base,
                "coda": spaces.Box(low=0.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32)
            })

        # buffers for one episode
        self._ep_obs_hashes: List[bytes] = []
        self._ep_actions: List[int] = []
        self._ep_rewards: List[float] = []

    def _coda_feature(self, obs_hash: bytes) -> np.ndarray:
        """
        Simple, stable feature: random Fourier-like hash mapped to [0,1]^K;
        importantly, this is *only* for exposing a compact feature to the policy.
        CoDA's split/merge is entirely in self.agent.
        """
        h = hash(obs_hash) & ((1<<64)-1)
        rng = np.random.default_rng(h)
        v = rng.random(self.feature_dim).astype(np.float32)
        # Add two interpretable dims: [is_salient, pc[s]]
        sid = self.agent._obs_to_sid.get(obs_hash, -1)
        pc = 0.0; sal = 0.0
        if sid >= 0 and sid < self.agent.n_states:
            pc = float(self.agent.prospective()[sid])
            sal = 1.0 if sid in self.agent.salient_cues else 0.0
        if self.feature_dim >= 2:
            v[0] = sal; v[1] = pc
        return v

    def reset(self, **kwargs):
        self._ep_obs_hashes.clear()
        self._ep_actions.clear()
        self._ep_rewards.clear()
        obs, info = self.env.reset(**kwargs)
        h = _hash_obs(obs)
        self._ep_obs_hashes.append(h)
        if self.expose_in_obs:
            return {"obs": obs, "coda": self._coda_feature(h)}, info
        else:
            return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # record transition
        h_next = _hash_obs(obs)
        self._ep_actions.append(int(action))
        self._ep_rewards.append(float(reward))
        self._ep_obs_hashes.append(h_next)

        # episode boundary -> update CoDA + split/merge
        if terminated or truncated:
            self.agent.update_with_episode(self._ep_obs_hashes, self._ep_actions, self._ep_rewards)
            self.agent.maybe_split()
            # We call merge lightly every episode with a small structural epsilon
            self.agent.maybe_merge(edge_eps=1e-6)

        if self.expose_in_obs:
            return {"obs": obs, "coda": self._coda_feature(h_next)}, reward, terminated, truncated, info
        else:
            return obs, reward, terminated, truncated, info

    # expose some debugging / metrics
    def coda_stats(self) -> Dict[str, float]:
        return self.agent.stats(thresh=0.9)
