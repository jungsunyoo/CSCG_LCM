#!/usr/bin/env python3
"""
eval_sb3.py

Evaluate a Stable-Baselines3 PPO model on a MiniGrid environment using
the same observation normalization as during training.

- Applies RGBImgObsWrapper and ImgObsWrapper for consistent observation preprocessing.
- When --coda is enabled, wraps with CoDAWrapper.
- Converts all image observations from HWC → CHW before model.predict calls.
"""

import argparse
import numpy as np
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO

try:
    from coda_rl.coda_wrapper import CoDAWrapper
except ImportError:
    CoDAWrapper = None


def to_chw(obs):
    """Convert image observations from HWC to CHW format."""
    if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[-1] in (1, 3):
        return np.transpose(obs, (2, 0, 1))
    return obs


def make_env(env_id: str, use_coda: bool):
    """Construct environment with the same normalization as training."""
    env = gym.make(env_id)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    if use_coda:
        if CoDAWrapper is None:
            raise ImportError("CoDAWrapper not found. Install coda_rl to use --coda.")
        env = CoDAWrapper(env, expose_in_obs=True)
    return env


def preprocess_obs(obs, use_coda: bool):
    """Prepare observation for model input."""
    if use_coda:
        return {"obs": to_chw(obs["obs"]), "coda": obs["coda"]}
    return to_chw(obs)


def evaluate(model_path: str, env_id: str, episodes: int, use_coda: bool):
    """Evaluate a PPO model over multiple episodes."""
    env = make_env(env_id, use_coda)
    model = PPO.load(model_path)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = preprocess_obs(obs, use_coda)

        # DEBUG: one-time shape check
        if ep == 0:
            if use_coda:
                print("[DEBUG] eval obs CHW shape:", obs["obs"].shape, "coda shape:", obs["coda"].shape, "dtype:", obs["obs"].dtype)
            else:
                print("[DEBUG] eval obs CHW shape:", obs.shape, "dtype:", obs.dtype)

        # --- Optional sanity print; comment out later ---
        # if use_coda:
        #     print("Eval shapes:", obs["obs"].shape, obs["coda"].shape)
        # else:
        #     print("Eval shape:", obs.shape)

        done = False
        total_r = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = preprocess_obs(next_obs, use_coda)
            total_r += reward
            done = terminated or truncated

        returns.append(total_r)
        print(f"Episode {ep+1}: return = {total_r:.3f}")

    returns = np.asarray(returns, dtype=float)
    mean = float(np.mean(returns))
    se = float(np.std(returns, ddof=1) / np.sqrt(max(1, len(returns))))
    print(f"\nMean ± SE: {mean:.3f} ± {se:.3f}")
    return mean, se


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO model on MiniGrid env.")
    parser.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-6x6-v0")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--episodes", type=int, default=30, help="Number of evaluation episodes")
    parser.add_argument("--coda", action="store_true", help="Enable CoDAWrapper during evaluation")
    args = parser.parse_args()

    evaluate(args.model, args.env_id, args.episodes, args.coda)


if __name__ == "__main__":
    main()
