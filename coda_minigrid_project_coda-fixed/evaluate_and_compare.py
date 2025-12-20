#!/usr/bin/env python3
"""
evaluate_and_compare.py

Evaluate PPO baseline and CoDA models on MiniGrid-DoorKey-6x6-v0
for 30 episodes each. Applies RGBImgObsWrapper and ImgObsWrapper for
consistent normalization, wraps with CoDAWrapper for CoDA evaluation,
and converts observations from HWC → CHW before model.predict.

Outputs mean ± SE returns for both models and saves results to:
    runs/compare_returns.txt

Also saves per-episode returns to:
    runs/curves/baseline_seed{SEED}.txt
    runs/curves/coda_seed{SEED}.txt
"""

import os
import numpy as np
import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO

from coda_rl.coda_wrapper import CoDAWrapper

# --- global reseed so each process run honors SEED env var ---
import random
def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

seed_everything(int(os.getenv("SEED", "0")))
# --------------------------------------------------------------

def to_chw(obs):
    """Convert image observations from HWC to CHW format."""
    if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[-1] in (1, 3):
        return np.transpose(obs, (2, 0, 1))
    return obs

def make_env(env_id, use_coda=False):
    """Construct environment with training-style normalization."""
    env = gym.make(env_id)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    if use_coda:
        env = CoDAWrapper(env, expose_in_obs=True)
    return env

def preprocess_obs(obs, use_coda):
    """Prepare observation for model input (CHW; include CoDA branch if used)."""
    if use_coda:
        # obs is a dict from CoDAWrapper: {"obs": HWC image, "coda": <…>}
        return {"obs": to_chw(obs["obs"]), "coda": obs["coda"]}
    return to_chw(obs)

def evaluate(model_path, env_id, episodes=30, use_coda=False):
    """
    Run evaluation loop for N episodes and return (mean, se, returns_list).
    """
    env = make_env(env_id, use_coda)
    model = PPO.load(model_path)
    returns = []

    for _ in range(episodes):
        obs, _ = env.reset()
        obs = preprocess_obs(obs, use_coda)
        done = False
        total_r = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = preprocess_obs(next_obs, use_coda)
            total_r += reward
            done = terminated or truncated

        returns.append(total_r)

    returns = np.asarray(returns, dtype=float)
    mean = float(np.mean(returns))
    se = float(np.std(returns, ddof=1) / np.sqrt(len(returns))) if len(returns) > 1 else float("nan")
    return mean, se, returns

def main():
    env_id = "MiniGrid-DoorKey-6x6-v0"
    baseline_path = "runs/ppo_baseline_MiniGrid-DoorKey-6x6-v0_s0/model.zip"
    coda_path = "runs/ppo_coda_MiniGrid-DoorKey-6x6-v0_s0/model.zip"

    baseline_mean, baseline_se, baseline_returns = evaluate(baseline_path, env_id, 30, use_coda=False)
    coda_mean, coda_se, coda_returns = evaluate(coda_path, env_id, 30, use_coda=True)

    os.makedirs("runs", exist_ok=True)
    out_path = "runs/compare_returns.txt"
    with open(out_path, "w") as f:
        f.write(f"Env: {env_id}\nEpisodes: 30\n")
        f.write(f"Baseline mean ± SE: {baseline_mean:.3f} ± {baseline_se:.3f}\n")
        f.write(f"CoDA mean ± SE: {coda_mean:.3f} ± {coda_se:.3f}\n")

    print(f"Saved results to {out_path}")
    print(f"Baseline mean ± SE: {baseline_mean:.3f} ± {baseline_se:.3f}")
    print(f"CoDA mean ± SE: {coda_mean:.3f} ± {coda_se:.3f}")

    # --- save per-episode returns for learning-curve/AUC analysis ---
    os.makedirs("runs/curves", exist_ok=True)
    seed_val = int(os.getenv("SEED", "0"))
    np.savetxt(f"runs/curves/baseline_seed{seed_val}.txt", baseline_returns)
    np.savetxt(f"runs/curves/coda_seed{seed_val}.txt",      coda_returns)

if __name__ == "__main__":
    main()
