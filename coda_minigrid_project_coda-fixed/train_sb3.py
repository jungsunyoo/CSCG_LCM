
import argparse, os, yaml, time, datetime as dt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import minigrid  # registers env ids
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from coda_rl.coda_wrapper import CoDAWrapper

class CSVLogger(BaseCallback):
    def __init__(self, env, log_path, check_freq=5000):
        super().__init__()
        self.env_ref = env
        self.log_path = log_path
        self.check_freq = check_freq
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("timesteps,frac_deterministic,norm_entropy,num_states,num_salient\n")
    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            try:
                stats = self.training_env.envs[0].env.coda_stats()
            except Exception:
                stats = {"frac_deterministic": np.nan, "norm_entropy": np.nan, "num_states": np.nan, "num_salient": np.nan}
            with open(self.log_path, "a") as f:
                f.write(f"{self.num_timesteps},{stats['frac_deterministic']},{stats['norm_entropy']},{stats['num_states']},{stats['num_salient']}\n")
        return True

def make_env(env_id, seed, use_coda, coda_params, feature_dim):
    def _thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)

        # Normalize MiniGrid observations to a pure Box image so SB3 can handle it robustly
        # This removes the non-numpy-flattenable "mission"/Text field.
        env = RGBImgObsWrapper(env)   # put pixel obs into the observation dict under key "image"
        env = ImgObsWrapper(env)      # drop dict and expose only the image (Box)

        if use_coda:
            # CoDA wrapper will produce a Dict({"obs": Box, "coda": Box}) expected by MultiInputPolicy
            env = CoDAWrapper(env, coda_cfg=coda_params or {}, feature_dim=feature_dim, expose_in_obs=True)
        # Baseline path now returns a pure Box observation (MlpPolicy),
        # CoDA path returns a Dict observation (MultiInputPolicy).
        return env
    return _thunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config file")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    env_id = cfg.get("env_id", "MiniGrid-DoorKey-6x6-v0")
    algo   = cfg.get("algo", "ppo")
    steps  = int(cfg.get("steps", 200_000))
    seed   = int(cfg.get("seed", 0))
    logdir = cfg.get("logdir", "runs")
    use_coda = bool(cfg.get("coda", False))
    coda_params = cfg.get("coda_params", {})
    feature_dim = int(cfg.get("feature_dim", 8))

    log_dir = os.path.join(logdir, f"{algo}_{'coda' if use_coda else 'baseline'}_{env_id.replace('/','-')}_s{seed}")
    os.makedirs(log_dir, exist_ok=True)

    # env
    env = DummyVecEnv([make_env(env_id, seed, use_coda, coda_params, feature_dim)])
    # after creating env = DummyVecEnv([...])
    # from gymnasium import spaces
    obs_space = env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space

    if isinstance(obs_space, spaces.Dict):
        policy = "MultiInputPolicy"
    elif isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 3:
        policy = "CnnPolicy"     # <-- IMPORTANT: images -> CNN
    else:
        policy = "MlpPolicy"
    print(f"[train] Selected policy: {policy}  |  obs_space: {obs_space}")
    model = PPO(policy, env, verbose=1, seed=seed, tensorboard_log=log_dir)
    # # Pick the right policy for the observation space (Dict -> MultiInput, Box -> Mlp)
    # obs_space = env.single_observation_space if hasattr(env, "single_observation_space") else env.observation_space
    # policy = "MultiInputPolicy" if isinstance(obs_space, spaces.Dict) else "MlpPolicy"

    # model = PPO(policy, env, verbose=1, seed=seed, tensorboard_log=log_dir)

    # logging callback
    logger = CSVLogger(env, os.path.join(log_dir, "metrics.csv"), check_freq=5000)

    # train
    start = time.time()
    model.learn(total_timesteps=steps, callback=logger)
    dur = time.time() - start
    print(f"Finished training in {dur/60:.1f} minutes. Logs -> {log_dir}")
    model.save(os.path.join(log_dir, "model.zip"))

if __name__ == "__main__":
    main()
