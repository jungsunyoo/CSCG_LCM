
import argparse, os, yaml, numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import TimeFeatureWrapper
import gymnasium as gym

from coda_rl.coda_wrapper import CoDAWrapper

def make_env(env_id:str, use_coda:bool, **coda_kwargs):
    def _init():
        env = gym.make(env_id)
        if use_coda:
            env = CoDAWrapper(env, **coda_kwargs)
        else:
            class ZeroTag(gym.ObservationWrapper):
                def __init__(self, env):
                    super().__init__(env)
                    base = env.observation_space
                    if isinstance(base, gym.spaces.Dict):
                        self.observation_space = gym.spaces.Dict({
                            **base.spaces,
                            "coda": gym.spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32)
                        })
                    else:
                        self.observation_space = gym.spaces.Dict({
                            "obs": base,
                            "coda": gym.spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32)
                        })
                def observation(self, obs):
                    import numpy as np
                    if isinstance(obs, dict):
                        return {**obs, "coda": np.zeros(8, dtype=np.float32)}
                    else:
                        return {"obs": obs, "coda": np.zeros(8, dtype=np.float32)}
            env = ZeroTag(env)
        return env
    return _init

class MetricsLogger:
    def __init__(self, env, path, freq=5000):
        self.env = env; self.path = path; self.freq = freq
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(self.path, "w") as f:
            f.write("timesteps,success_probe,frac_deterministic,norm_entropy,coda_space_bytes,num_salient,num_cues\n")
    def maybe_log(self, t):
        if t % self.freq != 0: return
        env = self.env.envs[0]
        succ = 0
        for _ in range(10):
            obs, info = env.reset()
            done=trunc=False; total_r=0.0
            while not (done or trunc):
                a = env.action_space.sample()
                obs, r, done, trunc, info = env.step(a); total_r += r
            succ += int(total_r > 0.0)
        success_rate = succ/10.0
        try:
            from coda_rl.coda_wrapper import CoDAWrapper as CW
            if isinstance(env, CW):
                m = env.markovianity_metrics(); s = env.coda_space(); bits = env.coda_bits()
            else:
                m = {"frac_deterministic": float("nan"), "norm_entropy": float("nan")}; s=0; bits={"num_salient":0,"num_cues":0}
        except Exception:
            m = {"frac_deterministic": float("nan"), "norm_entropy": float("nan")}; s=0; bits={"num_salient":0,"num_cues":0}
        with open(self.path, "a") as f:
            f.write(f"{t},{success_rate},{m['frac_deterministic']},{m['norm_entropy']},{s},{bits['num_salient']},{bits['num_cues']}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="path to YAML config")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    env_id = cfg.get("env_id", "MiniGrid-DoorKey-6x6-v0")
    algo = cfg.get("algo", "ppo")
    use_coda = bool(cfg.get("coda", False))
    steps = int(cfg.get("steps", 200_000))
    seed = int(cfg.get("seed", 0))
    logdir = cfg.get("logdir", "runs")
    coda_params = cfg.get("coda_params", {}) if use_coda else {}

    np.random.seed(seed)
    env_fn = make_env(env_id, use_coda=use_coda, **coda_params)
    from sb3_contrib.common.wrappers import TimeFeatureWrapper
    vec_env = DummyVecEnv([env_fn])
    vec_env = TimeFeatureWrapper(vec_env)

    run_name = f"{env_id}_{algo}_{'coda' if use_coda else 'nocoda'}_seed{seed}"
    out_dir = os.path.join(logdir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    metrics = MetricsLogger(vec_env, os.path.join(out_dir, "metrics.csv"), freq=5000)

    if algo == "ppo":
        model = PPO("MultiInputPolicy", vec_env, verbose=1, seed=seed, n_steps=1024, batch_size=256, ent_coef=0.01)
    elif algo == "rppo":
        model = RecurrentPPO("MultiInputLstmPolicy", vec_env, verbose=1, seed=seed, n_steps=1024, batch_size=256, ent_coef=0.01)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    timesteps = 0; chunk = 5000
    while timesteps < steps:
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        timesteps += chunk
        metrics.maybe_log(timesteps)

    model.save(os.path.join(out_dir, "model.zip"))
    print("Saved to", out_dir)

if __name__ == "__main__":
    main()
