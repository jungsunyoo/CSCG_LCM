# viz/viz_rollout_trained.py
import os, sys, re, argparse, numpy as np, imageio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # so coda_rl + train_sb3 import
import train_sb3  # <-- needed so SB3 can unpickle the custom SmallCombinedExtractor
import gymnasium as gym, minigrid
from gymnasium import spaces
from stable_baselines3 import PPO
from coda_rl.coda_wrapper import CoDAWrapper

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def resolve_env_id(env_id: str) -> str:
    try:
        gym.spec(env_id)
        return env_id
    except Exception:
        base = re.sub(r"-v\\d+$", "", env_id)
        keys = list(gym.envs.registry.keys())
        cand = sorted([k for k in keys if k.startswith(base)], reverse=True)
        if not cand:
            raise
        print(f"[viz] {env_id} not found; using {cand[0]}")
        return cand[0]

class SB3FriendlyObs(gym.ObservationWrapper):
    """Keep numeric fields only; transpose image to CHW; direction -> (1,)."""
    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space["image"].shape
        img_sp = spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)
        dir_sp = spaces.Box(low=0, high=3, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({"image": img_sp, "direction": dir_sp})
    def observation(self, obs):
        img = np.transpose(obs["image"], (2,0,1)).copy()
        direction = np.array([obs["direction"]], dtype=np.float32)
        return {"image": img, "direction": direction}

def make_env(env_id: str, use_coda: bool, **coda_kwargs):
    env = gym.make(env_id, render_mode="rgb_array")
    env = SB3FriendlyObs(env)
    if use_coda:
        env = CoDAWrapper(env, **coda_kwargs)
    else:
        class ZeroTag(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                base = env.observation_space
                self.observation_space = spaces.Dict({
                    **base.spaces,
                    "coda": spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float32),
                })
            def observation(self, obs):
                return {**obs, "coda": np.zeros(8, dtype=np.float32)}
        env = ZeroTag(env)
    return env

def overlay(frame, text_lines):
    h, w, _ = frame.shape
    box_h = 16*len(text_lines)+10
    box_w = min(w, max(260, max(8*len(t) for t in text_lines)+20))
    frame[:box_h, :box_w, :] = (frame[:box_h, :box_w, :]*0.35).astype(np.uint8)
    fig = plt.figure(figsize=(box_w/100, box_h/100), dpi=100)
    ax = fig.add_axes([0,0,1,1]); ax.set_axis_off()
    for i, t in enumerate(text_lines):
        ax.text(0.02, 1.0-(i+1)/(len(text_lines)+0.5), t, color="w", fontsize=9,
                family="monospace", ha="left", va="center")
    fig.canvas.draw()
    w2, h2 = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h2, w2, 4)[...,:3]
    plt.close(fig)
    frame[0:h2, 0:w2, :] = buf
    return frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-6x6-v0")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--coda", action="store_true", help="use if the model was trained with CoDA")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="trained_rollout.gif")
    args = ap.parse_args()

    env_id = resolve_env_id(args.env_id)
    env = make_env(env_id, use_coda=args.coda, K=8, theta_split=0.9, theta_merge=0.5, credible=0.95)
    model = PPO.load(args.model)  # uses train_sb3.SmallCombinedExtractor via the import above

    writer = imageio.get_writer(args.out, fps=8)
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            done=trunc=False; total=0.0; steps=0
            while not(done or trunc) and steps<args.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(int(action))
                total += float(r); steps += 1
                frame = env.render()
                text = [f"Ep {ep+1}/{args.episodes}  Step {steps}  Return={total:.1f}",
                        f"Policy: {'CoDA' if args.coda else 'Baseline'}"]
                if args.coda and hasattr(env, 'salient'):
                    mk = env.markovianity_metrics(); bits = env.coda_bits()
                    text += [f"Salient={bits['num_salient']}  Cues={bits['num_cues']}  Space={env.coda_space()}B",
                             f"Markovianity: det={mk['frac_deterministic']:.2f}  nH={mk['norm_entropy']:.2f}"]
                frame = overlay(frame, text)
                writer.append_data(frame)
    finally:
        writer.close(); env.close()
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
