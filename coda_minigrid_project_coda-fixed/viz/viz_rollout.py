# --- add at the very top ---
import os, sys, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import minigrid                  # registers envs
import numpy as np
import imageio

from coda_rl.coda_wrapper import CoDAWrapper
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# import argparse, numpy as np, imageio, gymnasium as gym

def make_env(env_id, use_coda, **kw):
    def _make(id_):
        return gym.make(id_, render_mode="rgb_array")
    try:
        env = _make(env_id)
    except gym.error.NameNotFound:
        base = re.sub(r"-v\\d+$", "", env_id)
        keys = list(gym.envs.registry.keys())
        cand = sorted([k for k in keys if k.startswith(base)], reverse=True)
        if not cand:
            raise
        fallback = cand[0]
        print(f"[viz] {env_id} not found; using {fallback}")
        env = _make(fallback)
    if use_coda:
        env = CoDAWrapper(env, **kw)
    return env


def overlay_frame(frame, text_lines):
    h, w, _ = frame.shape
    box_h = 16*len(text_lines)+10; box_w = min(w, max(220, max(8*len(t) for t in text_lines)+20))
    overlay = frame.copy()
    overlay[:box_h, :box_w, :] = (overlay[:box_h, :box_w, :]*0.35).astype(np.uint8)
    return overlay, (box_h, box_w)

def draw_text(frame, text_lines, box_h, box_w):
    fig = plt.figure(figsize=(box_w/100, box_h/100), dpi=100)
    ax = fig.add_axes([0,0,1,1]); ax.set_axis_off()
    for i, t in enumerate(text_lines):
        ax.text(0.02, 1.0-(i+1)/(len(text_lines)+0.5), t,
                color="w", fontsize=9, family="monospace", ha="left", va="center")
    fig.canvas.draw()
    import numpy as _np
    w, h = fig.canvas.get_width_height()
    buf = _np.frombuffer(fig.canvas.buffer_rgba(), dtype=_np.uint8).reshape(h, w, 4)[..., :3]
    plt.close(fig)
    frame[0:h, 0:w, :] = buf
    return frame


def record(env_id, use_coda, episodes, max_steps, out_path):
    env = make_env(env_id, use_coda, K=8, theta_split=0.9, theta_merge=0.5, credible=0.95)
    writer = imageio.get_writer(out_path, fps=8)
    try:
        for ep in range(episodes):
            obs, info = env.reset()
            done=trunc=False; steps=0; total_r=0.0
            while not(done or trunc) and steps<max_steps:
                a = env.action_space.sample()
                obs, r, done, trunc, info = env.step(a)
                total_r += r; steps += 1
                frame = env.render()
                text = [f"Ep {ep+1}/{episodes}  Step {steps}  R={total_r:.1f}", f"CoDA: {'ON' if use_coda else 'OFF'}"]
                if use_coda and isinstance(env, CoDAWrapper):
                    mk = env.markovianity_metrics(); bits = env.coda_bits()
                    text += [f"Salient={bits['num_salient']}  Cues={bits['num_cues']}  Space={env.coda_space()}B",
                             f"Markovianity: det={mk['frac_deterministic']:.2f}  nH={mk['norm_entropy']:.2f}"]
                    if getattr(env, "salient", None):
                        sal_list = sorted(list(env.salient))
                        text += [("CUES*: " + ", ".join(sal_list))[:48]]
                frame, (bh,bw) = overlay_frame(frame, text)
                frame = draw_text(frame, text, bh, bw)
                writer.append_data(frame)
    finally:
        writer.close(); env.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-6x6-v0")
    ap.add_argument("--coda", action="store_true")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="rollout.gif")
    args = ap.parse_args()
    record(args.env_id, args.coda, args.episodes, args.max_steps, args.out)

if __name__ == "__main__":
    main()
