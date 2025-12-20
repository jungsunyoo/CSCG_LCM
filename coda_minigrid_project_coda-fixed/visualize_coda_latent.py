#!/usr/bin/env python3
"""
Visualize latent state representations learned by a CoDA agent.

For each episode:
  • Save a PNG "latent_raster_ep{E}.png" showing the discrete CoDA state over time
    and a simple transition-matrix heatmap from that episode.
  • Optionally save an animation "episode_ep{E}.mp4" (or .gif fallback)
    with the environment frame on the left and the current CoDA latent vector on the right.
  • Optionally save a t-SNE scatter "tsne_ep{E}.png" if scikit-learn is available.

Usage (example):
  python visualize_coda_latent.py \
      --env_id MiniGrid-DoorKey-6x6-v0 \
      --model runs/ppo_coda_MiniGrid-DoorKey-6x6-v0_s0/model.zip \
      --episodes 5 --max_steps 200 --out runs/latent_viz --animate --fps 6
"""

import os
import sys
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from stable_baselines3 import PPO
except Exception as e:
    raise SystemExit("This script expects Stable-Baselines3 PPO to load the model. "
                     "Install stable-baselines3 and try again.") from e

# ---- Optional t-SNE (if available) ----
try:
    from sklearn.manifold import TSNE
    HAVE_TSNE = True
except Exception:
    HAVE_TSNE = False

# ---- Wrappers (MiniGrid + your CoDA wrapper) ----
try:
    from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
except Exception as e:
    raise SystemExit("This script expects gym-minigrid (MiniGrid) wrappers. "
                     "Install gym-minigrid / minigrid and try again.") from e

# Your CoDA wrapper should expose 'coda' in the observation dict.
try:
    from coda_rl.coda_wrapper import CoDAWrapper
except Exception as e:
    raise SystemExit("Could not import coda_rl.coda_wrapper.CoDAWrapper. "
                     "Make sure your CoDA wrapper is installed/importable.") from e

# -------------------------
# Utilities
# -------------------------

def seed_everything(seed: int):
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

def to_chw(obs):
    """Convert HWC image to CHW (if needed) for the model."""
    if isinstance(obs, np.ndarray) and obs.ndim == 3 and obs.shape[-1] in (1, 3):
        return np.transpose(obs, (2, 0, 1))
    return obs

def make_env(env_id: str, use_coda: bool = True, render: bool = True):
    kwargs = {"render_mode": "rgb_array"} if render else {}
    env = gym.make(env_id, **kwargs)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    if use_coda:
        env = CoDAWrapper(env, expose_in_obs=True)
    return env

def preprocess_obs(obs, use_coda: bool = True):
    """Prepare observation for the policy; keep 'coda' as-is for visualization."""
    if use_coda:
        # CoDAWrapper returns {"obs": image, "coda": ...}
        return {"obs": to_chw(obs["obs"]), "coda": obs["coda"]}
    return to_chw(obs)

def extract_coda_latent(coda_obj):
    """
    Robustly parse 'obs["coda"]'.
    Returns:
      state_id: int or None
      vec: 1D numpy array or None
    Tries common patterns:
      - integer state id
      - 1D array (one-hot or logits) -> argmax
      - dict with keys ['state_id','clone_id'] or ['features','onehot','z','vec']
    """
    state_id, vec = None, None

    # dict case
    if isinstance(coda_obj, dict):
        # Look for an explicit id
        for k in ["state_id", "clone_id", "node", "id"]:
            if k in coda_obj:
                try:
                    state_id = int(coda_obj[k])
                    break
                except Exception:
                    pass
        # Look for a vector
        if vec is None:
            for k in ["features", "onehot", "z", "vec"]:
                if k in coda_obj:
                    try:
                        arr = np.asarray(coda_obj[k]).reshape(-1)
                        vec = arr
                        break
                    except Exception:
                        pass
        if state_id is None and vec is not None:
            state_id = int(np.argmax(vec))
        return state_id, vec

    # array-like case
    try:
        arr = np.asarray(coda_obj)
        if arr.ndim == 0:
            state_id = int(arr)
            vec = None
        else:
            vec = arr.reshape(-1)
            state_id = int(np.argmax(vec))
        return state_id, vec
    except Exception:
        return None, None

def plot_episode_png(ep_idx, state_ids, trans_mat, out_dir):
    """
    Save a PNG with:
      top: raster of state id across time
      bottom: transition matrix heatmap for the episode
    """
    os.makedirs(out_dir, exist_ok=True)
    T = len(state_ids)
    fig = plt.figure(figsize=(8, 4), dpi=160)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.array(state_ids)[None, :], aspect="auto", interpolation="nearest")
    ax1.set_yticks([])
    ax1.set_xlabel("Step")
    ax1.set_title(f"Episode {ep_idx}: CoDA state (raster)")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(trans_mat, aspect="auto", interpolation="nearest")
    ax2.set_xlabel("Next state")
    ax2.set_ylabel("State")
    ax2.set_title("Episode transition matrix (counts)")

    plt.tight_layout()
    fp = os.path.join(out_dir, f"latent_raster_ep{ep_idx}.png")
    plt.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    return fp

def plot_tsne_png(ep_idx, vectors, ids, out_dir):
    """Optional t-SNE of latent vectors, colored by current discrete id."""
    if not HAVE_TSNE or len(vectors) < 5:
        return None
    X = np.vstack(vectors)
    try:
        X2 = TSNE(n_components=2, perplexity=min(30, max(5, len(X)//5)), init="random", learning_rate="auto").fit_transform(X)
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(4, 4), dpi=160)
    sc = ax.scatter(X2[:, 0], X2[:, 1], c=np.asarray(ids), s=10)
    ax.set_title(f"Episode {ep_idx}: t-SNE of CoDA latent")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fp = os.path.join(out_dir, f"tsne_ep{ep_idx}.png")
    plt.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    return fp

def build_transition_counts(ids):
    """Simple episode-level transition matrix from a sequence of int ids."""
    ids = [int(x) for x in ids if x is not None]
    if not ids:
        return np.zeros((1, 1), dtype=int)
    S = max(ids) + 1
    M = np.zeros((S, S), dtype=int)
    for i in range(len(ids) - 1):
        M[ids[i], ids[i+1]] += 1
    return M

def animate_episode(ep_idx, frames, vectors, ids, out_dir, fps=6):
    """
    Save an animation with:
      left: environment RGB frame
      right: bar-chart of the CoDA latent vector (or a text for the ID if no vector)
    """
    if not frames:
        return None

    os.makedirs(out_dir, exist_ok=True)
    H, W = frames[0].shape[:2]

    fig = plt.figure(figsize=(8, 4), dpi=160)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_vec = fig.add_subplot(gs[0, 1])

    im = ax_img.imshow(frames[0])
    ax_img.set_axis_off()
    ax_img.set_title(f"Episode {ep_idx}: env")

    # init bar plot
    if vectors and vectors[0] is not None:
        v0 = vectors[0].ravel()
        bars = ax_vec.bar(np.arange(len(v0)), v0)
        ax_vec.set_ylim(0, max(1.0, float(v0.max()) + 1e-6))
        ax_vec.set_title("CoDA latent (vector)")
    else:
        bars = None
        ax_vec.set_title("CoDA state id")
        ax_vec.text(0.5, 0.5, f"id={ids[0] if ids else '?'}", ha="center", va="center")
        ax_vec.set_xticks([]); ax_vec.set_yticks([])

    plt.tight_layout()

    def update(t):
        im.set_data(frames[t])
        if bars is not None and vectors[t] is not None:
            v = vectors[t].ravel()
            ymax = max(1.0, float(v.max()) + 1e-6)
            ax_vec.set_ylim(0, ymax)
            # adjust number of bars if needed
            if len(bars) != len(v):
                ax_vec.clear()
                new_bars = ax_vec.bar(np.arange(len(v)), v)
                ax_vec.set_ylim(0, ymax)
                ax_vec.set_title("CoDA latent (vector)")
                return [im] + list(new_bars)
            else:
                for b, val in zip(bars, v):
                    b.set_height(val)
                return [im] + list(bars)
        else:
            ax_vec.clear()
            ax_vec.set_title("CoDA state id")
            ax_vec.text(0.5, 0.5, f"id={ids[t] if t < len(ids) else '?'}", ha="center", va="center")
            ax_vec.set_xticks([]); ax_vec.set_yticks([])
            return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/max(fps,1), blit=False)
    # Try mp4; if ffmpeg is missing, fall back to gif
    mp4_path = os.path.join(out_dir, f"episode_ep{ep_idx}.mp4")
    gif_path = os.path.join(out_dir, f"episode_ep{ep_idx}.gif")
    try:
        anim.save(mp4_path, writer="ffmpeg", dpi=160, fps=fps)
        plt.close(fig)
        return mp4_path
    except Exception:
        try:
            anim.save(gif_path, writer=PillowWriter(fps=fps))
            plt.close(fig)
            return gif_path
        except Exception:
            plt.close(fig)
            return None

# -------------------------
# Rollout + visualization
# -------------------------

def run(args):
    seed_everything(args.seed)
    env = make_env(args.env_id, use_coda=True, render=True)
    model = PPO.load(args.model)

    os.makedirs(args.out, exist_ok=True)
    all_paths = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        obs_for_policy = preprocess_obs(obs, use_coda=True)

        frames, ids, vecs = [], [], []
        done = False
        steps = 0

        # Try to render the starting frame
        try:
            frames.append(env.render())
        except Exception:
            pass

        while not done and steps < args.max_steps:
            action, _ = model.predict(obs_for_policy, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Grab CoDA latent from *next* obs (after transition)
            # You can switch to current obs if you prefer.
            coda_obj = next_obs["coda"] if isinstance(next_obs, dict) and "coda" in next_obs else None
            sid, vec = extract_coda_latent(coda_obj)
            ids.append(sid)
            vecs.append(vec)

            # Render a frame if available
            try:
                frames.append(env.render())
            except Exception:
                pass

            # Prepare input for next step
            obs_for_policy = preprocess_obs(next_obs, use_coda=True)
            done = bool(terminated or truncated)
            steps += 1

        # Episode-level PNGs
        trans = build_transition_counts(ids)
        raster_path = plot_episode_png(ep, ids, trans, args.out)
        all_paths.append(raster_path)

        tsne_path = plot_tsne_png(ep, [v for v in vecs if v is not None], ids, args.out)
        if tsne_path: all_paths.append(tsne_path)

        # Animation
        if args.animate:
            anim_path = animate_episode(ep, frames, vecs, ids, args.out, fps=args.fps)
            if anim_path: all_paths.append(anim_path)

    # Save paths for convenience
    with open(os.path.join(args.out, "artifacts.txt"), "w") as f:
        for p in all_paths:
            f.write(str(p) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-6x6-v0")
    ap.add_argument("--model", type=str, required=True, help="Path to CoDA PPO model.zip")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--out", type=str, default="runs/latent_viz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--animate", action="store_true", help="Save an animation per episode")
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
