#!/usr/bin/env python3
"""
CoDA world-model graph visualization on MiniGrid DoorKey.

For each episode, this script saves:
  • aliased_ep{E}.png  -> nodes are positions only (before markovization)
  • coda_ep{E}.png     -> nodes are CoDA states placed at mean (x,y), colored by phase
  • (optional) coda_ep{E}.gif -> step-by-step reveal of the CoDA graph

It also prints a summary of "split after key" positions:
  positions (x,y) where CoDA assigns different state IDs pre-key vs post-key.

Usage:
  python coda_graph_viz.py \
    --model runs/ppo_coda_MiniGrid-DoorKey-6x6-v0_s0/model.zip \
    --env MiniGrid-DoorKey-6x6-v0 \
    --episodes 5 --max-steps 200 \
    --out runs/coda_graphs --animate --edge-thresh 0.15 --topk 1
"""

import os, argparse, math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

# Your CoDA wrapper must expose obs["coda"]
try:
    from coda_rl.coda_wrapper import CoDAWrapper
except Exception as e:
    raise SystemExit("Could not import coda_rl.coda_wrapper.CoDAWrapper. "
                     "Ensure it is installed and 'expose_in_obs=True' is used.") from e

# ------------------------ helpers: env + obs ------------------------

def to_chw(img):
    if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] in (1,3):
        return np.transpose(img, (2,0,1))
    return img

def make_env(env_id: str):
    env = __import__("gymnasium").make(env_id)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = CoDAWrapper(env, expose_in_obs=True)
    return env

_VEC_KEYS = ["z","latent","embedding","features","phi","rep","repr","vector","onehot","logits"]
_ID_KEYS  = ["state_id","clone_id","id","state","s","node","cluster"]

def unpack_coda(coda: Any) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """Return (vector, state_id). Falls back to argmax(vec) for id."""
    vec, sid = None, None
    if isinstance(coda, dict):
        for k in _VEC_KEYS:
            if k in coda:
                v = np.asarray(coda[k]).reshape(-1).astype(np.float32)
                if v.size: vec = v; break
        for k in _ID_KEYS:
            if k in coda:
                try: sid = int(np.asarray(coda[k]).reshape(-1)[0])
                except Exception: pass
                break
    else:
        arr = np.asarray(coda)
        if arr.ndim==0 or (arr.ndim==1 and arr.size==1):
            try: sid = int(arr.reshape(-1)[0])
            except Exception: sid = None
        else:
            vec = arr.reshape(-1).astype(np.float32)
    if sid is None and vec is not None and vec.size>0:
        sid = int(np.argmax(vec))
    return vec, sid

def read_labels(env) -> Dict[str, Any]:
    """MiniGrid readouts for phase/coloring."""
    e = env.unwrapped
    ax, ay = (getattr(e,"agent_pos",(np.nan,np.nan)) or (np.nan,np.nan))
    has_key = getattr(e, "carrying", None) is not None
    # any door open?
    door_open = False
    try:
        from minigrid.core.world_object import Door
        grid = e.grid
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x,y)
                if isinstance(obj, Door) and getattr(obj,"is_open",False):
                    door_open = True; raise StopIteration
    except StopIteration:
        pass
    except Exception:
        pass
    return dict(x=int(ax) if ax==ax else -1,
                y=int(ay) if ay==ay else -1,
                has_key=int(has_key),
                door_open=int(door_open))

# ------------------------ collect a single episode ------------------------

def run_episode(env, model, max_steps=200):
    obs, _ = env.reset()
    # obs is dict {"obs": HWC image, "coda": ...}
    data = dict(pos=[], phase=[], coda_id=[], coda_vec=[])
    done = False; steps=0

    while not done and steps < max_steps:
        lab = read_labels(env)
        data["pos"].append( (lab["x"], lab["y"]) )
        data["phase"].append( lab["has_key"] )  # 0 pre-key, 1 post-key

        vec, sid = unpack_coda(obs["coda"])
        data["coda_vec"].append(vec)
        data["coda_id"].append(sid if sid is not None else -1)

        # predict & step
        action, _ = model.predict({"obs": to_chw(obs["obs"]), "coda": obs["coda"]}, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
    return data

# ------------------------ build graphs ------------------------

def counts_from_seq(nodes: List[Any], edges: List[Tuple[Any,Any]]) -> Dict[Any, Dict[Any,int]]:
    C: Dict[Any, Dict[Any,int]] = {}
    for u,v in edges:
        C.setdefault(u,{}); C[u][v] = C[u].get(v,0)+1
    # ensure all nodes present
    for n in nodes: C.setdefault(n,{})
    return C

def normalize_counts(C: Dict[Any, Dict[Any,int]]) -> Dict[Any, Dict[Any,float]]:
    P: Dict[Any, Dict[Any,float]] = {}
    for u, row in C.items():
        s = float(sum(row.values()))
        if s<=0:
            P[u]={}
        else:
            P[u]={v:c/s for v,c in row.items()}
    return P

def out_entropy(row: Dict[Any,float]) -> float:
    if not row: return 0.0
    p = np.array(list(row.values()), dtype=float)
    p = p[p>0]
    return float(-(p*np.log(p)).sum())

# ------------------------ draw helpers (matplotlib only) ------------------------

def jitter(xy, by=0.08, seed=0):
    rng = np.random.RandomState(seed)
    return (xy[0]+(rng.rand()-0.5)*by, xy[1]+(rng.rand()-0.5)*by)

def draw_graph_png(title: str,
                   nodes_xy: Dict[Any, Tuple[float,float]],
                   P: Dict[Any, Dict[Any,float]],
                   node_color: Dict[Any, Tuple[float,float,float]],
                   node_label: Dict[Any,str],
                   out_path: Path,
                   edge_thresh: float = 0.15,
                   topk: int = 1):
    plt.figure(figsize=(6.4,6.0), dpi=160)
    ax = plt.gca()

    # Edges
    for u, row in P.items():
        if not row: continue
        # take top-k by prob (for legibility)
        items = sorted(row.items(), key=lambda kv: kv[1], reverse=True)
        if topk is not None and topk>0: items = items[:topk]
        for v, p in items:
            if p < edge_thresh: continue
            (x1,y1),(x2,y2) = nodes_xy[u], nodes_xy[v]
            lw = 1.0 + 4.0*float(p)  # thickness by prob
            ax.annotate("",
                        xy=(x2,y2), xytext=(x1,y1),
                        arrowprops=dict(arrowstyle="-|>", lw=lw, color="0.25", alpha=0.9))
            # place small 'p=' label mid-edge
            mx,my = (x1+x2)/2,(y1+y2)/2
            ax.text(mx,my, f"p={p:.2f}", fontsize=8, ha="center", va="center", color="0.25")

    # Nodes
    for n,(x,y) in nodes_xy.items():
        c = node_color.get(n,(0.7,0.8,1.0))
        ax.scatter([x],[y], s=260, c=[c], edgecolors="k", linewidths=1.0, zorder=3)
        ax.text(x,y, node_label.get(n,str(n)), ha="center", va="center", fontsize=10, zorder=4)

    # Aesthetics
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close()

# ------------------------ per-episode pipeline ------------------------

def aliased_graph_episode(ep: Dict[str,Any]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    BEFORE markovization: nodes = positions only (x,y).
    """
    pos = ep["pos"]
    nodes = sorted(set(pos))
    edges = list(zip(pos[:-1], pos[1:]))
    C = counts_from_seq(nodes, edges)
    P = normalize_counts(C)

    # draw positions on the grid
    nodes_xy = {xy: (xy[0], xy[1]) for xy in nodes}
    # uniform color, label with (x,y) and entropy
    node_color = {xy: (0.80,0.90,1.0) for xy in nodes}
    node_label = {}
    for n in nodes:
        H = out_entropy(P[n])
        node_label[n] = f"{n}\nH={H:.2f}"
    return nodes_xy, P, node_color, node_label

def coda_graph_episode(ep: Dict[str,Any]) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    AFTER markovization: nodes = CoDA state ids, placed at mean (x,y),
    colored by phase (pre-key vs post-key).
    """
    ids = list(map(int, ep["coda_id"]))
    pos = ep["pos"]
    phase = ep["phase"]

    nodes = sorted(set(ids))
    # mean (x,y) per id
    xy = {n:[] for n in nodes}
    ph = {n:[] for n in nodes}
    for sid,(x,y),phv in zip(ids, pos, phase):
        xy[sid].append( (x,y) )
        ph[sid].append( phv )
    nodes_xy = {}
    node_color = {}
    node_label = {}
    for n in nodes:
        if len(xy[n])==0:
            nodes_xy[n]=(0,0)
        else:
            xs,ys = zip(*xy[n])
            mx,my = float(np.mean(xs)), float(np.mean(ys))
            nodes_xy[n]=(mx,my)
        # color by majority phase: pre-key (blue-ish) vs post-key (orange-ish)
        if len(ph[n])>0 and np.mean(ph[n])>=0.5:
            node_color[n]=(1.0,0.78,0.55)  # post-key
            phase_txt="post"
        else:
            node_color[n]=(0.70,0.85,1.0)  # pre-key
            phase_txt="pre"
        node_label[n]=f"{n}\n{phase_txt}"

    # transitions over CoDA states
    edges = list(zip(ids[:-1], ids[1:]))
    C = counts_from_seq(nodes, edges)
    P = normalize_counts(C)

    # annotate entropy on the label
    for n in nodes:
        H = out_entropy(P[n])
        node_label[n] += f"\nH={H:.2f}"
    return nodes_xy, P, node_color, node_label

def detect_splits_after_key(ep: Dict[str,Any]) -> List[Tuple[int,int]]:
    """
    For each position (x,y), check if we saw distinct CoDA ids pre-key vs post-key.
    Returns list of positions where splitting occurred.
    """
    pos = ep["pos"]; ids = ep["coda_id"]; phase = ep["phase"]
    bucket = {}
    for (x,y),sid,ph in zip(pos, ids, phase):
        bucket.setdefault((x,y), {0:set(),1:set()})
        bucket[(x,y)][int(ph)].add(int(sid) if sid is not None else -1)
    splits=[]
    for xy, d in bucket.items():
        if d[0] and d[1]:
            # if the sets of ids used pre vs post are disjoint → split
            if d[0].isdisjoint(d[1]):
                splits.append(xy)
    return sorted(splits)

# ------------------------ animation (optional) ------------------------

def animate_coda_episode(ep_idx: int,
                         ep: Dict[str,Any],
                         out_dir: Path,
                         edge_thresh=0.15,
                         topk=1):
    """Write a GIF revealing the CoDA graph step-by-step (no ffmpeg needed)."""
    try:
        from PIL import Image
    except Exception:
        return  # skip if Pillow not installed

    ids = list(map(int, ep["coda_id"]))
    pos = ep["pos"]; phase=ep["phase"]

    # fixed node positions/colors/labels for the whole episode
    nodes_xy, P_full, node_color, node_label = coda_graph_episode(ep)

    frames=[]
    for t in range(1, len(ids)):
        # partial transitions up to t
        edges = list(zip(ids[:t], ids[1:t+1]))
        C = counts_from_seq(sorted(set(ids[:t+1])), edges)
        P = normalize_counts(C)
        fp = out_dir / f"_tmp_ep{ep_idx:03d}_{t:03d}.png"
        draw_graph_png(f"CoDA graph (ep {ep_idx}, t={t})",
                       nodes_xy, P, node_color, node_label,
                       fp, edge_thresh=edge_thresh, topk=topk)
        frames.append(fp)

    imgs=[Image.open(p) for p in frames]
    if imgs:
        gif_path = out_dir / f"coda_ep{ep_idx:03d}.gif"
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=120, loop=0)
    # cleanup
    for p in frames:
        try: p.unlink()
        except Exception: pass

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to CoDA PPO model.zip")
    ap.add_argument("--env", default="MiniGrid-DoorKey-6x6-v0")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out", default="runs/coda_graphs")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--edge-thresh", type=float, default=0.15, help="hide edges with p < thresh")
    ap.add_argument("--topk", type=int, default=1, help="draw top-k outgoing edges per node")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args.env); model = PPO.load(args.model)

    for ep_idx in range(args.episodes):
        ep = run_episode(env, model, max_steps=args.max_steps)

        # BEFORE (aliased)
        nodes_xy, P, node_color, node_label = aliased_graph_episode(ep)
        draw_graph_png("Aliased (before markovization): nodes = positions",
                       nodes_xy, P, node_color, node_label,
                       out_dir / f"aliased_ep{ep_idx:03d}.png",
                       edge_thresh=args.edge_thresh, topk=args.topk)

        # AFTER (CoDA states)
        nodes_xy, P, node_color, node_label = coda_graph_episode(ep)
        # jitter clones that sit on the same average (x,y)
        used = {}
        for n,xy in list(nodes_xy.items()):
            key=(round(xy[0],2), round(xy[1],2))
            kcount = used.get(key,0)
            if kcount>0:
                nodes_xy[n] = (xy[0]+0.08*kcount, xy[1]+0.08*kcount)
            used[key]=kcount+1

        draw_graph_png("CoDA (after markovization): nodes = CoDA states (color=phase)",
                       nodes_xy, P, node_color, node_label,
                       out_dir / f"coda_ep{ep_idx:03d}.png",
                       edge_thresh=args.edge_thresh, topk=args.topk)

        # Splitting report
        splits = detect_splits_after_key(ep)
        if splits:
            print(f"[ep {ep_idx}] split after key at positions: {splits}")
        else:
            print(f"[ep {ep_idx}] no clear split detected (by has_key)")

        if args.animate:
            animate_coda_episode(ep_idx, ep, out_dir, edge_thresh=args.edge_thresh, topk=args.topk)

    print(f"Saved graphs to: {out_dir}")

if __name__ == "__main__":
    main()
