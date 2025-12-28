
"""
coda_viz_simple.py
==================
Small visualization helpers for Snapshot objects from coda_simple.CoDASimple.

Dependencies: networkx, matplotlib
"""

from __future__ import annotations
from typing import Any, Dict, Hashable, List, Optional, Tuple
import math
import networkx as nx
import matplotlib.pyplot as plt


def grid_positions(rows: int, cols: int, obs_start: int = 0, origin: str = "top-left") -> Dict[Hashable, Tuple[float, float]]:
    pos: Dict[Hashable, Tuple[float, float]] = {}
    k = obs_start
    for r in range(rows):
        for c in range(cols):
            x = float(c)
            y = float(-r) if origin == "top-left" else float(r)
            pos[k] = (x, y)
            k += 1
    return pos


def _jitter(i: int, radius: float = 0.22) -> Tuple[float, float]:
    ang = (i * 2.399963229728653) % (2 * math.pi)
    return radius * math.cos(ang), radius * math.sin(ang)


def snapshot_to_nx(snapshot) -> nx.DiGraph:
    G = nx.DiGraph()
    for sid, nd in snapshot.nodes.items():
        if nd.get("active", True):
            G.add_node(sid, **nd)
    for (s, a), succ in snapshot.edges.items():
        if s not in G:
            continue
        for s_next, c in succ.items():
            if s_next not in G:
                continue
            G.add_edge(s, s_next, action=a, count=c)
    return G


def plot_snapshot(
    snapshot,
    *,
    ax=None,
    title: Optional[str] = None,
    layout: str = "grid",           # "grid" | "spring"
    obs_pos: Optional[Dict[Hashable, Tuple[float, float]]] = None,
    node_label: str = "sid",        # "sid" | "obs" | "sid_obs"
    salient_color: str = "red",
    base_color: str = "#b9dfee",
    figsize=(12, 8),
    show_edge_labels: bool = True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    G = snapshot_to_nx(snapshot)
    salient = set(getattr(snapshot, "salient", []) or [])

    if layout == "spring":
        pos = nx.spring_layout(G, seed=0)
    else:
        if obs_pos is None:
            obs_pos = {}
        by_obs: Dict[Hashable, List[int]] = {}
        for sid, data in G.nodes(data=True):
            by_obs.setdefault(data.get("obs"), []).append(sid)
        pos = {}
        for obs, sids in by_obs.items():
            base = obs_pos.get(obs, (float(obs) if isinstance(obs, (int, float)) else 0.0, 0.0))
            sids = sorted(sids)
            for j, sid in enumerate(sids):
                dx, dy = _jitter(j, radius=0.24 if j > 0 else 0.0)
                pos[sid] = (base[0] + dx, base[1] + dy)

    node_colors = [salient_color if sid in salient else base_color for sid in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=900, edgecolors="white", linewidths=1.0)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=18, width=2.6)

    labels = {}
    for sid, data in G.nodes(data=True):
        if node_label == "obs":
            labels[sid] = str(data.get("obs"))
        elif node_label == "sid_obs":
            labels[sid] = f"{sid}\n({data.get('obs')})"
        else:
            labels[sid] = str(sid)
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=11)

    if show_edge_labels:
        edge_labels = {}
        for (s, a), succ in snapshot.edges.items():
            tot = sum(succ.values())
            if tot <= 0:
                continue
            for s_next, c in succ.items():
                if (s in G) and (s_next in G):
                    edge_labels[(s, s_next)] = f"{a}, p={c/tot:.2f}"
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=8, rotate=False)

    ax.set_title(title or getattr(snapshot, "title", "Cognitive graph"), fontsize=18)
    ax.axis("off")
    return fig, ax


def save_snapshots(snapshots: List[Any], out_dir: str, prefix: str = "iter", **plot_kwargs):
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, snap in enumerate(snapshots):
        fig, _ = plot_snapshot(snap, title=snap.title, **plot_kwargs)
        fp = os.path.join(out_dir, f"{prefix}_{i:03d}.png")
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(fp)
    return paths
