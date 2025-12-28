# coda_viz_zipper.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import math
import io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def fig_to_rgb(fig):

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())  # HxWx4
    return rgba[:, :, :3].copy()

def grid_positions(size: int) -> Dict[int, Tuple[float, float]]:
    # row-major; y inverted so row 0 is top
    pos = {}
    for r in range(size):
        for c in range(size):
            idx = r * size + c
            pos[idx] = (c, -r)
    return pos

def _offset_for_clone(k: int) -> Tuple[float, float]:
    # small diagonal offset to show overlap (like your animation)
    return (0.07 * k, -0.07 * k)

def build_nx_graph(agent) -> nx.DiGraph:
    G = nx.DiGraph()
    # nodes are latent ids; store obs
    for s, obs in agent.latent_to_obs.items():
        G.add_node(s, obs=obs)

    # edges: for each latent/action, add edge to each successor latent with prob
    for s in agent.latent_to_obs.keys():
        for a in [0, 1]:
            probs = agent.trans_probs(s, a)
            for sp, p in probs.items():
                G.add_edge(s, sp, action=a, p=p)
    return G

def plot_cognitive_graph(
    agent,
    grid_size: int = 4,
    title: Optional[str] = None,
    highlight_cue: Optional[int] = None,
    highlight_new: Optional[int] = None,
    ax=None,
):
    """
    Draw a grid-layout cognitive graph:
    - blue edges: Right (action=0)
    - green edges: Down  (action=1)
    - label edges with p
    - cue node (red), newly created clone (yellow)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure

    ax.clear()
    ax.set_axis_off()

    base_pos = grid_positions(grid_size)

    # build positions for latent nodes: same obs overlap with offsets
    obs_to_latents = agent.obs_to_latents
    pos = {}
    for obs, latents in obs_to_latents.items():
        for k, s in enumerate(latents):
            dx, dy = _offset_for_clone(k) if k > 0 else (0.0, 0.0)
            x0, y0 = base_pos.get(obs, (0.0, 0.0))
            pos[s] = (x0 + dx, y0 + dy)

    G = build_nx_graph(agent)

    # node colors
    node_colors = []
    for s in G.nodes():
        if highlight_new is not None and s == highlight_new:
            node_colors.append("#ffeb3b")  # yellow
        elif highlight_cue is not None and s == highlight_cue:
            node_colors.append("#e53935")  # red
        else:
            node_colors.append("#bfe3f2")  # pale blue

    # node border and labels
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=2000,
        node_color=node_colors,
        edgecolors="black",
        linewidths=2,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=28, font_color="black")

    # draw edges by action with separate styles
    def draw_edges_for_action(a: int, color: str):
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("action") == a]
        widths = []
        for u, v in edges:
            p = G.edges[u, v]["p"]
            widths.append(2.0 + 4.0 * p)  # thicker if high p (visual cue)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges,
            edge_color=color,
            arrows=True,
            arrowsize=25,
            width=widths,
            connectionstyle="arc3,rad=0.0",
        )

        # edge labels "p=..."
        labels = {}
        for u, v in edges:
            p = G.edges[u, v]["p"]
            labels[(u, v)] = f"p={p:.1f}"
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax,
            edge_labels=labels,
            font_size=14,
            label_pos=0.55,
            rotate=False,
            bbox=dict(alpha=0.0, edgecolor="none"),
        )

    draw_edges_for_action(0, "blue")
    draw_edges_for_action(1, "green")

    if title:
        ax.set_title(title, fontsize=22)

    fig.tight_layout()
    return fig, ax

def render_frames_to_gif(frames: List[Any], gif_path: str, fps: int = 2):
    """
    frames: list of PIL images or numpy arrays.
    Requires imageio.
    """
    import imageio
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(gif_path, frames, duration=duration)

def simulate_and_collect_frames(
    agent,
    env,
    policies: List,
    n_episodes: int = 30,
    grid_size: int = 4,
    title_prefix: str = "",
):
    """
    Runs episodes, collecting a matplotlib-rendered image after each clone creation.
    Returns list of RGB frames (numpy arrays).
    """
    import numpy as np

    frames_rgb = []
    events: List[Dict[str, Any]] = []

    # run
    for ep in range(n_episodes):
        policy = policies[ep % len(policies)]
        agent.run_episode(env, policy_fn=policy, frames=events, terminal_obs=env.terminal_obs)

        # after each event appended in this episode, render it
        while events:
            ev = events.pop(0)
            cue = ev.get("cue", None)
            new = ev.get("new", None)
            title = ev.get("title", None) or f"{title_prefix}ep={ep}"

            # fig, ax = plt.subplots(figsize=(12, 8))
            # plot_cognitive_graph(
            #     agent,
            #     grid_size=grid_size,
            #     title=title,
            #     highlight_cue=cue,
            #     highlight_new=new,
            #     ax=ax,
            # )
            # fig.canvas.draw()
            # w, h = fig.canvas.get_width_height()
            # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            # frames_rgb.append(img)
            # plt.close(fig)

    # final frame
    # import numpy as np
    # fig, ax = plt.subplots(figsize=(12, 8))
    # plot_cognitive_graph(agent, grid_size=grid_size, title=f"{title_prefix}final", ax=ax)
    # fig.canvas.draw()
    # w, h = fig.canvas.get_width_height()
    # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    # frames_rgb.append(img)

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_cognitive_graph(
        agent,
        grid_size=grid_size,
        title=title,
        highlight_cue=cue,
        highlight_new=new,
        ax=ax,
    )
    img = fig_to_rgb(fig)
    frames_rgb.append(img)
    plt.close(fig)

    # plt.close(fig)

    return frames_rgb
