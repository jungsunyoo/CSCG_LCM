import numpy as np
from collections import defaultdict
import random
from tqdm import trange
import copy
import networkx as nx
import matplotlib.pyplot as plt
import sys
import igraph
from matplotlib import cm, colors
random.seed(42)
import seaborn as sns
from typing import Optional, Union, Sequence
from spatial_environments import * #ContinuousTMaze, GridEnvRightDownNoCue, GridEnvRightDownNoSelf, GridEnvDivergingMultipleReward, GridEnvDivergingSingleReward
from scipy.stats import beta
from abc import ABC, abstractmethod

class Agent(ABC): 
    """ Abstract Base Class for agents """
    def __init__(
        self, 
        gamma: float = 0.9, 
        lambda_: float = 0.9, 
        seed: Optional[Union[int, np.random.Generator]] = None,
        ):
        
        self.gamma = gamma
        self.lambda_ = lambda_
        self.rng = (
            seed if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )
        self.clone_dict = dict()
        self.reverse_clone_dict = dict()
        self.cogmap: np.ndarray = np.empty(0)   # placeholder; fill later
        self.cue_memory = dict()
        
    @abstractmethod        
    def add_clone_dict(self, new_clone, successor):
        self.clone_dict[new_clone] = successor
        
    @abstractmethod    
    def add_reverse_clone_dict(self, new_clone, successor):
        self.reverse_clone_dict[successor] = new_clone

    @abstractmethod
    def add_cue_memory(self, cue, sensitivity):
        # self.cue_memory.append(cue)
        self.cue_memory[cue] = sensitivity

    @abstractmethod
    def plot_graph(self, T, niter, 
                #    reward_terminal = [16], env_size = (4,4),
                   highlight_node=-1, highlight_node_2 = -1,
                   threshold=0.3,
                   save=False,savename='img'):
        """
        Function that draws the current environment
        !!!BASE VERSION ONLY WORKS FOR GRID ENVIRONMENTS!!!
        """
        G = nx.DiGraph()
        n_state = np.shape(T)[0]
        n_action = np.shape(T)[1]
        # Add nodes
        for s in range(n_state):
            G.add_node(s)
            
        # Define colors/labels for each action
        action_colors = {0: "blue",   # Right
                        1: "green",  # Down
                        2: "orange", # Left
                        3: "purple"} # Up
        action_labels = {0: "R", 1: "D", 2: "L", 3: "U"}
        
        # Add edges
        edge_colors = []
        for s in range(n_state):
            for a in range(n_action):
                for s_next in range(n_state):
                    prob = T[s, a, s_next]
                    if prob > threshold:
                        # Create a directed edge from s to s_next
                        # lbl = f"{action_labels[a]}, p={prob:.1f}"
                        lbl = f"p={prob:.1f}"
                        G.add_edge(s, s_next, label=lbl)

                        # Choose an edge color based on the action
                        edge_colors.append(action_colors[a])
        # Build position dictionary for states
        pos = {}
        horizontal_scale = 3
        vertical_scale = 3
        offset = 0.2

        # For a 4x4 grid, you typically have states 0..15 internally.
        # If self.env_size = (4,4), that means 4 rows, 4 cols.
        # row = s // cols, col = s % cols  => row = s//4, col = s%4
        for s in range(n_state):
            row = s // self.env_size[1]  # self.env_size=(4,4)-> row = s // 4
            col = s % self.env_size[1]   # col = s % 4
            # Node indexing in your environment might be 1-based => map s->s+1
            # Plot them with negative row so the first row appears on top
            pos[s] = (col * horizontal_scale, -row * vertical_scale)

            # If s is an unrewarded terminal, offset it next to the associated rewarded terminal
            if s in self.unrewarded_terminals:
                idx = self.unrewarded_terminals.index(s)
                rew_terminal = self.rewarded_terminals[idx]
                pos[s] = (pos[rew_terminal][0] + offset, 
                        pos[rew_terminal][1] + offset)
            elif s > self.num_unique_states-1:
            # else:
                # If you have extra states or clones, you might offset them from a 'clone_dict' etc.
                # This is just your original logic. If you don't need it, remove it.
                pos[s] = (pos[self.clone_dict[s]][0] + offset, 
                        pos[self.clone_dict[s]][1] + offset)

        # Draw figure
        plt.figure(figsize=(12, 8))

        # Determine node colors
        colors = []
        for node in G.nodes():
            state = int(node)
            if state == highlight_node:
                colors.append("red")
            elif state == highlight_node_2:
                colors.append("yellow")
            # elif self.is_rewarded_terminal(state):
            #     colors.append("lightgreen")
            # elif self.is_unrewarded_terminal(state):
            #     colors.append("lightcoral")
            else:
                colors.append("lightblue")

        # Draw nodes, labels
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=5000, 
                                   edgecolors='black',      # <-- border colour
                                   linewidths=2.5           # <-- border thickness)
        )
        nx.draw_networkx_labels(G, pos, font_size=40, font_color='black')

        # Get edges in the order added to match edge_colors
        edges = list(G.edges())
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                            arrowstyle='->', arrowsize=70, width=8)
        nx.draw_networkx_edge_labels(G, pos,
                                    edge_labels=nx.get_edge_attributes(G, 'label'),
                                    font_size=15, label_pos=0.5)

        plt.axis('off')
        plt.title(f"Graph at episode {niter}", size=20)
        plt.tight_layout()

        # Save figure if desired
        # final_name = f"{savename}_episode_{niter}"
        final_name = f"{savename}"
        if save:
            plt.savefig(final_name)
        plt.show()



