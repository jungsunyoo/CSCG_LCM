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

class Environment:
    """Abstract Base Class for all environment classes"""
    def __init__(self, rewarded_termimals:list, unrewarded_termimals:list, start:int, cues:list):  # could potentially make this a list of cues - order of cues matters
        self.rewarded_terminals = rewarded_termimals
        self.unrewarded_terminals = unrewarded_termimals

        self.start_state = start
        self.cue_states = cues

        self.clone_dict = dict()
        self.reverse_clone_dict = dict()

    def reset(self):
        """
        Reset environment to the start:
          - current_state=1
          - visited_cue=False
        Returns current_state (1).
        """
        pass

    def get_valid_actions(self, state=None):
        """
        Return the list of valid actions for the current state
        (or a given state).
        """
        pass

    def step(self, action):
        """
        Step with a guaranteed valid action. If an invalid action is given,
        we can either ignore or raise an Exception. We'll raise an error.
        """
        pass

    def plot_graph(self, T, niter, 
                #    reward_terminal = [16], env_size = (4,4),
                   highlight_node=-1, highlight_node_2 = -1,
                   save=True,savename='img'):
        """
        Function that draws the current environment
        !!!BASE VERSION ONLY WORKS FOR GRID ENVIRONMENTS!!!
        """
        G = nx.DiGraph()
        n_state = np.shape(T)[0]
        n_action = np.shape(T)[1]
        # Add nodes
        for s in range(n_state-1):
            G.add_node(s+1)

        # Add edges with color coding for each action (optional)
        edge_colors = []
        for s in range(n_state):
            for a in range(n_action):  # actions 0 or 1
                for s_next in range(n_state):
                    prob = T[s, a, s_next]
                    if prob > 0:
                        # Add a directed edge from s to s_next
                        # Label with a=0 or a=1, or probability, or both
                        G.add_edge(s, s_next, label=f"A{a}, p={prob:.1f}")
                        
                        # (Optional) color edges differently for each action
                        if a == 0:  
                            edge_colors.append("blue")   # right edges in blue
                        else:
                            edge_colors.append("green")  # down edges in green

        # Build position dictionary
        pos = {}
        horizontal_scale = 3
        vertical_scale = 3
        offset = 0.2
        for s in range(n_state):
            row = s // self.env_size[0] #4
            col = s % self.env_size[1] #4
            pos[s+1] = (col * horizontal_scale, -row * vertical_scale)
            if s in self.unrewarded_terminals: 
                # if s==17: 
                # pos[s] = (pos[16][0]+offset, pos[16][1] + offset)
                idx = self.unrewarded_terminals.index(s)
                pos[s] = (pos[self.rewarded_terminals[idx]][0]+offset, 
                          pos[self.rewarded_terminals[idx]][1] + offset)
            elif s > self.num_unique_states:
            # elif s > 17: 
                pos[s] = (pos[self.clone_dict[s]][0]+offset, pos[self.clone_dict[s]][1]+offset)

        plt.figure(figsize=(12,8))
        # Create a color list; default 'lightblue', but 'red' for special_node
        colors = []
        for node in G.nodes():
            state = int(node)
            if state == highlight_node:
                colors.append("red")
            elif state == highlight_node_2:
                colors.append("yellow")
            elif self.is_rewarded_terminal(state):
                colors.append("lightgreen")
            elif self.is_unrewarded_terminal(state):
                colors.append("lightcoral")
            else:
                colors.append("lightblue")
        
        # nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1200)
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1200)
        nx.draw_networkx_labels(G, pos, font_size=20, font_color='black')

        # We need to extract edges in the same order they were added 
        # so that edge_colors aligns properly
        edges = list(G.edges())
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                            arrowstyle='->', arrowsize=20, width=3)
        nx.draw_networkx_edge_labels(G, pos, 
                                    edge_labels=nx.get_edge_attributes(G, 'label'),
                                    font_size=10, label_pos=0.5)

        plt.axis('off')
        plt.title("Graph at iteration {}".format(niter), size=20)
        plt.tight_layout()
    
        savename = savename+"_iteration_" + str(niter)
        if save: 
            plt.savefig(savename)
        plt.show()

    
    def is_terminal(self, state:int) -> bool:
        return state in self.rewarded_terminals or state in self.unrewarded_terminals
    
    def is_rewarded_terminal(self, state:int) -> bool:
        return state in self.rewarded_terminals

    def is_unrewarded_terminal(self, state:int) -> bool:
        return state in self.unrewarded_terminals
    
class GridEnv(Environment):
    """
    A non-Markovian grid environment of size env_size (rows x cols), with states numbered:

      1   2   3   ...   cols
      cols+1  cols+2  ... 2*cols
      ...
      up to rows*cols

    - The agent can move RIGHT (0), DOWN (1), LEFT (2), or UP (3).
    - No self-transitions at borders:
        If an action would go out of bounds, that action is not allowed from that state.
    - "Cue" logic: must visit any state in cue_states to get +1 reward upon reaching
      rewarded_terminal. If it hasn't visited a cue, it transitions instead to
      an unrewarded terminal state (which yields -1).
    """

    def __init__(self, cue_states=[2], env_size=(4,4), rewarded_terminal=[16]):
        """
        cue_states       : list of states that serve as 'cues' (must be visited 
                           to get +1 at terminal)
        env_size         : (rows, cols)
        rewarded_terminal: list of states that yield +1 if cue was visited, else -1
        """
        self.env_size = env_size
        
        # Build a mapping of (row, col) -> state (1-indexed)
        self.pos_to_state = {}
        state_counter = 0
        for r in range(self.env_size[0]):
            for c in range(self.env_size[1]):
                state_counter += 1
                self.pos_to_state[(r,c)] = state_counter
        
        self.state_to_pos = {s: rc for rc, s in self.pos_to_state.items()}

        # In your original code, for each rewarded terminal, you define a 
        # corresponding 'unrewarded_terminal' by shifting the index. 
        # For example: if 16 is a rewarded terminal, 17 would be the unrewarded one, etc.
        # We'll keep that logic, but adapt to however many rewarded terminals you have.
        unrewarded_terminals = [s + state_counter for s in range(len(rewarded_terminal))]

        # Start state is always 1 in this example, but you can change as needed.
        start_state = 1

        # Call super constructor (assuming 'Environment' base class)
        super().__init__(rewarded_termimals=rewarded_terminal, 
                         unrewarded_termimals=unrewarded_terminals,
                         start=start_state,
                         cues=cue_states)
        
        # Actions:
        #  0 -> right  (0, +1)
        #  1 -> down   (+1, 0)
        #  2 -> left   (0, -1)
        #  3 -> up     (-1, 0)
        self.base_actions = {
            0: (0, +1),   # right
            1: (+1, 0),   # down
            2: (0, -1),   # left
            3: (-1, 0)    # up
        }

        # Precompute valid actions per state
        self.valid_actions = self._build_valid_actions()

        # Total states = actual grid states + # of unrewarded terminal states
        self.num_unique_states = state_counter + len(self.unrewarded_terminals)

        # Reset environment
        self.reset()

    def _build_valid_actions(self):
        """
        Precompute valid actions for each non-terminal state.
        Valid = leads to a different state (i.e., in-bounds).
        """
        valid_dict = {}
        rows, cols = self.env_size

        for r in range(rows):
            for c in range(cols):
                s = self.pos_to_state[(r, c)]
                valid_actions_s = []
                for a, (dr, dc) in self.base_actions.items():
                    nr, nc = r + dr, c + dc
                    # Check in-bounds
                    if 0 <= nr < rows and 0 <= nc < cols:
                        next_s = self.pos_to_state[(nr, nc)]
                        # Only include if next_s != s (avoid self-transition)
                        if next_s != s:
                            valid_actions_s.append(a)
                valid_dict[s] = valid_actions_s

        # For terminal states, no valid actions
        for terminal in self.rewarded_terminals + self.unrewarded_terminals:
            valid_dict[terminal] = []

        return valid_dict

    def reset(self):
        """
        Reset environment to start state and mark cue as not visited.
        Returns current_state.
        """
        self.current_state = self.start_state
        self.visited_cue = False
        return self.current_state

    def get_valid_actions(self, state=None):
        """
        Return the list of valid actions from the (current) state.
        """
        if state is None:
            state = self.current_state
        return self.valid_actions[state]

    def step(self, action):
        """
        Step in environment with a guaranteed-valid action. 
        Returns (next_state, reward, done).
        """
        if action not in self.get_valid_actions():
            raise ValueError(f"Action {action} is not valid from state {self.current_state}.")

        # If already in terminal, no change
        if self.current_state in (self.rewarded_terminals + self.unrewarded_terminals):
            return self.current_state, 0, True

        # Compute next state
        r, c = self.state_to_pos[self.current_state]
        dr, dc = self.base_actions[action]
        nr, nc = r + dr, c + dc
        next_state = self.pos_to_state[(nr, nc)]

        # Check if next_state is a cue
        if next_state in self.cue_states:
            self.visited_cue = True

        reward = 0
        done = False

        # If next_state is a rewarded terminal
        if next_state in self.rewarded_terminals:
            done = True
            if self.visited_cue:
                reward = +1
            else:
                reward = -1
                # Move to the corresponding "unrewarded" terminal label
                idx = self.rewarded_terminals.index(next_state)
                next_state = self.unrewarded_terminals[idx]

        # Update current state
        self.current_state = next_state
        return next_state, reward, done
    
        # Create the graph
    def plot_graph(self, T, niter,
                highlight_node=-1, highlight_node_2=-1,
                save=True, savename='img'):
        """
        Plots the state transition graph using NetworkX.
        - T: transition probability matrix, shape [n_state, n_action, n_state]
        - niter: current iteration index (for the figure title)
        - highlight_node: optional state to highlight (e.g., in red)
        - highlight_node_2: optional second state to highlight (e.g., in yellow)
        - save: whether to save the figure as an image
        - savename: prefix for the saved image file
        """

        # Create directed graph
        G = nx.DiGraph()
        n_state = T.shape[0]
        n_action = T.shape[1]

        # Add nodes (we assume T includes all states 0..n_state-1 internally)
        for s in range(n_state-1):
            G.add_node(s+1)  # your code indexes nodes as 1..n_state

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
                    if prob > 0:
                        # Create a directed edge from s to s_next
                        lbl = f"{action_labels[a]}, p={prob:.1f}"
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
            pos[s+1] = (col * horizontal_scale, -row * vertical_scale)

            # If s is an unrewarded terminal, offset it next to the associated rewarded terminal
            if s in self.unrewarded_terminals:
                idx = self.unrewarded_terminals.index(s)
                rew_terminal = self.rewarded_terminals[idx]
                pos[s] = (pos[rew_terminal][0] + offset, 
                        pos[rew_terminal][1] + offset)
            elif s > self.num_unique_states:
                # If you have extra states or clones, you might offset them from a 'clone_dict' etc.
                # This is just your original logic. If you don't need it, remove it.
                pos[s] = (pos[clone_dict[s]][0] + offset, 
                        pos[clone_dict[s]][1] + offset)

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
        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1200)
        nx.draw_networkx_labels(G, pos, font_size=20, font_color='black')

        # Get edges in the order added to match edge_colors
        edges = list(G.edges())
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors,
                            arrowstyle='->', arrowsize=20, width=3)
        nx.draw_networkx_edge_labels(G, pos,
                                    edge_labels=nx.get_edge_attributes(G, 'label'),
                                    font_size=10, label_pos=0.5)

        plt.axis('off')
        plt.title(f"Graph at iteration {niter}", size=20)
        plt.tight_layout()

        # Save figure if desired
        final_name = f"{savename}_iteration_{niter}"
        if save:
            plt.savefig(final_name)
        plt.show()

class GridEnvRightDownNoSelf(Environment):
    """
    A 3x3 grid with states numbered 1..9:

        1   2   3
        4   5   6
        7   8   9

    - The agent can only move RIGHT (0) or DOWN (1).
    - No self-transitions at borders:
        If an action would go out of bounds, that action is not allowed
        from that state.
    - Same cue logic: must visit 5 for a +1 reward at 9, else goes to 10 with -1.
    """

    def __init__(self, cue_states=[2], env_size = (4,4), rewarded_terminal=[16]):
        self.env_size = env_size
        self.pos_to_state = {}        
        state = 0
        for i in range(self.env_size[0]):
            for j in range(self.env_size[1]):
                state += 1
                self.pos_to_state[(i,j)] = state
                
        
        self.state_to_pos = {s: rc for rc, s in self.pos_to_state.items()}

        # self.unrewarded_terminals = self.rewarded_terminals+1  # not in the grid, just a label
        unrewarded_terminals = [s+state+1 for s in range(len(rewarded_terminal))]
        start_state = 1

        super().__init__(rewarded_termimals=rewarded_terminal, 
                         unrewarded_termimals=unrewarded_terminals,
                         start = start_state,
                         cues=cue_states)
        
        # Actions as integers: 0=right, 1=down
        # right => (0, +1)
        # down  => (+1, 0)
        self.base_actions = {
            0: (0, +1),   # right
            1: (+1, 0)    # down
        }
        
        # Instead of having a static [0,1] action space, we have a
        # per-state action set (no invalid moves).
        self.valid_actions = self._build_valid_actions()

        self.num_unique_states = state + len(self.unrewarded_terminals) # hard coded for now # done!
        self.reset()

    def _build_valid_actions(self):
        """
        Precompute valid actions for each non-terminal grid state.
        A 'valid' action is one that leads to a NEW in-bounds state.
        """
        valid_dict = {}
        for row in range(self.env_size[0]):
            for col in range(self.env_size[1]):
                s = self.pos_to_state[(row, col)]
                # We'll store all actions that yield a different state (no self-transitions)
                valid_dict[s] = []
                for a, (dr, dc) in self.base_actions.items():
                    new_r = row + dr
                    new_c = col + dc
                    if 0 <= new_r < self.env_size[0] and 0 <= new_c < self.env_size[1]:
                        next_s = self.pos_to_state[(new_r, new_c)]
                        # Only count it if next_s != s (which can't happen in 3x3, but let's be explicit)
                        if next_s != s:
                            valid_dict[s].append(a)
        
        # For terminal states, no actions are valid
        for terminals in self.rewarded_terminals:
            valid_dict[terminals] = []
        for terminals in self.unrewarded_terminals:
            valid_dict[terminals] = []

        return valid_dict

    def reset(self):
        """
        Reset environment to the start:
          - current_state=1
          - visited_cue=False
        Returns current_state (1).
        """
        self.current_state = self.start_state
        self.visited_cue = False
        return self.current_state

    def get_valid_actions(self, state=None):
        """
        Return the list of valid actions for the current state
        (or a given state).
        """
        if state is None:
            state = self.current_state
        return self.valid_actions[state]

    def step(self, action):
        """
        Step with a guaranteed valid action. If an invalid action is given,
        we can either ignore or raise an Exception. We'll raise an error.
        """
        if action not in self.get_valid_actions():
            raise ValueError(f"Action {action} is not valid from state {self.current_state}.")

        # If we're already in a terminal (9 or 10), episode is over.
        if self.current_state in self.rewarded_terminals + self.unrewarded_terminals:
            return self.current_state, 0, True

        # Move
        row, col = self.state_to_pos[self.current_state]
        dr, dc = self.base_actions[action]
        next_row = row + dr
        next_col = col + dc

        next_state = self.pos_to_state[(next_row, next_col)]

        # Check cue
        if next_state in self.cue_states:
            self.visited_cue = True

        reward = 0
        done = False

        if self.is_terminal(next_state):    
            done = True
            if self.visited_cue:
                reward = +1
            else:
                reward = -1
                idx = self.rewarded_terminals.index(next_state)
                next_state = self.unrewarded_terminals[idx]

        # Update current state
        self.current_state = next_state
        return next_state, reward, done


# TODO: Update this
class GridEnvRightDownNoCue(Environment):
    """
    A 3x3 grid with states numbered 1..9:

        1   2   3
        4   5   6
        7   8   9

    - The agent can only move RIGHT (0) or DOWN (1).
    - No self-transitions at borders:
        If an action would go out of bounds, that action is not allowed
        from that state.
    - Same cue logic: must visit 5 for a +1 reward at 9, else goes to 10 with -1.
    """

    def __init__(self):
        # Grid layout: (row, col) -> state
        self.pos_to_state = {
            (0,0): 1, (0,1): 2, (0,2): 3, (0,3): 4,
            (1,0): 5, (1,1): 6, (1,2): 7, (1,3): 8,
            (2,0): 9, (2,1): 10, (2,2): 11, (2,3): 12,
            (3,0): 13, (3,1): 14, (3,2): 15, (3,3): 16,
        }
        self.state_to_pos = {s: rc for rc, s in self.pos_to_state.items()}
        
        # Actions as integers: 0=right, 1=down
        # right => (0, +1)
        # down  => (+1, 0)
        self.base_actions = {
            0: (0, +1),   # right
            1: (+1, 0)    # down
        }
        
        # Instead of having a static [0,1] action space, we have a
        # per-state action set (no invalid moves).
        self.valid_actions = self._build_valid_actions()

        # Special states
        self.start_state = 1
        # self.cue_state = cue_state
        self.rewarded_terminal = 16
        self.unrewarded_terminal = 17  # not in the grid, just a label

        self.reset()

    def _build_valid_actions(self):
        """
        Precompute valid actions for each non-terminal grid state.
        A 'valid' action is one that leads to a NEW in-bounds state.
        """
        valid_dict = {}
        for row in range(4):
            for col in range(4):
                s = self.pos_to_state[(row, col)]
                # We'll store all actions that yield a different state (no self-transitions)
                valid_dict[s] = []
                for a, (dr, dc) in self.base_actions.items():
                    new_r = row + dr
                    new_c = col + dc
                    if 0 <= new_r < 4 and 0 <= new_c < 4:
                        next_s = self.pos_to_state[(new_r, new_c)]
                        # Only count it if next_s != s (which can't happen in 3x3, but let's be explicit)
                        if next_s != s:
                            valid_dict[s].append(a)

        # For terminal states 9 and 10, no actions are valid
        # valid_dict[9] = []
        # valid_dict[10] = []
        valid_dict[16] = []
        valid_dict[17] = []

        return valid_dict

    def reset(self):
        """
        Reset environment to the start:
          - current_state=1
          - visited_cue=False
        Returns current_state (1).
        """
        self.current_state = self.start_state
        self.visited_cue = False
        return self.current_state

    def get_valid_actions(self, state=None):
        """
        Return the list of valid actions for the current state
        (or a given state).
        """
        if state is None:
            state = self.current_state
        return self.valid_actions[state]

    def step(self, action):
        """
        Step with a guaranteed valid action. If an invalid action is given,
        we can either ignore or raise an Exception. We'll raise an error.
        """
        if action not in self.get_valid_actions():
            raise ValueError(f"Action {action} is not valid from state {self.current_state}.")

        # If we're already in a terminal (9 or 10), episode is over.
        if self.current_state in [self.rewarded_terminal, self.unrewarded_terminal]:
            return self.current_state, 0, True

        # Move
        row, col = self.state_to_pos[self.current_state]
        dr, dc = self.base_actions[action]
        next_row = row + dr
        next_col = col + dc

        next_state = self.pos_to_state[(next_row, next_col)]
        
        # if self.current_state == 15
        # if next_state == 16:


        # Check cue
        # if next_state == self.cue_state:
        #     self.visited_cue = True

        reward = 0
        done = False

        # Terminal condition: 8 -> 9
        if next_state == 16:
            done = True
            # if self.visited_cue:
                # reward = +1
            if np.random.rand() < 0.5:
                next_state = self.unrewarded_terminal
            else: 
                # next_state=17       
                next_state = self.rewarded_terminal
            # else:
                # reward = -1
                

        # Update
        self.current_state = next_state
        return next_state, reward, done
    


# ------------------------------------------------------------------------------------
# ------------------------------ Diverging Environments ------------------------------
# ------------------------------------------------------------------------------------


# TODO: Update this
class GridEnvDivergingSingleReward(Environment):
    """
    A 3x3 grid with states numbered 1..9:

        1   2   3
        4   5   6
        7   8   9

    - The agent can only move RIGHT (0) or DOWN (1).
    - No self-transitions at borders:
        If an action would go out of bounds, that action is not allowed
        from that state.
    - Same cue logic: must visit 5 for a +1 reward at 9, else goes to 10 with -1.
    """

    def __init__(self, cue_states=[2], env_size = (4,4), rewarded_terminal=[16]):
        self.env_size = env_size
        self.pos_to_state = {}        
        state = 0
        for i in range(self.env_size[0]):
            for j in range(self.env_size[1]):
                state += 1
                self.pos_to_state[(i,j)] = state

        self.state_to_pos = {s: rc for rc, s in self.pos_to_state.items()}

        unrewarded_terminals = [s+state+1 for s in range(len(rewarded_terminal))]
        start_state = 1

        super().__init__(rewarded_termimals=rewarded_terminal, 
                         unrewarded_termimals=unrewarded_terminals,
                         start = start_state,
                         cues=cue_states)
        
        # Actions as integers: 0=right, 1=down
        # right => (0, +1)
        # down  => (+1, 0)
        self.base_actions = {
            0: (0, +1),   # right
            1: (+1, 0)    # down
        }
        
        # Instead of having a static [0,1] action space, we have a
        # per-state action set (no invalid moves).
        self.valid_actions = self._build_valid_actions()

        self.num_unique_states = state + len(self.unrewarded_terminals) # hard coded for now # done!
        self.reset()

    def _build_valid_actions(self):
        """
        Precompute valid actions for each non-terminal grid state.
        A 'valid' action is one that leads to a NEW in-bounds state.
        """
        valid_dict = {}
        for row in range(4):
            for col in range(4):
                s = self.pos_to_state[(row, col)]
                # We'll store all actions that yield a different state (no self-transitions)
                valid_dict[s] = []
                for a, (dr, dc) in self.base_actions.items():
                    new_r = row + dr
                    new_c = col + dc
                    if 0 <= new_r < 4 and 0 <= new_c < 4:
                        next_s = self.pos_to_state[(new_r, new_c)]
                        # Only count it if next_s != s (which can't happen in 3x3, but let's be explicit)
                        if next_s != s:
                            valid_dict[s].append(a)

        
        # Terminal states have no valid actions
        for terminal in self.unrewarded_terminals:
            valid_dict[terminal] = []
        for terminal in self.rewarded_terminals:
            valid_dict[terminal] = []

        return valid_dict

    def reset(self):
        """
        Reset environment to the start:
          - current_state=1
          - visited_cue=False
        Returns current_state (1).
        """
        self.current_state = self.start_state
        self.visited_cue = False
        return self.current_state

    def get_valid_actions(self, state=None):
        """
        Return the list of valid actions for the current state
        (or a given state).
        """
        if state is None:
            state = self.current_state
        return self.valid_actions[state]

    def step(self, action):
        """
        Step with a guaranteed valid action. If an invalid action is given,
        we can either ignore or raise an Exception. We'll raise an error.
        """
        if action not in self.get_valid_actions():
            raise ValueError(f"Action {action} is not valid from state {self.current_state}.")

        # If we're already in a terminal (9 or 10), episode is over.
        if self.is_terminal(self.current_state):
            return self.current_state, 0, True

        # Move
        row, col = self.state_to_pos[self.current_state]
        dr, dc = self.base_actions[action]
        next_row = row + dr
        next_col = col + dc

        next_state = self.pos_to_state[(next_row, next_col)]

        # Check cue
        if next_state in self.cue_states:
            self.visited_cue = True

        reward = 0
        done = self.is_terminal(next_state)


        # TODO: COME BACK TO THIS

        if done:
            if self.visited_cue and next_state == 16:  # only reward if the cue has been visited and the right terminal state is visited
                reward = +1
                next_state = self.rewarded_terminal
            else:
                reward = -1
                next_state = 17

        # Update
        self.current_state = next_state
        return next_state, reward, done
    



class GridEnvDivergingMultipleReward(Environment):
    """
    A 3x3 grid with states numbered 1..9:

        1   2   3
        4   5   6
        7   8   9

    - The agent can only move RIGHT (0) or DOWN (1).
    - No self-transitions at borders:
        If an action would go out of bounds, that action is not allowed
        from that state.
    - Same cue logic: must visit 5 for a +1 reward at 9, else goes to 10 with -1.
    """

    def __init__(self, cue_state=2):
        # Grid layout: (row, col) -> state
        self.pos_to_state = {
            (0,0): 1, (0,1): 2, (0,2): 3, (0,3): 4,
            (1,0): 5, (1,1): 6, (1,2): 7, (1,3): 8,
            (2,0): 9, (2,1): 10, (2,2): 11, (2,3): 12,
            (3,0): 13, (3,1): 14, (3,2): 15, (3,3): 16,
        }

        self.state_to_pos = {s: rc for rc, s in self.pos_to_state.items()}
        
        # Actions as integers: 0=right, 1=down
        # right => (0, +1)
        # down  => (+1, 0)
        self.base_actions = {
            0: (0, +1),   # right
            1: (+1, 0)    # down
        }
        
        # Instead of having a static [0,1] action space, we have a
        # per-state action set (no invalid moves).
        self.valid_actions = self._build_valid_actions()

        # Special states
        self.start_state = 1
        self.cue_state = cue_state
        self.rewarded_terminal = 16 
        self.terminal_states = {4, 8, 12, 16, 17}

        self.current_state = self.start_state
        self.visited_cue = False

    def _build_valid_actions(self):
        """
        Precompute valid actions for each non-terminal grid state.
        A 'valid' action is one that leads to a NEW in-bounds state.
        """
        valid_dict = {}
        for row in range(4):
            for col in range(4):
                s = self.pos_to_state[(row, col)]
                # We'll store all actions that yield a different state (no self-transitions)
                valid_dict[s] = []
                for a, (dr, dc) in self.base_actions.items():
                    new_r = row + dr
                    new_c = col + dc
                    if 0 <= new_r < 4 and 0 <= new_c < 4:
                        next_s = self.pos_to_state[(new_r, new_c)]
                        # Only count it if next_s != s (which can't happen in 3x3, but let's be explicit)
                        if next_s != s:
                            valid_dict[s].append(a)

        # For terminal states 9 and 10, no actions are valid
        # valid_dict[9] = []
        # valid_dict[10] = []
        valid_dict[16] = []
        valid_dict[17] = []

        return valid_dict

    def reset(self):
        """
        Reset environment to the start:
          - current_state=1
          - visited_cue=False
        Returns current_state (1).
        """
        self.current_state = self.start_state
        self.visited_cue = False
        return self.current_state

    def get_valid_actions(self, state=None):
        """
        Return the list of valid actions for the current state
        (or a given state).
        """
        if state is None:
            state = self.current_state
        return self.valid_actions[state]

    def step(self, action):
        """
        Step with a guaranteed valid action. If an invalid action is given,
        we can either ignore or raise an Exception. We'll raise an error.
        """
        if action not in self.get_valid_actions():
            raise ValueError(f"Action {action} is not valid from state {self.current_state}.")

        # If we're already in a terminal (9 or 10), episode is over.
        if self.current_state in self.terminal_states:
            return self.current_state, 0, True

        # Move
        row, col = self.state_to_pos[self.current_state]
        dr, dc = self.base_actions[action]
        next_row = row + dr
        next_col = col + dc

        next_state = self.pos_to_state[(next_row, next_col)]

        # Check cue
        if next_state == self.cue_state:
            self.visited_cue = True

        reward = 0
        done = False

        # Terminal condition: 8 -> 9

        reward = 0
        done = next_state in self.terminal_states

        if done:
            if self.visited_cue:  # only reward if the cue has been visited and the right terminal state is visited
                reward = +1
                next_state = self.rewarded_terminal  # set all the rewards to 16 - for simplicity sake
            else:
                reward = -1
                next_state = 17

        # Update
        self.current_state = next_state
        return next_state, reward, done
    


# ------------------------------------------------------------------------------------
# -------------------------------- Other Environments --------------------------------
# ------------------------------------------------------------------------------------



class ContinuousTMaze(Environment):
    """
     6|4 ---3--- 7|5
       \    |    /
        \   |   /
         \  |  /
          \ | /
           1|2
    
    6 and 7 rewarded terminal states, while 4 and 5 are unrewarded terminal states
    """
    def __init__(self):
        super().__init__(rewarded_termimals=[6, 7],
                         unrewarded_termimals=[4, 5],
                         start=random.choice([1, 2]),
                         cues=[])
        
        self.num_unique_states = 4 # hard coded for now

        # Special states
        self.current_state = self.start_state

        self.running = True

    def reset(self):
        self.start_state = random.choice([1, 2])  # randomly get put into either of the starting states
        self.current_state = self.start_state
        self.running = True

        return self.current_state

    def get_valid_actions(self, state=None):
        if state == 1 or state == 2 or state == 4 or state == 5:
            return [1]
        elif state == 3:
            return [0, 1]
        else:
            return None

    def step(self, action):
        reward = 0
        
        if self.current_state == 1 or self.current_state == 2:
            self.current_state = 3
            
        elif self.current_state == 3:
            if action == 0:  # left
                if self.start_state == 1:
                    self.current_state = 4  # unrewarded terminal
                else:  # start_state == 2
                    self.current_state = 6  # rewarded terminal
                    reward = 1
                self.running = False
                
            elif action == 1:  # right
                if self.start_state == 1:
                    self.current_state = 7  # rewarded terminal
                    reward = 1
                else:  # start_state == 2
                    self.current_state = 5  # unrewarded terminal
                self.running = False
                
        return self.running, reward
    
    def plot_graph(self, T, niter, highlight_node=0, highlight_node_2=0, save=True, savename='nogrid'):
        # plot_graph_nogrid(transition_probs,graphiter, cue, new_clone,savename=savename)
        # transition_probs,graphiter, cue, new_clone,savename=savename
        n_state = np.shape(T)[0]
        n_action = np.shape(T)[1]
        # T = np.random.rand(n_state, n_action, n_state)
        # Normalize so that each [s,a,:] sums to 1 (like proper probabilities)
        # for s in range(n_state):
        #     for a in range(n_action):
        #         T[s, a, :] /= T[s, a, :].sum()

        # Initialize a directed graph
        G = nx.DiGraph()

        # Add edges
        for s in range(n_state):
            for a in range(n_action):
                for s_next in range(n_state):
                    prob = T[s, a, s_next]
                    if prob > 0:
                        # You can store an edge label that contains the action 
                        # and probability.
                        G.add_edge(
                            f"{s}", f"{s_next}", 
                            label=f"A{a}, p={prob:.2f}"
                        )
        # Create a color list; default 'lightblue', but 'red' for special_node
        colors = []
        for node in G.nodes():
            state = int(node)
            if state == highlight_node:
                colors.append("red")
            elif state == highlight_node_2:
                colors.append("yellow")
            elif self.is_rewarded_terminal(state):
                colors.append("lightgreen")
            elif self.is_unrewarded_terminal(state):
                colors.append("lightcoral")
            else:
                colors.append("lightblue")
        # Layout the graph
        pos = nx.shell_layout(G)  # you can choose any layout

        # Draw nodes and edges
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=colors)
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15)

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.axis('off')
        plt.title("Iteration {}".format(niter))
        plt.show()
