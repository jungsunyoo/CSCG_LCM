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
from spatial_environments import * #ContinuousTMaze, GridEnvRightDownNoCue, GridEnvRightDownNoSelf, GridEnvDivergingMultipleReward, GridEnvDivergingSingleReward

 
def generate_dataset(env, n_episodes=10, max_steps=20):
    """
    Run 'n_episodes' episodes in the environment. Each episode ends
    either when the environment signals 'done' or when we hit 'max_steps'.

    Returns:
        A list of (state_sequence, action_sequence) pairs.
        - state_sequence: list of visited states
        - action_sequence: list of chosen actions
    """
    dataset = []

    for episode_idx in range(n_episodes):
        # Prepare lists to store states & actions for this episode
        states = []
        actions = []

        # Reset env to start a new episode
        state = env.reset()

        for t in range(max_steps):
            states.append(state)

            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                # No valid actions => we must be in a terminal or stuck
                break

            # Example: pick a random valid action
            action = np.random.choice(valid_actions)
            actions.append(action)

            # Step in the environment
            next_state, reward, done = env.step(action)
            state = next_state

            if done:
                # Also record the final state
                states.append(state)

                # if state == 16:
                #     print(f"rewarded path: {states}")
                break
                
        # Store (states, actions) for this episode
        if done: # only append datasets that reached terminal state
            dataset.append([states, actions])

    return dataset


def generate_inhibition_dataset(env:'GridLatentInhibition', n_episodes=36, max_steps=10):
    """
    Run 'n_episodes' episodes in the environment. Each episode ends
    either when the environment signals 'done' or when we hit 'max_steps'.

    Returns:
        A list of (state_sequence, action_sequence) pairs.
        - state_sequence: list of visited states
        - action_sequence: list of chosen actions
    """
    dataset = []
    phase = 1

    for episode_idx in range(n_episodes):
        if episode_idx > 9:  # end preexposure
            phase = 2
            
        # Prepare lists to store states & actions for this episode
        states = []
        actions = []

        # Reset env to start a new episode
        state = env.reset()

        for t in range(max_steps):
            states.append(state)

            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                # No valid actions => we must be in a terminal or stuck
                break

            # Example: pick a random valid action
            action = np.random.choice(valid_actions)
            actions.append(action)

            # Step in the environment
            next_state, reward, done = env.step(action, phase)

            state = next_state

            if done:
                # Also record the final state
                states.append(state)

                # if state == 16:
                #     print(f"rewarded path: {states}")
                break
                
        # Store (states, actions) for this episode
        if done: # only append datasets that reached terminal state
            dataset.append([states, actions])

    return dataset


def transition_matrix(dataset):
    """
    Given a dataset of episodes, each episode being (states_seq),
    build a 2D count matrix of shape [max_state+1, max_state+1].
    
    NO ACTION!
    
    Returns:
        transition_counts (np.ndarray): counts[s,s_next]
            The number of times we observed (state=s) --> (next_state=s_next).
    """
    # 1) Collect all observed states and actions to determine indexing bounds
    all_states = set()
    # all_actions = set()
    
    for states_seq, _ in dataset:
        for s in states_seq:
            all_states.add(s)

    max_state = max(all_states) if all_states else 0
    
    # 2) Initialize a 3D count array
    #    We'll assume states range from 0..max_state
    #    and actions range from 0..max_action
    transition_counts = np.zeros((max_state+1,  max_state+1), dtype=int)
    
    # 3) Fill in the counts by iterating over each episode's transitions
    for states_seq, _ in dataset:
        # for each step t in the episode
        # print(len(states_seq), len(actions_seq))
        for t in range(len(states_seq)-1):
            s = states_seq[t]
            # a = actions_seq[t]
            s_next = states_seq[t+1]
            transition_counts[s, s_next] += 1
    
    return transition_counts

def transition_matrix_action(dataset):
    """
    Given a dataset of episodes, each episode being (states_seq, actions_seq),
    build a 3D count matrix of shape [max_state+1, max_action+1, max_state+1].
    
    Returns:
        transition_counts (np.ndarray): counts[s, a, s_next]
            The number of times we observed (state=s) --(action=a)--> (next_state=s_next).
    """
    # 1) Collect all observed states and actions to determine indexing bounds
    all_states = set()
    all_actions = set()
    
    for states_seq, actions_seq in dataset:
        for s in states_seq:
            all_states.add(s)
        for a in actions_seq:
            all_actions.add(a)
    
    max_state = max(all_states) if all_states else 0
    max_action = max(all_actions) if all_actions else 0
    
    # 2) Initialize a 3D count array
    #    We'll assume states range from 0..max_state
    #    and actions range from 0..max_action
    transition_counts = np.zeros((max_state+1, max_action+1, max_state+1), dtype=int)
    
    # 3) Fill in the counts by iterating over each episode's transitions
    for states_seq, actions_seq in dataset:
        # for each step t in the episode
        # print(len(states_seq), len(actions_seq))
        for t in range(len(actions_seq)):
            s = states_seq[t]
            a = actions_seq[t]
            s_next = states_seq[t+1]
            transition_counts[s, a, s_next] += 1
    
    return transition_counts

def row_normalize(matrix):
    """
    Returns a row-normalized copy of 'matrix'.
    Each row of the result sums to 1.
    """
    # Convert to float to avoid integer division issues
    matrix = matrix.astype(float)
    
    # Sum over columns, keep dimension for broadcasting
    row_sums = matrix.sum(axis=1, keepdims=True)

    # Avoid division by zero by replacing zeros with 1.0
    row_sums[row_sums == 0] = 1.0
    
    # Divide each row by its sum
    normalized = matrix / row_sums
    
    return normalized
def retrospective_transition_matrix(P, z):
    """
    Given an n x n prospective transition matrix P and a 1D stationary 
    distribution vector z of length n, compute the time-reversed (retrospective)
    transition matrix P_r where:
    
        P_r(i,j) = (z[i] * P(i,j)) / z[j].
        
    Parameters
    ----------
    P : np.ndarray (shape: (n, n))
        Prospective transition matrix (row-stochastic).
    z : np.ndarray (shape: (n,))
        Stationary distribution, satisfying zP = z and sum(z)=1.
        
    Returns
    -------transition_matrix_action
    P_r : np.ndarray (shape: (n, n))
        The retrospective transition matrix.  Often, by default, this might
        be "column-stochastic" if you interpret P_r(i,j) as the probability
        of i -> j in the reversed chain.  In typical Markov chain notation,
        P_r is used as P_r(j|i) = P(X_t = i | X_{t+1} = j), etc.
    """
    # Convert to float arrays (just to ensure no integer division occurs).
    P = np.asarray(P, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Check dimensions
    n, m = P.shape
    if n != m:
        raise ValueError("P must be square.")
    if z.shape[0] != n:
        raise ValueError("z must have length n to match P's dimensions.")
    
    # Check for zeros in z to avoid division by zero
    if np.any(z == 0):
        raise ValueError("z contains zero entries; cannot compute retrospective transitions.")

    # Compute retrospective transition matrix:
    # P_r(i,j) = z[i]/z[j] * P(i,j)
    # Using broadcasting: z[:, None] is shape (n,1), z[None, :] is shape (1,n).
    # So z[:, None] / z[None, :] is shape (n,n).
    # Then we multiply elementwise by P.
    P_r = P * (z[:, None] / z[None, :])
    
    return P_r

def successor_representations(dataset, gamma=0.9, alpha=0.1, n_states=None, n_passes=1):
    """
    Learns the successor representation M via a TD-like update on raw trajectories:
        M[s, :] <- M[s, :] + alpha * ( e_s + gamma*M[s_next, :] - M[s, :] ).
    
    Parameters
    ----------
    states : list or np.ndarray
        Sequence of states visited (s_0, s_1, ..., s_T). 
    gamma : float
        Discount factor.
    alpha : float
        Learning rate.
    n_states : int
        Total number of discrete states. If None, we'll infer from max state in 'states'.
    n_passes : int
        Number of passes (epochs) over the entire dataset to refine the estimate.

    Returns
    -------
    M : np.ndarray of shape (n_states, n_states)
        Learned SR matrix.
    """
    if n_states is None:
        # n_states = int(np.max(states))  # infer if states are 0-based
        # n_states = max(max(state) for state in states)
        n_states = max(max(pair[0]) for pair in dataset) + 1

    # Initialize
    M = np.zeros((n_states, n_states))
    
    # for _ in range(n_passes):
    for states, actions in dataset:
        for t in range(len(states) - 1):
            s  = states[t]
            s_next = states[t+1]

            # One-hot vector for s
            e_s = np.zeros(n_states)
            e_s[s] = 1.0

            # TD update
            M[s, :] += alpha * (e_s + gamma * M[s_next, :] - M[s, :])
            
    return M

def successor_representations(dataset, gamma=0.9, alpha=0.1, n_states=None, n_passes=1):
    """
    Learns the successor representation M via a TD-like update on raw trajectories:
        M[s, :] <- M[s, :] + alpha * ( e_s + gamma*M[s_next, :] - M[s, :] ).
    
    Parameters
    ----------
    states : list or np.ndarray
        Sequence of states visited (s_0, s_1, ..., s_T). 
    gamma : float
        Discount factor.
    alpha : float
        Learning rate.
    n_states : int
        Total number of discrete states. If None, we'll infer from max state in 'states'.
    n_passes : int
        Number of passes (epochs) over the entire dataset to refine the estimate.

    Returns
    -------
    M : np.ndarray of shape (n_states, n_states)
        Learned SR matrix.
    """
    if n_states is None:
        # n_states = int(np.max(states))  # infer if states are 0-based
        # n_states = max(max(state) for state in states)
        n_states = max(max(pair[0]) for pair in dataset)
    # print(n_states)
    # Initialize
    M = np.zeros((n_states, n_states))
    
    for _ in range(n_passes):
        for states, actions in dataset:
            for t in range(len(states) - 1):
                s  = states[t]-1
                s_next = states[t+1]-1
                # print(s,s_next)
                # One-hot vector for s
                e_s = np.zeros(n_states)
                e_s[s] = 1.0

                # TD update
                # M[s, :] += alpha * (e_s + gamma * M[s_next, :] - M[s, :])
                M[s] = (1-alpha) * M[s] + alpha * (e_s + gamma * M[s_next])
            # now do a final update for the last state
            s_terminal = states[-1] - 1
            e_s = np.zeros(n_states)
            e_s[s_terminal] = 1.0
            M[s_terminal] = (1 - alpha) * M[s_terminal] + alpha * e_s
    return M
# def update_SR(self, s, s_new):
#     self.M[s] = (1-self.alpha)* self.M[s] + self.alpha * ( self.onehot[s] + self.gamma * self.M[s_new]  )

def predecessor_representations(dataset, gamma=0.9, alpha=0.1, n_states=None, n_passes=1):
    """
    Learns the predecessor representation matrix P via a TD-like update on raw trajectories:
        P[s_next, :] <- P[s_next, :] + alpha * ( e_{s_next} + gamma * P[s, :] - P[s_next, :] ).

    Parameters
    ----------
    dataset : list of (states, actions)
        Each element is a tuple: (states, actions) where:
           - states is a list/array: s_0, s_1, ..., s_T
           - actions can be ignored here; we only need states for SR/PR learning.
    gamma : float
        Discount factor.
    alpha : float
        Learning rate.
    n_states : int or None
        Total number of discrete states. If None, will infer from max in dataset.
        Assumes states are 1-based, so we do s-1 for zero-based indexing.
    n_passes : int
        Number of passes (epochs) over the entire dataset to refine the estimate.

    Returns
    -------
    P : np.ndarray of shape (n_states, n_states)
        Learned PR matrix.  Row i is the predecessor representation vector for state i.
    """
    # Infer number of states if not given
    if n_states is None:
        # e.g. if states are 1-based, we take the max of them
        n_states = max(max(seq[0]) for seq in dataset)

    # Initialize
    P = np.zeros((n_states, n_states))

    for _ in range(n_passes):
        for states, actions in dataset:
            states = np.array(states)

            # Optional: "first-state update" if you think of s_0 as having no predecessor
            s_first = states[0] - 1
            e_first = np.zeros(n_states)
            e_first[s_first] = 1.0
            # P[s_first, :] <- P[s_first, :] + alpha*( e_first - P[s_first, :] )
            P[s_first] = (1 - alpha)*P[s_first] + alpha * e_first

            # For each transition (s -> s_next), update row of s_next
            for t in range(len(states) - 1):
                s = states[t]     - 1
                s_next = states[t+1] - 1

                e_s_next = np.zeros(n_states)
                e_s_next[s_next] = 1.0

                # TD update:
                # P[s_next, :] <- P[s_next, :] + alpha*( e_s_next + gamma*P[s, :] - P[s_next, :] )
                P[s_next] = (1 - alpha)*P[s_next] + alpha*(e_s_next + gamma * P[s])
                
    return P

def successor_representations_action(dataset, gamma=0.9, alpha=0.1, n_states=None, n_passes=1):
    """
    Learns the successor representation M via a TD-like update on raw trajectories:
        M[s, :] <- M[s, :] + alpha * ( e_s + gamma*M[s_next, :] - M[s, :] ).
    
    Parameters
    ----------
    states : list or np.ndarray
        Sequence of states visited (s_0, s_1, ..., s_T). 
    gamma : float
        Discount factor.
    alpha : float
        Learning rate.
    n_states : int
        Total number of discrete states. If None, we'll infer from max state in 'states'.
    n_passes : int
        Number of passes (epochs) over the entire dataset to refine the estimate.

    Returns
    -------
    M : np.ndarray of shape (n_states, n_states)
        Learned SR matrix.
    """
    if n_states is None:
        # n_states = int(np.max(states))  # infer if states are 0-based
        # n_states = max(max(state) for state in states)
        n_states = max(max(pair[0]) for pair in dataset) 
        n_action = max(max(pair[1]) for pair in dataset) +1

    # Initialize
    M = np.zeros((n_states, n_action, n_states))
    
    # for _ in range(n_passes):
    for states, actions in dataset:
        
        for t in range(len(states) - 1):
            s  = states[t]-1
            a = actions[t]
            s_next = states[t+1]-1
            
            # One-hot vector for s
            e_s = np.zeros(n_states)
            e_s[s] = 1.0

            # TD update
            M[s, a, :] += alpha * (e_s + gamma * M[s_next, :,:] - M[s, a,:])
            
    return M

def compute_eligibility_traces(states, n_states, gamma=0.9, lam=0.8):
    """
    Compute eligibility traces for a single episode's state sequence.
    
    Parameters
    ----------
    states : list or 1D array
        Sequence of visited states (zero-based indices).
    n_states : int
        Total number of discrete states.
    gamma : float
        Discount factor.
    lam : float
        Lambda parameter for eligibility decay.
    
    Returns
    -------
    E : np.ndarray of shape (len(states), n_states)
        E[t, s] = the eligibility of state s after observing the t-th state in 'states'.
    """
    # We'll keep a running "eligibility vector" e for all states,
    # and store its value at each step in E.
    E = np.zeros((len(states), n_states))  # E[t, s] = eligibility of state s at time t
    
    e = np.zeros(n_states)  # current eligibility vector (initially all zeros)
    
    # Iterate over each visited state in the trajectory
    for t, s in enumerate(states):
        # Decay the existing eligibilities
        e *= gamma * lam
        
        # Increment eligibility for the current state by 1
        e[s] += 1.0
        
        # Store a snapshot of the eligibility vector at this time step
        E[t] = e.copy()
        
    return E

def compute_eligibility_traces_normalized(states, n_states, gamma=0.9, lam=0.8):
    """
    Compute a normalized eligibility trace for a single episode's state sequence:
    each time step's eligibility vector sums to 1.
    """
    E = np.zeros((len(states), n_states))  # E[t, s]
    e = np.zeros(n_states)  # current eligibility vector
    
    for t, s in enumerate(states):
        # Decay
        e *= gamma * lam
        
        # Increment for current state
        e[s] += 1.0
        
        # Normalize so sum(e) = 1 (provided sum(e) > 0, which it will be here)
        sum_e = e.sum()
        if sum_e > 0:
            e /= sum_e
        
        E[t] = e.copy()
    
    return E

def compute_transition_entropies(transition_probs, tol=1e-9):
    """
    Given transition_probs[s, a, s_next], compute the Shannon entropy
    (in bits, i.e. log base 2) of each (s, a) distribution.
    
    Returns:
        entropies: A 2D array of shape [S, A], where entropies[s, a]
                   is the entropy of transition_probs[s, a, :].
    
    Notes:
      - If the total probability mass for (s, a) is ~0 (i.e. no data),
        we set entropy to 0 by default (or you could mark it as NaN).
      - We ignore states that are purely out-of-bounds or never visited.
    """
    S, A, _ = transition_probs.shape
    entropies = np.zeros((S, A), dtype=float)
    
    for s in range(S):
        for a in range(A):
            dist = transition_probs[s, a]  # shape = [S]
            
            # Sum of probabilities (should be ~1 if we have data)
            total_prob = dist.sum()
            if total_prob < tol:
                # Means no data or zero-prob distribution
                entropies[s, a] = 0.0
                continue
            
            # Identify the non-zero probabilities (to avoid log(0))
            p_nonzero = dist[dist > tol]
            
            # Normalize them so they sum to 1
            p_nonzero /= p_nonzero.sum()
            
            # Shannon entropy in bits
            #  E = - sum(p * log2(p))
            ent = -np.sum(p_nonzero * np.log2(p_nonzero))
            entropies[s, a] = ent
            
    return entropies

def find_stochastic_state_actions_by_entropy(entropies, eps=1e-9):
    """
    Given a 2D array of entropies[s,a], return a list of (s,a) pairs
    that are strictly > eps in entropy (i.e. non-deterministic).
    """
    stochastic_pairs = []
    S, A = entropies.shape
    for s in range(S):
        for a in range(A):
            # If entropy is basically 0 => deterministic
            if entropies[s, a] > eps:
                stochastic_pairs.append((s,a))
    return stochastic_pairs

# Functions for contingency & splitting

def get_unique_states(dataset):
    all_states = []
    for states_seq, _ in dataset:
        all_states.extend(states_seq)  # Flatten the list
    unique_states = np.unique(all_states)
    return unique_states

def get_unique_states_from_env(env):
    return [x for x in env.pos_to_state.values()]

def has_state(sequence, state):
    """Return True if the episode's state sequence contains state=5."""
    return state in sequence

def has_transition(s,sprime,sequence):
    """Return True if the episode's state sequence contains a transition 15->16."""
    for i in range(len(sequence) - 1):
        if sequence[i] == s and sequence[i + 1] == sprime:
            return True
    return False

# s=12
# sprime=16
# sprime2 = 17
def calculate_contingency_old(dataset, s, sprime, sprime2, env_size):
    unique_states = get_unique_states(dataset)
    contingency_states = []
    for curr_state in unique_states:
        # if curr_state<100:
        # if (curr_state < s or curr_state > 17):    # maybe here
        if (curr_state < s or curr_state > env_size[0]*env_size[1]+1): 
            # the +1 should be generalized with number of "stochastic states" soon
            
            # print(curr_state)
            # episodes_with_state = 0
            # episodes_with_state_and_transition = 0
            # other =0
            # curr_state = 6

            total = 0
            a=0
            b=0
            c=0
            d=0
            conditioned_contingency=0
            # print("Current state: {}".format(curr_state))
            for states_seq, actions_seq in dataset:
                if has_state(states_seq,s):
                    total += 1
                    if has_state(states_seq, curr_state):
                    
                        
                        # episodes_with_state += 1
                        if has_transition(s,sprime,states_seq): 
                            # episodes_with_state_and_transition += 1   
                            a += 1
                            # if curr_state==18:
                            #     print('a:')
                            #     print(states_seq)
                            # print('transition: {}'.format(states_seq))
                        elif has_transition(s,sprime2, states_seq): 
                            # print(states_seq)
                            b+=1
                            # if curr_state==18:
                            #     print('b:')
                            #     print(states_seq)
                    else: 
                        # print('here')
                        if has_transition(s,sprime,states_seq): 
                            # episodes_with_state_and_transition += 1   
                            c += 1
                            # if curr_state==18:
                            #     print('c:')
                            #     print(states_seq)                            
                            
                            # print('transition: {}'.format(states_seq))
                        elif has_transition(s,sprime2, states_seq): 
                            # print(states_seq)
                            d+=1
                            # if curr_state==18:
                            #     print('d:')
                            #     print(states_seq)                            
                    assert total == a+b+c+d
            # if curr_state == 18: 
            #     print(a/(a+b), d/(c+d))
            #     print(a,b,c,d)
            # if a+b != 0: 
            #     print("forward contingency: {}".format(a/(a+b)))
            # else: 
            #     print("no forward contingency")
            # if c+d != 0: 
            #     print("backward contingency: {}".format(d/(c+d)))
            # else: 
            #     print("no backward contingency")
            if a+b !=0 and c+d != 0: 
                # if (a/(a+b)==1 and d/(c+d)==1):
                if a/(a+b)==1: # and d/(c+d)==1):
                    
                    contingency_states.append(curr_state)
        # print(f"state {curr_state} has a value {a} and b value {b} leading to a/(a+b) = {a/(a+b) if a+b != 0 else -1}")
        # print(f"state {curr_state} has a value {a} and c value {c} leading to a/(a+c) = {a/(a+c) if a+c != 0 else -1}")
        
    # print(f"contigency states: {contingency_states}")
    return contingency_states

# s=12
# sprime=16
# sprime2 = 17
def calculate_contingency(dataset, sprime, sprime2, env_size):
    unique_states = get_unique_states(dataset)
    contingency_states = []
    E_r, E_nr = conditioned_eligibility_traces(dataset, env_size, sprime, sprime2)

    E_r[E_r==0] = 1e-3
    # E_nr[E_nr==0] = 1e-3
    E_c = E_r / (E_r + E_nr )    
    # E_c = E_r_ / (E_r_ + E_nr_ )
    possible_cues = np.where(E_c==1)
    
    # Preprocessing because we don't want first stage (0) or terminal state (24)
    # Flatten the nested list to a single list
    flattened = [x for sub in possible_cues for x in sub]

    # Convert to a set to get unique values
    unique_vals = set(flattened)

    # Define the values to exclude
    exclude = {0, env_size[0]*env_size[1]-1}

    # Filter out unwanted values
    result = [val for val in unique_vals if val not in exclude]

    return result
        


def calculate_contingency_tmaze(dataset, s, sprime, sprime2):
    unique_states = get_unique_states(dataset)
    contingency_states = []
    for curr_state in unique_states:
        # if curr_state<100:
        # if (curr_state < s or curr_state > 17):
            # print(curr_state)
            # episodes_with_state = 0
            # episodes_with_state_and_transition = 0
            # other =0
            # curr_state = 6

        total = 0
        a=0
        b=0
        c=0
        d=0
        conditioned_contingency=0
        # print("Current state: {}".format(curr_state))
        for states_seq, actions_seq in dataset:
            if has_state(states_seq,s):
                total += 1
                if has_state(states_seq, curr_state):
                
                    
                    # episodes_with_state += 1
                    if has_transition(s,sprime,states_seq): 
                        # episodes_with_state_and_transition += 1   
                        a += 1
                        # if curr_state==18:
                        #     print('a:')
                        #     print(states_seq)
                        # print('transition: {}'.format(states_seq))
                    elif has_transition(s,sprime2, states_seq): 
                        # print(states_seq)
                        b+=1
                        # if curr_state==18:
                        #     print('b:')
                        #     print(states_seq)
                else: 
                    # print('here')
                    if has_transition(s,sprime,states_seq): 
                        # episodes_with_state_and_transition += 1   
                        c += 1
                        # if curr_state==18:
                        #     print('c:')
                        #     print(states_seq)                            
                        
                        # print('transition: {}'.format(states_seq))
                    elif has_transition(s,sprime2, states_seq): 
                        # print(states_seq)
                        d+=1
                        # if curr_state==18:
                        #     print('d:')
                        #     print(states_seq)                            
                # assert total == a+b+c+d
        # if curr_state == 18: 
        #     print(a/(a+b), d/(c+d))
        #     print(a,b,c,d)
        # if a+b != 0: 
        #     print("forward contingency: {}".format(a/(a+b)))
        # else: 
        #     print("no forward contingency")
        # if c+d != 0: 
        #     print("backward contingency: {}".format(d/(c+d)))
        # else: 
        #     print("no backward contingency")
        if a+b !=0 and c+d != 0: 
            # if (a/(a+b)==1 and d/(c+d)==1):
            if a/(a+b)==1: # and d/(c+d)==1):
                
                contingency_states.append(curr_state)

    return contingency_states

                
def get_successor_states(transition_counts,s,a):
    next_states = transition_counts[s,a]
    sprime = np.where(next_states!=0)[0]
    return sprime


def conditioned_eligibility_traces_old(dataset, env_size):
    n_states = max(max(pair[0]) for pair in dataset) + 1
    E_r = np.zeros((1,n_states))
    E_nr = np.zeros((1,n_states))
    for state_seq, _ in dataset:
        E = compute_eligibility_traces(state_seq, n_states)
        if state_seq[-1] == env_size[0]*env_size[1]:
            # print(E)
            E_r += E[-1,:]
            # print(E_r)
            etmap = np.reshape(E_r[-1,:env_size[0]*env_size[1]], (env_size[0],env_size[1]))
            etmap = np.transpose(etmap)
            # sns.heatmap(etmap) 
            # plt.show()       
            
        else: 
            E_nr += E[-1,:]
    return E_r, E_nr

def conditioned_eligibility_traces(dataset, sprime, sprime2, lam = 0.8, gamma=0.9):
    n_states = max(max(pair[0]) for pair in dataset) + 1
    E_r = np.zeros((1,n_states))
    E_nr = np.zeros((1,n_states))
    for state_seq, _ in dataset:
        E = compute_eligibility_traces(state_seq, n_states, lam=lam)
        if state_seq[-1] == sprime: # like 16 (R)
            # print(E)
            E_r += E[-1,:]
            # print(E_r)
            # etmap = np.reshape(E_r[-1,:env_size[0]*env_size[1]], (env_size[0],env_size[1]))
            # etmap = np.transpose(etmap)
            # sns.heatmap(etmap) 
            # plt.show()       
            
        elif state_seq[-1] == sprime2: # like 17 (nR)
            E_nr += E[-1,:]
    return E_r, E_nr

# def conditioned_eligibility_traces_abstract(dataset, env_size, sprime, sprime2):
#     n_states = max(max(pair[0]) for pair in dataset) + 1
#     E_r = np.zeros((1,n_states))
#     E_nr = np.zeros((1,n_states))
#     for state_seq, _ in dataset:
#         E = compute_eligibility_traces(state_seq, n_states)
#         if state_seq[-1] == sprime: # like 16 (R)
#             # print(E)
#             E_r += E[-1,:]
#             # # print(E_r)
#             # etmap = np.reshape(E_r[-1,:env_size[0]*env_size[1]], (env_size[0],env_size[1]))
#             # etmap = np.transpose(etmap)
#             # sns.heatmap(etmap) 
#             # plt.show()       
            
#         elif state_seq[-1] == sprime2: # like 17 (nR)
#             E_nr += E[-1,:]
#     return E_r, E_nr