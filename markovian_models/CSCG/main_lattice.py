# sys.path.append('naturecomm_cscg')

# !pip install cairocffi
# !pip install python-igraph==0.9.8
# !pip install cairocffi
# !pip install igraph==0.9.8

from cscg_actions import *
from ged import *
from util import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random
import pickle


# import numpy as np
# import numpy as np

def generate_custom_colors(num_unique_observations):
    # Define a fixed set of custom colors as RGB values
    predefined_colors = np.array([
        [214, 214, 214],
        [85, 35, 157],
        [253, 252, 144],
        [114, 245, 144],
        [151, 38, 20],
        [239, 142, 192],
        [214, 134, 48],
        [140, 194, 250],
        [72, 160, 162],
    ])

    # If the number of unique observations is greater than the number of predefined colors,
    # cycle through the predefined colors to ensure enough colors are available
    if num_unique_observations > len(predefined_colors):
        extra_colors_needed = num_unique_observations - len(predefined_colors)
        additional_colors = np.tile(predefined_colors, (extra_colors_needed // len(predefined_colors) + 1, 1))
        custom_colors = np.vstack((predefined_colors, additional_colors))[:num_unique_observations]
    else:
        custom_colors = predefined_colors[:num_unique_observations]

    return custom_colors


# Function to create a lattice graph with variable nodes, observations, and aliased states
def create_modular_graph_varied(num_nodes=15, num_observations=10000, num_aliased_states=10, num_modules=3):
    if num_nodes < 4:
        raise ValueError("num_nodes must be at least 4 to allow for meaningful connectivity.")

    if num_nodes < num_modules:
        raise ValueError("Number of nodes must be at least equal to the number of modules to form a meaningful structure.")

    # Initialize the adjacency matrix
    T = np.zeros((num_nodes, num_nodes))

    # Calculate the size of each module
    module_size = num_nodes // num_modules

    for module_index in range(num_modules):
        module_start = module_index * module_size
        # For the last module, extend to the end of the node list
        module_end = module_start + module_size if module_index < num_modules - 1 else num_nodes

        # Fully connect nodes within the module
        for i in range(module_start, module_end):
            for j in range(module_start, module_end):
                if i != j:
                    T[i, j] = 1.0
    # Optionally, add sparse inter-module connections
    # Example: Connecting last node of one module to first node of the next module
    for module_index in range(num_modules - 1):
        module_end = (module_index + 1) * module_size - 1
        next_module_start = (module_end + 1) % num_nodes
        T[module_end, next_module_start] = 1.0
        T[next_module_start, module_end] = 1.0
    # connect first and last module
    T[0, num_nodes-1] = 1.0
    T[num_nodes-1,0] = 1.0



    # Generate observations based on random walks on the lattice graph
    states = [np.random.choice(range(num_nodes))]  # Start from a random state
    for _ in range(1, num_observations):
        current_state = states[-1]
        possible_next_states = np.where(T[current_state, :] > 0)[0]
        next_state = np.random.choice(possible_next_states)
        states.append(next_state)


    # Map states to observations with aliasing
    if num_aliased_states > num_nodes or num_aliased_states < 1:
        raise ValueError("num_aliased_states must be between 1 and the number of nodes.")


    unique_obs = np.arange(num_nodes - num_aliased_states)
    for n in range(num_aliased_states):
      unique_obs = np.append(unique_obs,random.choice(unique_obs))
    state_to_obs = unique_obs # Aliasing version

    # Create observation data
    x = state_to_obs[states]

    # plt.matshow(T)
    # plt.show()

    return x

# Function to create a lattice graph with variable nodes, observations, and aliased states
def create_lattice_graph_varied(num_nodes=15, num_observations=10000, num_aliased_states=10):
    if num_nodes < 4:
        raise ValueError("num_nodes must be at least 4 to allow for meaningful connectivity.")

    # Initialize the adjacency matrix
    T = np.zeros((num_nodes, num_nodes))

    # Connect each node to its immediate and second-order neighbors with wrapping
    for i in range(num_nodes):
        for offset in [-2, -1, 1, 2]:  # Immediate and second-order neighbors
            j = (i + offset) % num_nodes
            T[i, j] = 1.0

    # Generate observations based on random walks on the lattice graph
    states = [np.random.choice(range(num_nodes))]  # Start from a random state
    for _ in range(1, num_observations):
        current_state = states[-1]
        possible_next_states = np.where(T[current_state, :] > 0)[0]
        next_state = np.random.choice(possible_next_states)
        states.append(next_state)


    # Map states to observations with aliasing
    if num_aliased_states > num_nodes or num_aliased_states < 1:
        raise ValueError("num_aliased_states must be between 1 and the number of nodes.")


    unique_obs = np.arange(num_nodes - num_aliased_states)
    for n in range(num_aliased_states):
      unique_obs = np.append(unique_obs,random.choice(unique_obs))
    state_to_obs = unique_obs # Aliasing version

    # Create observation data
    x = state_to_obs[states]

    # plt.matshow(T)

    return x

def train(seed, alpha, niter, num_nodes=30, num_observations=50000, num_clones=10, num_modules=3, aliasing=3):
    # var_nodes = np.arange(5,50,5)
    seed = int(seed)
    alpha = float(alpha)    
    niter = int(niter)
    
    # num_nodes = 30
    # num_observations = 50000
    # num_aliased_states = 2  # Adjust this to change the number of aliased states
    # num_clones = 10
    # num_modules=3
    
    
    # var_clones = np.arange(1,15,1)
    # total_modularity_scores = []
    # var_aliasing = np.arange(2,4,1)
    
    
    
    # for num_clones in var_clones:
    modularity_scores = []
    # for aliasing in var_aliasing:
    # for alpha in np.arange(0,1,0.3):
    # aliasing = 3

    num_aliased_states = num_nodes//aliasing  # Adjust this to change the number of aliased states

    # #1. MODULAR GRAPH
    # # print("modular graph".format(num_clones))
    # print("{} nodes, {} aliased states, {} modules, {} alpha".format(num_nodes, num_aliased_states,num_modules, alpha))

    # # Create observation data
    # x = create_modular_graph_varied(num_nodes, num_observations, num_aliased_states, num_modules)
    # a = np.zeros(len(x), dtype=int)

    # n_clones = np.ones(max(x)+1, dtype=np.int64) * num_clones
    # container = TableContainer()
    # # n_clones = np.ones(n_emissions, dtype=np.int64) * nclone
    # # container = TableContainer()
    # filename = 'model_modular_alpha_' + str(alpha) + '_seed_' + str(seed) + '.pkl'
    
    # chmm_ = CHMM_LCM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, container=container,alpha=alpha,seed=seed, filename=filename)  # Initialize the model
    # progression = chmm_.learn_em_T(x, a, n_iter=niter,
    #                                 # term_early=False,
    #                                 )  # Training   use n_iter=1000 for better training
    # # progression = chmm.learn_em_E(x, a, n_iter=1000)  # Training   use n_iter=1000 for better training
    # chmm = chmm_.load_model()
    # # Consolidate learning. Takes a few seconds
    # chmm.pseudocount = 0.0
    # chmm.learn_viterbi_T(x, a, n_iter=niter)
    
    # with open(filename, 'wb') as file:
    #     pickle.dump(chmm, file)
    
    # states = chmm.decode(x, a)[1]
    # n_states = len(np.unique(states))

    # custom_colors = generate_custom_colors(max(x)+1)/256
    # arr = np.arange(max(x)+1)
    # np.random.shuffle(arr)
    # cmap = colors.ListedColormap(custom_colors[arr])

    # temp_output_file = f"modular_graph_num_nodes_{num_nodes}.png"  # Temporary file for each clone
    # graph, modularity_score = plot_graph_modularity(chmm, x, a, output_file=temp_output_file, cmap=cmap)
    # # print('Ground truth number of nodes: {}, number of nodes recovered {}'.format(num_nodes, len(v)))
    # # Display the image inline
    # display(Image(filename=temp_output_file))

    # modularity_scores.append(modularity_score)

    # n_clones = 0
    # for roomid in range(len(container.groups_of_tables)):
    #     print("Room {} has {} tables (clones)".format(roomid, len(container.groups_of_tables[roomid])))
    #     n_clones+=len(container.groups_of_tables[roomid])
    # print("Total clones used: {}".format(n_clones))
    # print("Clones that would have been used by the original code: {}".format(len(container.groups_of_tables) * 5))

# total_modularity_scores.append(modularity_scores)


  #2. LATTICE GRAPH
#   print('\n')
    print("{} clones: lattice graph".format(num_clones))
    print("{} nodes, {} aliased states".format(num_nodes, num_aliased_states))

    # Create observation data
    x = create_lattice_graph_varied(num_nodes, num_observations, num_aliased_states)
    a = np.zeros(len(x), dtype=int)

    n_clones = np.ones(max(x)+1, dtype=np.int64) * num_clones
    # n_clones = np.ones(max(x)+1, dtype=np.int64) * num_clones
    container = TableContainer()
    filename = 'model_lattice_alpha_' + str(alpha) + '_seed_' + str(seed) + '.pkl'
    chmm_ = CHMM_LCM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, container=container,alpha=alpha,seed=seed, filename=filename)  # Initialize the model
    progression = chmm_.learn_em_T(x, a, n_iter=niter,
                                    # term_early=False,
                                    )  # Training   use n_iter=1000 for better training
    chmm = chmm_.load_model()
    # refine learning
    chmm.pseudocount = 0.0
    chmm.learn_viterbi_T(x, a, n_iter=niter)
    
    with open(filename, 'wb') as file:
        pickle.dump(chmm, file)    
    
    states = chmm.decode(x, a)[1]
    n_states = len(np.unique(states))
    n_states

    custom_colors = generate_custom_colors(max(x)+1)/256
    arr = np.arange(max(x)+1)
    np.random.shuffle(arr)

    cmap = colors.ListedColormap(custom_colors[arr])

    temp_output_file = f"lattice_graph_num_clones_{num_clones}.png"  # Temporary file for each clone
    graph, v, g = plot_graph(chmm, x, a, output_file=temp_output_file, cmap=cmap)
    print('Ground truth number of nodes: {}, number of nodes recovered {}'.format(num_nodes, len(v)))
    # Display the image inline
    display(Image(filename=temp_output_file))

def main():
    seed = sys.argv[1:][0]
    alpha = sys.argv[1:][1]
    niter = sys.argv[1:][2]
    train(seed, alpha, niter)

if __name__ == "__main__":
    main()