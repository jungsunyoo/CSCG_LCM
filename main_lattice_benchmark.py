from cscg_actions_orig import *

from ged import *
from util import *
import numpy as np
import igraph as ig
import pickle
import random
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

# def train(seed, nclone, niter):
def train(seed, nclone, niter, num_nodes=30, num_observations=50000, num_modules=3, aliasing=3):
    seed = int(seed)
    # alpha = float(alpha)
    niter = int(niter)
    nclone = int(nclone)
    num_aliased_states = num_nodes//aliasing  # Adjust this to change the number of aliased states
    print("{} nodes, {} aliased states, {} nclones".format(num_nodes, num_aliased_states,nclone))

    filename = 'model_lattice_benchmark_nclone_' + str(nclone) + '_seed_' + str(seed) + '.pkl'
    # n_emissions = room.max() + 1
    # nclone=1
    # a, x, rc = datagen_structured_obs_room(room, length=50000)     #Use length=50000 for bigger room
    # for nclone in clones:
    # alphas = [alpha]
    # for alpha in np.arange(0,1,0.3):
    # for alpha in alphas:
    # n_clones = np.ones(n_emissions, dtype=np.int64) * nclone
    
    
    # Create observation data
    x = create_lattice_graph_varied(num_nodes, num_observations, num_aliased_states)
    a = np.zeros(len(x), dtype=int)

    n_clones = np.ones(max(x)+1, dtype=np.int64) * nclone   
    
    
    # container = TableContainer()
    # chmm_ = CHMM_LCM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, container=container,alpha=alpha,seed=seed, filename=filename)  # Initialize the model
    # progression = chmm_.learn_em_T(x, a, n_iter=niter,
    #                                 # term_early=False,
    #                                 )  # Training   use n_iter=1000 for better training
    chmm = CHMM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, seed=seed)  # Initialize the model
    progression = chmm.learn_em_T(x, a, n_iter=niter, term_early=False)  # Training


    
    chmm.pseudocount = 0.0
    chmm.learn_viterbi_T(x, a, n_iter=niter)
    
    with open(filename, 'wb') as file:
        pickle.dump(chmm, file)




def main():
    seed = sys.argv[1:][0]
    nclone = sys.argv[1:][1]
    niter = sys.argv[1:][2]
    train(seed, nclone, niter)

if __name__ == "__main__":
    main()