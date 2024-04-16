from cscg_actions import *
from ged import *
from util import *
import numpy as np
import igraph as ig
import pickle

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



# room = np.array(
#     [
#         [1, 2, 3, 0, 3,],
#         [1, 1, 3, 2, 3,],
#         [1, 1, 2, 0, 1,],
#         [0, 2, 1, 1, 3,],
#         [3, 3, 1, 0, 1,],
#         [2, 1, 2, 3, 3,],
#     ]
# )

# Uncomment this for generating data from a bigger room. Will take longer to train.

room = np.array(
    [
        [1, 2, 3, 0, 3, 1, 1, 1],
        [1, 1, 3, 2, 3, 2, 3, 1],
        [1, 1, 2, 0, 1, 2, 1, 0],
        [0, 2, 1, 1, 3, 0, 0, 2],
        [3, 3, 1, 0, 1, 0, 3, 0],
        [2, 1, 2, 3, 3, 3, 2, 0],
    ]
)


# Plot the layout of the room
cmap = colors.ListedColormap(custom_colors[-4:])
plt.matshow(room, cmap=cmap)
plt.title('Figure 1: Room Layout')
plt.savefig("figures/rectangular_room_layout.pdf")




# import numpy as np


def grid_to_directed_igraph(grid):
    """
    Convert a 2D numpy array to a directed igraph.Graph.
    Each cell has bidirectional connections to its horizontal and vertical neighbors.
    """
    rows, cols = grid.shape
    adjacency_matrix = np.zeros((rows * cols, rows * cols), dtype=int)

    index = lambda r, c: r * cols + c

    for r in range(rows):
        for c in range(cols):
            current_index = index(r, c)

            # North
            if r > 0:
                north_index = index(r - 1, c)
                adjacency_matrix[current_index, north_index] = 1
                adjacency_matrix[north_index, current_index] = 1

            # South
            if r < rows - 1:
                south_index = index(r + 1, c)
                adjacency_matrix[current_index, south_index] = 1
                adjacency_matrix[south_index, current_index] = 1

            # East
            if c < cols - 1:
                east_index = index(r, c + 1)
                adjacency_matrix[current_index, east_index] = 1
                adjacency_matrix[east_index, current_index] = 1

            # West
            if c > 0:
                west_index = index(r, c - 1)
                adjacency_matrix[current_index, west_index] = 1
                adjacency_matrix[west_index, current_index] = 1

    # Creating an igraph from the adjacency matrix
    graph = ig.Graph.Adjacency((adjacency_matrix > 0).tolist(), mode=ig.ADJ_DIRECTED)
    return graph

# Example room array
room = np.array([
    [1, 2, 3, 0, 3, 1, 1, 1],
    [1, 1, 3, 2, 3, 2, 3, 1],
    [1, 1, 2, 0, 1, 2, 1, 0],
    [0, 2, 1, 1, 3, 0, 0, 2],
    [3, 3, 1, 0, 1, 0, 3, 0],
    [2, 1, 2, 3, 3, 3, 2, 0],
])

# directed_igraph = grid_to_directed_igraph(room)
# print("Directed Graph Representation with igraph:")
# print(directed_igraph)



# Generate data from the room and train a CSCG. Takes about a minute
# clones = np.arange(10, 220, 50)
# clones=[70]
# nclone = 70
# import igraph
def train(seed, alpha,niter):
    seed = int(seed)
    alpha = float(alpha)
    niter = int(niter)
    filename = 'model_spatial_alpha_' + str(alpha) + '_seed_' + str(seed) + '.pkl'
    n_emissions = room.max() + 1
    nclone=1
    a, x, rc = datagen_structured_obs_room(room, length=50000)     #Use length=50000 for bigger room
    # for nclone in clones:
    alphas = [alpha]
    # for alpha in np.arange(0,1,0.3):
    for alpha in alphas:
        n_clones = np.ones(n_emissions, dtype=np.int64) * nclone
        container = TableContainer()
        chmm_ = CHMM_LCM(n_clones=n_clones, pseudocount=2e-3, x=x, a=a, container=container,alpha=alpha,seed=seed, filename=filename)  # Initialize the model
        progression = chmm_.learn_em_T(x, a, n_iter=niter,
                                        # term_early=False,
                                        )  # Training   use n_iter=1000 for better training
        # progression = chmm.learn_em_E(x, a, n_iter=1000)  # Training   use n_iter=1000 for better training

        # Consolidate learning. Takes a few seconds
        # viterbi = 
        # load the best model and do viterbi training
        # with open
        chmm = chmm_.load_model()
        # with open(filename, 'rb') as f:
            # loaded_model = pickle.load(f)
        # chmm = loaded_model
        
        
        chmm.pseudocount = 0.0
        chmm.learn_viterbi_T(x, a, n_iter=niter)
        
        with open(filename, 'wb') as file:
            pickle.dump(chmm, file)

        # graph = plot_graph(
        #     chmm, x, a, output_file="figures/rectangular_room_graph.pdf", cmap=cmap
        # )
        # graph

        # cmap = colors.ListedColormap(custom_colors[arr])

        temp_output_file = f"rectangular_room_graph_large_num_clones_{nclone}.png"  # Temporary file for each clone
        graph, v, g = plot_graph(chmm, x, a, output_file=temp_output_file, cmap=cmap)
        print('Ground truth number of nodes: {}, number of nodes recovered {}'.format(len(room.flatten()), len(v)))





        # Display the image inline
        display(Image(filename=temp_output_file))



        n_clones = 0
        for roomid in range(len(container.groups_of_tables)):
            print("Room {} has {} tables (clones)".format(roomid, len(container.groups_of_tables[roomid])))
            n_clones+=len(container.groups_of_tables[roomid])
        print("Total clones used: {}".format(n_clones))
        print("Clones that would have been used by the original code: {}".format(len(container.groups_of_tables) * 70))

    #saving
    


def main():
    seed = sys.argv[1:][0]
    alpha = sys.argv[1:][1]
    niter = sys.argv[1:][2]
    train(seed, alpha, niter)

if __name__ == "__main__":
    main()