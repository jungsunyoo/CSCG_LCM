
## A Few utility functions

import numpy as np
# from chmm_actions import CHMM, forwardE, datagen_structured_obs_room
import matplotlib.pyplot as plt
import igraph
from matplotlib import cm, colors
import os
import networkx as nx

custom_colors = (
    np.array(
        [
            [214, 214, 214],
            [85, 35, 157],
            [253, 252, 144],
            [114, 245, 144],
            [151, 38, 20],
            [239, 142, 192],
            [214, 134, 48],
            [140, 194, 250],
            [72, 160, 162],
        ]
    )
    / 256
)
if not os.path.exists("figures"):
    os.makedirs("figures")


def graph_edit_distance_nx(chmm, x, a, gt_A, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30):
    # pdb.set_trace()
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)    
    
    # ged
    gt_G = nx.from_numpy_array(gt_A)
    constructed_G = nx.from_numpy_array(A)    
    
    cost = nx.optimize_edit_paths(constructed_G, gt_G, timeout=100)
    try:
        first_cost = next(cost)
        min_ged = first_cost[-1]
    except StopIteration:
        min_ged = np.nan    
    # if next(cost): 
    #     min_ged = next(cost)[-1]
    # else: 
    #     min_ged = np.nan
    # if cost: 
    #     min_ged = next(cost)[-1]
    # else: 
    #     min_ged = np.nan    
    
    return min_ged 
    
import numpy as np
import networkx as nx
from matplotlib import cm

def graph_edit_distance_nx_norm(chmm, x, a, gt_A, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30):
    # pdb.set_trace()
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)    
    
    # Create graphs from adjacency matrices
    gt_G = nx.from_numpy_array(gt_A)
    constructed_G = nx.from_numpy_array(A)
    
    # Create an empty graph G0 with the same number of nodes as gt_G and constructed_G
    G0 = nx.empty_graph(n=len(gt_G))
    
    # Compute GED
    cost_gt_constructed = nx.optimize_edit_paths(constructed_G, gt_G, timeout=100)
    cost_gt_G0 = nx.optimize_edit_paths(gt_G, G0, timeout=100)
    cost_constructed_G0 = nx.optimize_edit_paths(constructed_G, G0, timeout=100)
    
    try:
        min_ged_gt_constructed = next(cost_gt_constructed)[-1]
    except StopIteration:
        min_ged_gt_constructed = np.nan
    
    try:
        min_ged_gt_G0 = next(cost_gt_G0)[-1]
    except StopIteration:
        min_ged_gt_G0 = np.nan
    
    try:
        min_ged_constructed_G0 = next(cost_constructed_G0)[-1]
    except StopIteration:
        min_ged_constructed_G0 = np.nan
    
    # Normalize GED
    if not np.isnan(min_ged_gt_G0) and not np.isnan(min_ged_constructed_G0):
        normalized_ged = min_ged_gt_constructed / (min_ged_gt_G0 + min_ged_constructed_G0)
    else:
        normalized_ged = np.nan
    
    return normalized_ged


def return_A(
    chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30
):
    # pdb.set_trace()
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    # g = igraph.Graph.Adjacency((A > 0).tolist())
    # node_labels = np.arange(x.max() + 1).repeat(chmm.n_clones)[v]
    # if multiple_episodes:
    #     node_labels -= 1
    # colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
    # # out=[]
    # out = igraph.plot(
    #     g,
    #     output_file,
    #     layout=g.layout("kamada_kawai"),
    #     vertex_color=colors,
    #     vertex_label=v,
    #     vertex_size=vertex_size,
    #     margin=50,
    # )

    return A
    
def plot_graph(
    chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30
):
    # pdb.set_trace()
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    g = igraph.Graph.Adjacency((A > 0).tolist())
    node_labels = np.arange(x.max() + 1).repeat(chmm.n_clones)[v]
    if multiple_episodes:
        node_labels -= 1
    colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
    # out=[]
    out = igraph.plot(
        g,
        output_file,
        layout=g.layout("kamada_kawai"),
        vertex_color=colors,
        vertex_label=v,
        vertex_size=vertex_size,
        margin=50,
    )

    return out, v, g

def plot_graph_infomap(
    chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30
):
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    g = igraph.Graph.Adjacency((A > 0).tolist())
    node_labels = np.arange(x.max() + 1).repeat(chmm.n_clones)[v]
    if multiple_episodes:
        node_labels -= 1
    colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
    comms = g.community_infomap()
    print(len(comms))

    out = igraph.plot(
        comms,
        output_file,
        layout=g.layout("kamada_kawai"),
        mark_groups=True,
        vertex_label=v,
        vertex_size=vertex_size,
        margin=50,
    )

    return len(comms)

def plot_graph_modularity(
    chmm, x, a, output_file, cmap=cm.Spectral, multiple_episodes=False, vertex_size=30
):
    states = chmm.decode(x, a)[1]

    v = np.unique(states)
    if multiple_episodes:
        T = chmm.C[:, v][:, :, v][:-1, 1:, 1:]
        v = v[1:]
    else:
        T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    g = igraph.Graph.Adjacency((A > 0).tolist())

    # Convert the directed graph to an undirected graph
    g = g.as_undirected()

    node_labels = np.arange(x.max() + 1).repeat(chmm.n_clones)[v]
    if multiple_episodes:
        node_labels -= 1
    colors = [cmap(nl)[:3] for nl in node_labels / node_labels.max()]
    # comms = g.community_infomap()

    # print(len(comms))
    # Detect communities using the Louvain method
    communities = g.community_multilevel()
    # print(communities)
    modularity_score = g.modularity(communities)
    # print("Modularity Score:", np.round(modularity_score,2))
    # Optionally, visualize the graph with its communities
    out = igraph.plot(communities, output_file, layout=g.layout("kamada_kawai"),
                      mark_groups=True, vertex_label=v, vertex_size=vertex_size, margin=50,)
    # out = igraph.plot(
    #     comms,
    #     output_file,
    #     layout=g.layout("kamada_kawai"),
    #     mark_groups=True,
    #     vertex_label=v,
    #     vertex_size=vertex_size,
    #     margin=50,
    # )

    return out, modularity_score, v, g


def get_mess_fwd(chmm, x, pseudocount=0.0, pseudocount_E=0.0):
    n_clones = chmm.n_clones
    E = np.zeros((n_clones.sum(), len(n_clones)))
    last = 0
    for c in range(len(n_clones)):
        E[last : last + n_clones[c], c] = 1
        last += n_clones[c]
    E += pseudocount_E
    norm = E.sum(1, keepdims=True)
    norm[norm == 0] = 1
    E /= norm
    T = chmm.C + pseudocount
    norm = T.sum(2, keepdims=True)
    norm[norm == 0] = 1
    T /= norm
    T = T.mean(0, keepdims=True)
    log2_lik, mess_fwd = forwardE(
        T.transpose(0, 2, 1), E, chmm.Pi_x, chmm.n_clones, x, x * 0, store_messages=True
    )
    return mess_fwd


def place_field(mess_fwd, rc, clone):
    assert mess_fwd.shape[0] == rc.shape[0] and clone < mess_fwd.shape[1]
    field = np.zeros(rc.max(0) + 1)
    count = np.zeros(rc.max(0) + 1, int)
    for t in range(mess_fwd.shape[0]):
        r, c = rc[t]
        field[r, c] += mess_fwd[t, clone]
        count[r, c] += 1
    count[count == 0] = 1
    return field / count