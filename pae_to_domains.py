# Author: Tristan Croll
# Github: https://github.com/tristanic/pae_to_domains
# Modified by OMG

import igraph
import numpy as np
import networkx as nx
from collections import defaultdict
from networkx.algorithms import community

def domains_from_pae_matrix_label_propagation(
    pae_matrix: np.ndarray,
    pae_power: int = 1,
    pae_cutoff: float = 5.0,
    random_seed: int = 1,
) -> list[list[int]]:
    """ Takes a predicted aligned error (PAE) matrix and uses a fast label propagation
    clustering algorithm to partition the model into approximately rigid regions.

    #TODO: Add reference to the algorithm used.

    ## Arguments:

    - **pae_matrix (np.ndarray)**:<br />
        PAE matrix.

    - **pae_power (int, optional)**:<br />
        Each edge in the graph will be weighted proportional to (1/pae**pae_power).

    - **pae_cutoff (float, optional)**:<br />
        Edges will be created for residue pairs satisfying: `pae`<`pae_cutoff`.

    - **random_seed (int, optional)**:<br />
        Random seed for the label propagation algorithm.

    ## Returns:

    - **clusters (list)**:<br />
        A list of lists, with each list containing the residues indices
        belonging to one community.
    """

    weights = 1/pae_matrix**pae_power

    g = nx.Graph()
    size = weights.shape[0]
    g.add_nodes_from(range(size))

    edges = np.argwhere(pae_matrix < pae_cutoff)
    sel_weights = weights[edges.T[0], edges.T[1]]
    wedges = [(i,j,w) for (i,j),w in zip(edges,sel_weights)]
    g.add_weighted_edges_from(wedges)

    clusters = list(community.fast_label_propagation_communities(g, weight='weight' ,seed=random_seed)) # type: ignore
    clusters = [list(c) for c in clusters]

    return clusters

def domains_from_pae_matrix_networkx(
    pae_matrix: np.ndarray,
    pae_power: int = 1,
    pae_cutoff: float = 5.0,
    graph_resolution:float = 1,
) -> list[list[int]]:
    """
    Takes a predicted aligned error (PAE) matrix representing the predicted
    error in distances between each pair of residues in a model, and uses a
    graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    #TODO: Add reference to the algorithm used.

    Arguments:

    - **pae_matrix (np.ndarray)**:<br />
        PAE matrix as a (n_residues x n_residues) numpy array.<br />
        Diagonal elements should be set to some non-zero.

    - **pae_power (int, optional)**:<br />
        Each edge in the graph will be weighted proportional to (1/pae**pae_power).

    - **pae_cutoff (float, optional)**:<br />
        Graph edges will only be created for residue pairs with `pae`<`pae_cutoff`.

    - **graph_resolution (float, optional)**:<br />
        Regulates how aggressive the clustering algorithm is.
        Smaller values lead to larger clusters.
        > [!IMPORTANT]
        > `graph_resolution` should be larger than zero, and values larger than 5
        > are unlikely to be useful.

    Returns:

    - **clusters (list)**:<br />
        A list of lists, with each list containing the residues indices
        belonging to one community.
    """

    weights = 1/pae_matrix**pae_power

    g = nx.Graph()
    size = weights.shape[0]
    g.add_nodes_from(range(size))
    edges = np.argwhere(pae_matrix < pae_cutoff)
    sel_weights = weights[edges.T[0], edges.T[1]]
    wedges = [(i,j,w) for (i,j),w in zip(edges,sel_weights)]
    g.add_weighted_edges_from(wedges)

    clusters = community.greedy_modularity_communities(g, weight='weight', resolution=graph_resolution) # type: ignore

    if isinstance(clusters, list):
        clusters = [list(c) for c in clusters]
    else:
        raise ValueError(
            f"""

            Unexpected output type from community detection algorithm.
            Expected a list of frozen sets, but got {type(clusters)}.
            """
        )

    return clusters

def domains_from_pae_matrix_igraph(
    pae_matrix: np.ndarray,
    pae_power: int = 1,
    pae_cutoff: float = 5.0,
    graph_resolution:float = 1,
) -> list[list[int]]:
    """
    Takes a predicted aligned error (PAE) matrix representing the predicted
    error in distances between each pair of residues in a model, and uses a
    graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    #TODO: Add reference to the algorithm used.

    Arguments:

    - **pae_matrix (np.ndarray)**:<br />
        PAE matrix as a (n_residues x n_residues) numpy array.<br />
        Diagonal elements should be set to some non-zero.

    - **pae_power (int, optional)**:<br />
        Each edge in the graph will be weighted proportional to (1/pae**pae_power).

    - **pae_cutoff (float, optional)**:<br />
        Graph edges will only be created for residue pairs with `pae`<`pae_cutoff`.

    - **graph_resolution (float, optional)**:<br />
        Regulates how aggressive the clustering algorithm is.
        Smaller values lead to larger clusters.
        > [!IMPORTANT]
        > `graph_resolution` should be larger than zero, and values larger than 5
        > are unlikely to be useful.

    Returns:

    - **clusters (list)**:<br />
        A list of lists, with each list containing the residues indices
        belonging to one community.
    """

    weights = 1/pae_matrix**pae_power

    g = igraph.Graph()
    size = weights.shape[0]
    g.add_vertices(range(size))
    edges = np.argwhere(pae_matrix < pae_cutoff)
    sel_weights = weights[edges.T[0], edges.T[1]]
    g.add_edges(edges)
    g.es['weight']=sel_weights

    vc = g.community_leiden(
        weights='weight',
        resolution_parameter=graph_resolution/100,
        n_iterations=-1
    )

    membership = np.array(vc.membership)

    clusters = defaultdict(list)
    for i, c in enumerate(membership):
        clusters[c].append(i)
    clusters = list(sorted(clusters.values(), key=lambda l:(len(l)), reverse=True))
    return clusters