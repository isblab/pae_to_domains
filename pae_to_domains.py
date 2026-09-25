"""
[pae_to_domains](https://github.com/isblab/af_pipeline/tree/main/af_pipeline/pae_to_domains/pae_to_domains.py)
============================
Author: Tristan Croll<br />
Github: https://github.com/tristanic/pae_to_domains<br />
Modified by OMG
"""

import igraph
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from networkx.algorithms import community
from af_pipeline.utils.misc_utils import symmetrize_matrix

def domains_from_pae_matrix_label_propagation(
    pae_matrix: np.ndarray,
    pae_power: int = 1,
    pae_cutoff: float = 5.0,
    random_seed: int = 1,
) -> list[list[int]]:
    """ Takes a predicted aligned error (PAE) matrix and uses a fast label propagation
    clustering algorithm to partition the model into approximately rigid regions.

    Refer to [^fast_label_propagation] for more details on the algorithm.

    [^fast_label_propagation]: Traag, V.A., Šubelj, L. "Large network community detection by fast label propagation" (https://doi.org/10.1038/s41598-023-29610-z)


    ## Arguments:

    - **pae_matrix (np.ndarray)**:<br />
        PAE matrix.

    - **pae_power (int, optional)**:<br />
        Each edge in the graph will be weighted proportional to (`1/pae**pae_power`).

    - **pae_cutoff (float, optional)**:<br />
        Edges will be created for residue pairs satisfying: `pae < pae_cutoff`.

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

def obtain_optimal_modularity_resolution(
    G: nx.Graph,
    partition: list[list[int]],
    weight: str = 'weight'
):
    """ Obtain optimal resolution parameter as empirical estimate based on the
    current partition of the graph.

    Refer to Newman et al 2016 (https://arxiv.org/abs/1606.02319)

    ## Arguments:

    - **G (nx.Graph)**:<br />
        The input graph.

    - **partition (list[list[int]])**:<br />
        The current partition of the graph.

    - **weight (str, optional):**:<br />
        The edge attribute to use as weight. Default is 'weight'.

    ## Returns:

    - **float**:<br />
        The optimal resolution parameter.
    """

    graph_weight = G.size(weight=weight)
    graph_weight = 2.0 * graph_weight
    if graph_weight == 0:
        return 1.0

    strengths = dict(G.degree(weight=weight))

    node_to_comm = {
        node: idx for idx, comm in enumerate(partition) for node in comm
    }
    num_communities = len(partition)

    sigma_r = np.zeros(num_communities)
    for idx, community in enumerate(partition):
        sigma_r[idx] = sum(strengths[node] for node in community)

    W_in = sum(
        data.get(weight, 1.0)
        for u, v, data in G.edges(data=True)
        if node_to_comm.get(u) == node_to_comm.get(v)
    )

    graph_weight_in = 2.0 * W_in
    graph_weight_out = graph_weight - graph_weight_in

    expected_int = np.sum(sigma_r ** 2) / graph_weight
    expected_ext = graph_weight - expected_int

    w_in = graph_weight_in / expected_int if expected_int > 0 else 0.0
    w_out = graph_weight_out / expected_ext if expected_ext > 0 else 0.0

    # Safeguard against unresolvable or uniform structures
    if w_in <= 0 or w_out <= 0 or w_in == w_out:
        return 1.0

    return (w_in - w_out) / (np.log(w_in) - np.log(w_out))

def domains_from_pae_matrix_networkx(
    pae_matrix: np.ndarray,
    pae_power: int = 1,
    pae_cutoff: float = 5.0,
    graph_resolution:float = 0.1,
    find_optimal_resolution: bool = True,
) -> list[list[int]]:
    """
    Takes a predicted aligned error (PAE) matrix representing the predicted
    error in distances between each pair of residues in a model, and uses a
    graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    Refer to [^greedy_modularity_communities] for more details on the algorithm.

    [^greedy_modularity_communities]: Clauset, A., Newman, M.E.J., Moore, C. "Finding community structure in very large networks" (https://doi.org/10.1103/PhysRevE.70.066111)

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

    - **find_optimal_resolution (bool, optional)**:<br />
        If True, the function will iteratively adjust the `graph_resolution`
        parameter.

    Returns:

    - **clusters (list)**:<br />
        A list of lists, with each list containing the residues indices
        belonging to one community.
    """

    pae_matrix = symmetrize_matrix(pae_matrix)
    weights = 1/pae_matrix**pae_power

    g = nx.Graph()
    size = weights.shape[0]
    g.add_nodes_from(range(size))
    edges = np.argwhere(pae_matrix < pae_cutoff)
    sel_weights = weights[edges.T[0], edges.T[1]]
    wedges = [(i,j,w) for (i,j),w in zip(edges,sel_weights)]
    g.add_weighted_edges_from(wedges)
    n_iter = 0
    stop = False

    delta = 1e-3
    clusters = community.greedy_modularity_communities(
        g, weight='weight', resolution=graph_resolution
    )

    if not find_optimal_resolution:
        if isinstance(clusters, list):
            return [list(c) for c in clusters]
        else:
            raise ValueError(
                f"""

                Unexpected output type from community detection algorithm.
                Expected a list of frozen sets, but got {type(clusters)}.
                """
            )

    while not stop:
        old_resolution = graph_resolution
        graph_resolution = obtain_optimal_modularity_resolution(
            G=g, partition=clusters, weight='weight'
        )

        Q = community.modularity(g, clusters, weight='weight', resolution=graph_resolution) # type: ignore

        if abs(graph_resolution - old_resolution) < delta:
            stop = True
            print("Stopping due to convergence.")
        else:
            best_clusters = clusters

        if n_iter > 10:
            stop = True
            print("Stopping due to too many iterations.")
        else:
            best_clusters = clusters

    if isinstance(best_clusters, list):
        clusters = [list(c) for c in best_clusters]
    else:
        raise ValueError(
            f"""

            Unexpected output type from community detection algorithm.
            Expected a list of frozen sets, but got {type(best_clusters)}.
            """
        )

    return clusters

def domains_from_pae_matrix_igraph(
    pae_matrix: np.ndarray,
    pae_power: int = 1,
    pae_cutoff: float = 5.0,
    graph_resolution:float = 1,
    random_seed:int = 47,
) -> list[list[int]]:
    """
    Takes a predicted aligned error (PAE) matrix representing the predicted
    error in distances between each pair of residues in a model, and uses a
    graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    Refer to [^leiden_algorithm] for more details on the algorithm.

    [^leiden_algorithm]: Traag, V.A., Waltman, L., van Eck, N.J. "From Louvain to Leiden: guaranteeing well-connected communities" (https://doi.org/10.1038/s41598-019-41695-z)

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

    igraph.set_random_number_generator(random.Random(random_seed))
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