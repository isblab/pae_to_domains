"""
[pae_to_domains](https://github.com/isblab/af_pipeline/tree/main/af_pipeline/pae_to_domains)
===================================

- Based on [^pae_to_domains] by Tristan Croll.

- Functions implementing community-detection algorithms to identify domains from PAE matrices.

- Currently, the following algorithms are implemented:
    - [Fast label propagation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.label_propagation.fast_label_propagation_communities.html)
    - [Greedy modularity communities](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.modularity_max.greedy_modularity_communities.html)
    - [Leiden algorithm](https://igraph.org/python/versions/0.10.1/api/igraph.community.html)

[^pae_to_domains]: Tristan Croll, "Graph-based community clustering approach to extract protein domains from a predicted aligned error matrix": https://github.com/tristanic/pae_to_domains
"""