import math
import random
import collections
import hypernetx as hnx
import hypernetx.algorithms.hypergraph_modularity as hmod

from cotengra.pathfinders.h_louvain import hLouvain

import numpy as np

from cotengra.core import PartitionTreeBuilder
from cotengra.core import ContractionTree
from cotengra.hypergraph import HyperGraph
from cotengra.hyperoptimizers.hyper import register_hyper_function

def modularitygain(edge, edges, sumedges, nodes, weights, degs, vol, volall, alpha):
    clusternodes = set(edges[edge])
    clusterincidentedges = set.union(*[set(nodes[node]) for node in clusternodes])
    intraclusteredges = []
    interclusteredges = []
    for e in clusterincidentedges:
        if set(edges[e]).issubset(clusternodes): intraclusteredges.append(e)
        else: interclusteredges.append(e)
    gain = sum([weights['edge_weight_map'][e] for e in intraclusteredges])/sumedges
    gain = gain + alpha * (sum([degs[d]*(sum([vol[cluster] ** d for cluster in clusternodes]) - sum([vol[cluster] for cluster in clusternodes]) ** d)/(volall ** d) for d in range(2, len(degs))]))/sum(weights['edge_weights'])
    return gain, clusternodes, intraclusteredges, interclusteredges

def mycmn(
    inputs,
    output,
    size_dict,
    weight_nodes="linear",
    weight_edges="log",
    alpha = 2,
    temp = 1,
    parts = 2,
):
    hg = HyperGraph(inputs, output, size_dict)
    for edge in output: hg.remove_edge(edge)

    edges = hg.edges.copy()
    nodes = hg.nodes.copy()
    weights = hg.compute_weights(weight_nodes = weight_nodes, weight_edges = weight_edges)
    vol = {k:v for k,v in enumerate(weights['node_weights'])}
    volall = sum(vol.values())
    sumedges = sum([weights['edge_weight_map'][e] for e in edges])
    degs = [0 for _ in range(hg.get_num_nodes())]
    for k, v in hg.edges.items():
        if len(v) != 1: degs[len(v)] += weights['edge_weight_map'][k]
    
    gainlookup = {edge : modularitygain(edge, edges, sumedges, nodes, weights, degs, vol, volall, alpha) for edge in edges}

    clusterid = hg.get_num_nodes() - 1

    path =[]
    clusterlabels = {i:frozenset({i}) for i in range(hg.get_num_nodes())}
    while len(edges) >= 1:
        # choose random from the 5 max values
        n = len(gainlookup)
        maxgain, clusternodes, intraclusteredges, interclusteredges = random.choices(sorted(gainlookup.values(), key = lambda x: x[0], reverse = True), [(1-temp)**(i)*temp for i in range(n)])[0]
        # Boltzmann weighted choice
        #choice = random.choices(list(gainlookup.keys()), [math.exp(max(v[0],0)/temp)-1 for v in gainlookup.values()])
        #maxgain, clusternodes, intraclusteredges, interclusteredges = gainlookup[choice[0]]
        # choose max value
        #maxgain, clusternodes, intraclusteredges, interclusteredges = max(gainlookup.values(), key = lambda x: x[0])
        if maxgain >= 0 or len(nodes) >= 16:
            #delete intra cluster edges
            for contractionedge in intraclusteredges:
                edges.pop(contractionedge)
                gainlookup.pop(contractionedge)
            #update cluster node
            clusterid += 1
            for node in clusternodes: nodes.pop(node)
            #update vol
            nodes.update({clusterid : interclusteredges})
            vol[clusterid] = sum([vol[i] for i in clusternodes])
            #update surrounding edges
            edges.update({edge : tuple(map(lambda x: clusterid if x in clusternodes else x,list(edges[edge]))) for edge in interclusteredges})
            sumedges = sum([weights['edge_weight_map'][e] for e in edges]) # todo smarter
            path.append(tuple(clusternodes))
            #update clusterlabels
            clusterlabels.update({clusterid : frozenset.union(*[clusterlabels.pop(node) for node in clusternodes])})
            gainlookup.update({edge : modularitygain(edge, edges, sumedges, nodes, weights, degs, vol, volall, alpha) for edge in interclusteredges})
        else:
            break
    tree = ContractionTree.from_path(inputs, output, size_dict, ssa_path=path)
    tree.contract_nodes(list(clusterlabels.values()), optimize="auto-hq", check=True)
    return tree

def to_hypernetxgraph(
    inputs, output, size_dict, weight_nodes="const", weight_edges="log"
):
    hg = HyperGraph(inputs, output, size_dict)
    #for edge in output: hg.remove_edge(edge)
    winfo = hg.compute_weights(weight_nodes = weight_nodes, weight_edges = weight_edges)
    hnxg = hnx.Hypergraph(hg.edges)
    for e,w in winfo["edge_weight_map"].items():
        hnxg.edges[e].weight = w
    return hmod.precompute_attributes(hnxg)

def kumar(inputs,
    output,
    size_dict,
    weight_nodes="linear",
    weight_edges="log",
    parts = 2,
    ):
    hg = to_hypernetxgraph(inputs, output, size_dict, weight_nodes, weight_edges)
    sets = hmod.kumar(hg, 0.01)
    #cluster = len(sets)
    membership = [-1 for _ in range(hg.number_of_nodes())]
    for i,s in enumerate(sets):
        for e in s:
            membership[e] = i
    assert all(i >= 0 for e in membership)
    #otherwise add
    #for i, v in enumerate(membership):
    #    if v == -1:
    #        membership[i] = cluster
    #        cluster += 1
    return membership

def louvain(inputs,
    output,
    size_dict,
    weight_nodes="linear",
    weight_edges="log",
    change_mode = "iter",
    community_factor = 2,
    b = 0.5,
    beta = 1.0,
    parts = 2,
    ):
    seed = random.randint(0, 9999999)
    hg = to_hypernetxgraph(inputs, output, size_dict, weight_nodes, weight_edges)
    hl = hLouvain(HG = hg,hmod_type=hmod.strict,
                                    beta = beta,
                                    delta_it = 0.0001, 
                                    delta_phase = 0.0001,
                                    random_seed = seed,
                                    )
    alphas = []
    for i in range(30):
        alphas.append(1-((1-b)**i))
    A, q2, alphas_out = hl.combined_louvain_alphas(alphas = alphas,
                                                change_mode = change_mode,
                                                random_seed = seed,
                                                community_factor = community_factor)
    membership = [-1 for _ in range(hg.number_of_nodes())]
    for i,s in enumerate(A):
        for e in s:
            membership[e] = i
    assert all(i >= 0 for e in membership)
    return membership
    

register_hyper_function(
    name="mycmn",
    ssa_func=mycmn,
    space={
        "weight_edges": {"type": "STRING", "options": ["const", "log"]},
        "alpha": {"type": "FLOAT", "min": 0.0, "max": 1.0},
        "temp": {"type": "FLOAT", "min": 0.1, "max": 1.0},
    },
    constants={
        "parts": 2,
    },
)

register_hyper_function(
    name="louvain",
    ssa_func=PartitionTreeBuilder(louvain).trial_fn,
    space={
        "weight_edges": {"type": "STRING", "options": ["const", "log"]},
        "change_mode": {"type": "STRING", "options": ["iter", "phase", "communities, modifications"]},
        "community_factor": {"type": "INT", "min": 2, "max": 4},
        "beta": {"type": "FLOAT", "min": 0.9, "max": 1.0},
        "b": {"type": "FLOAT", "min": 0.2, "max": 0.8},
    },
    constants={
        "parts": 2,
    },
)

register_hyper_function(
    name="kumar",
    ssa_func=PartitionTreeBuilder(kumar).trial_fn,
    space={
        "weight_edges": {"type": "STRING", "options": ["const", "log"]},
    },
    constants={
        "parts": 2,
    },
)