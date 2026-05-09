import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import pdb

from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed

import igraph as ig

import util

##############################
# I use three dictionary structures 
# 1. clusters: {cluster: indices of cluster}
# 2. partition: {cluster: {A, X}} where X is the obsm["X_pca"] restricted to the indices of the cluster
#     the labels indices and A is the corresponding knn adjacency matrix
# 3. cutoffs: {cluster: {resolution: labels}} where resolution is the 
#     cutoff resolution and labels is the optimal subclustering of the cluster

# I use two data.frames
# 1. df_LRI: {resolution, cluster, LRI, n_cl} gives the LRI of the cluster
#  at different resolutions
# 2. df_cutoffs: {cluster, resolution} gives the resolution at which the LRI
#  of the cluster is above the threshold

##############################
# true resolutions are resolutions computed from A formed by the full graph
# functions for computing clustering across resolutions given a graph

def construct_clusters(l):
    clusters = {}
    for i in np.unique(l):
        indices = np.where(l == i)[0]
        clusters[i] = indices

    return clusters

def construct_clustersA_local(X, clusters,k_nn=20):
    n = X.shape[0]
    clustersA = {}   
    for key, indices in clusters.items():
        X_partition = X[indices]
        A_partition = util.knn_graph(X_partition, k_nn=k_nn).toarray()
        clustersA[key] = {'A':A_partition, 'cluster_frequency':len(indices)/n,
                          'indices':np.arange(len(indices))}

    return clustersA

def construct_clustersA_global(X, clusters, k_nn=20):
    A = util.knn_graph(X, k_nn=k_nn).toarray()
    clustersA = {}   
    for key, indices in clusters.items():
        clustersA[key] = {'A':A, 'cluster_frequency':1,
                          'indices':indices}

    return clustersA

# Local Rand Index 
def LRI(baseline_subset, partition_vector):
   
    baseline_partition = partition_vector[baseline_subset]
    n = len(baseline_partition)

    if n < 2:
        return 0.0
    total_pairs = n * (n - 1) / 2
    unique, counts = np.unique(baseline_partition, return_counts=True)
    same_subset_pairs = np.sum(counts * (counts - 1) / 2)
    fraction_same_subset = same_subset_pairs / total_pairs
    return fraction_same_subset



# def LRI_grid(A, subset=None, resolutions=None, debug=False):
#     results = []
#     g = ig.Graph.Adjacency(A.tolist(), mode="DIRECTED")
#     l = None
#     for r in resolutions:
#         l = util.leiden_from_graph(g, r, start_l=None)
#         lri = LRI(subset, np.array(l))
#         f = np.array([np.sum(l==ll) for ll in np.unique(l)])/len(l)
#         results.append({'resolution': r, 
#                         'LRI': lri,
#                         'n_clusters': len(np.unique(l)),
#                         'min_freq': np.min(f)})
#     return pd.DataFrame(results)

# find the resolution at which the LRI is below the threshold
def LRI_cutoff(A, 
                    subset=None,
                    LRI_threshold=0.9, 
                    tol=0.05, 
                    debug=True):
 
    if subset is None:
        subset = np.arange(A.shape[0])
    start_resolution = 0
    end_resolution = 1  
    if debug:
        print("finding resolution interval")
    g = ig.Graph.Adjacency(A.tolist(), mode="DIRECTED")
    l = None

    while True:
        l = util.leiden_from_graph(g, end_resolution, 
                                   start_l=l) 
        rand_index = LRI(subset, np.array(l))
        if debug:
            print(start_resolution, end_resolution, rand_index)

        if rand_index <= LRI_threshold:
            break
        start_resolution = end_resolution
        end_resolution *= 2
        

    if debug:
        print("applying bisection search")
    while True:
        mid_resolution = (start_resolution + end_resolution) / 2
        l = util.leiden_from_graph(g, mid_resolution, 
                                   start_l=l) 
        rand_index = LRI(subset, np.array(l))
        if debug:
            print(start_resolution, mid_resolution, end_resolution, rand_index)

        if rand_index <= LRI_threshold:
            end_resolution = mid_resolution
        else:
            start_resolution = mid_resolution

        
       
        if end_resolution - start_resolution < tol:
            break

    # there is some stochasticity in the clustering, so adjust until the LRI is above the threshold
    if debug:
        print("adjusting resolution")
    r = start_resolution
    while True:
        l = util.leiden_from_graph(g, r, 
                                   start_l=l) 
        rand_index = LRI(subset, np.array(l))
        if rand_index <= LRI_threshold:
            break
        r += tol

    if debug:
        print("final resolution")
        print(r, rand_index)

    l = np.array([str(s) for s in l])

    return r, l[subset], rand_index

######################################################
# def grid(clustersA, resolutions, 
#          min_cluster_size=500,
#          debug=False):
#     results = []
    
#     # Process each cluster sequentially
#     for key in clustersA.keys():
#         if debug:
#             print(f"cluster: {key}")
#         A = clustersA[key]['A']
#         subset = clustersA[key]['indices']
#         if len(subset) < min_cluster_size:
#             continue

#         df = LRI_grid(A, subset, resolutions, debug)
#         df["cluster"] = key
#         df["n"] = len(subset)
    
#         results.append(df)

#     return pd.concat(results)


# given the LRI_over_resolutions dataframe, compute the resolution cutoffs
def cutoffs(clustersA, LRI_threshold=0.1, 
            min_cluster_size=500,
            tol=0.05, debug=True,
            debug_down=False, n_jobs=8):
    # Initialize empty cutoffs dictionary
    cutoffs = {}
    
    # Process each cluster sequentially
    for key in clustersA.keys():
        if debug:
            print(f"cluster: {key}")
        A = clustersA[key]['A']
        freq = clustersA[key]['cluster_frequency']
        subset = clustersA[key]['indices']
        if len(subset) < min_cluster_size:
            continue
        r, l, ri = LRI_cutoff(A, subset, 
                              LRI_threshold, tol, 
                              debug=debug_down)
        min_freq = np.min([np.array([np.sum(l==ll) for ll in np.unique(l)])/len(l)])
        cutoffs[key] = {
            "r0": r,
            "cluster_frequency": freq,
            "LRI": ri,
            "n_partitions": len(np.unique(l)),
            "min_freq": min_freq,
            "n": len(subset)
        }

    df = pd.DataFrame({
        "cluster": [key for key in cutoffs.keys()], 
        "r0": [cutoffs[key]["r0"] for key in cutoffs],
        "cluster_frequency": [cutoffs[key]["cluster_frequency"] for key in cutoffs],
        "LRI": [cutoffs[key]["LRI"] for key in cutoffs],
        "n_partitions": [cutoffs[key]["n_partitions"] for key in cutoffs],
        "min_freq": [cutoffs[key]["min_freq"] for key in cutoffs],
        "n": [cutoffs[key]["n"] for key in cutoffs]
    })
    
    return df



