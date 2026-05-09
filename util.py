import igraph as ig
import leidenalg as la

import numpy as np
import scanpy as sc
import pandas as pd
import os
import anndata as ad

from scipy.spatial import ConvexHull
from scipy.spatial import distance
import alphashape

from copy import deepcopy

from scipy.stats import nbinom, poisson, norm, uniform, expon
from scipy import sparse
from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD as tSVD
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

from sklearn.metrics import confusion_matrix



import matplotlib.pyplot as plt
import seaborn as sns

#import umap.umap_ as umap


import pdb
import copy



def counts2ann(X, genes=None, barcodes=None, 
               min_gene_count=0, min_cell_count=0,
               obs=None, check_counts=False):
    
    if type(X) == np.ndarray:
        X = sparse.csr_matrix(X)
    s = ad.AnnData(X)
    if barcodes is None:
        barcodes = np.array(["cell" + str(i) for i in range(X.shape[0])])
    if genes is None:
        genes = np.array(["gene" + str(i) for i in range(X.shape[1])])
    
    s.obs_names = barcodes
    s.var_names = genes
    s.var["gene"] = genes
    
    if obs is not None:
        s.obs = obs
        s.obs.index = s.obs.index.astype(str)
    
    if check_counts:
        cell_counts = np.array(s.X.sum(axis=1)).flatten()  # sum rows (cells)
        s = s[cell_counts >= min_cell_count,:]
        gene_counts = np.array(s.X.sum(axis=0)).flatten()  # sum columns (genes)
        s = s[:,gene_counts >= min_gene_count]
 
    return s


def normalize(s, Lbar=10000):

    X = s.X
    rs = np.array(X.sum(axis=1)).flatten()
    X.data /= rs[X.nonzero()[0]]
    X.data *= Lbar

    s.X = X
    return s


def gene_selection(s, n_genes, Poisson=True):
    
    d = sc.pp.highly_variable_genes(s, flavor="seurat_v3",
                                    inplace=False)
    d = d.sort_values(by="variances_norm", ascending=False)
    
    p = s.shape[1]
    n_genes = np.min((p, n_genes))
    top = d.index[:n_genes]
    
    s = s[:,top]
    return s

def scale(s):
    
    X = s.X.toarray()

    L = np.sum(X, 1)
    theta = np.sum(X, 0) / np.sum(X)   
    m = L.reshape(len(L), 1) @ theta.reshape(1, len(theta))

    # add small number to avoid division by zero
    Z = (X - m) / (np.sqrt(m) + 1E-10)
    s.obsm["X_scaled"] = Z


def stabilize(s):
    Z = s.obsm["X_scaled"]
    Z = stabilize_matrix(Z)
    s.obsm["X_scaled"] = Z

# normalization, dim red, graph construction, clustering
# s : annData object
def embed(s, pca_dim=50):
    
    Xr = s.obsm["X_scaled"]
    s.obsm["X_pca"] = PCA(Xr, k=pca_dim)

def form_graph(s, k_nn=20):
    knn = knn_graph(s.obsm["X_pca"], k_nn=k_nn)
    s.uns["neighbors"] = {"connectivities_key": "connectivities",      
                           "distances_key": "distances_knn",
                           'params': {'n_neighbors': k_nn,
                                      'method': 'knn',
                                      'random_state': 0,
                                      'use_rep': "X_pca"}}
    s.obsp["connectivities"] = knn

def compute_leiden(s, resolution=1):
    l = leiden(s.obsp["connectivities"], resolution=resolution)
    s.obs["leiden"] = l

    return l


def pipeline(s, k_nn=20, pca_dim=50, resolution=1):
    scale(s)
    stabilize(s)
    print("creating embedding")
    embed(s, pca_dim=pca_dim)
    print("creating graph")
    form_graph(s, k_nn=k_nn)
    if resolution is not None:
        print("creating leiden")
        compute_leiden(s, resolution=resolution)
    print("creating umap")
    sc.tl.umap(s)


# def normalize_matrix(X, Lbar=10000):
#     L = np.sum(X, 1)
#     X = X / L[:, np.newaxis]
#     X = Lbar * X
#     return X

def scale_matrix(X):
    L = np.sum(X, 1)
    theta = np.sum(X, 0) / np.sum(X)
    m = L.reshape(len(L), 1) @ theta.reshape(1, len(theta))
    X = (X - m) / (np.sqrt(m) + 1E-10)
    return X

def stabilize_matrix(X, clip_cutoff=10):
    # Clip values to the range [-clip_cutoff, clip_cutoff]
    X = np.clip(X, -clip_cutoff, clip_cutoff)
    return X

def permute_matrix(X, celltype, return_sparse=True):
    X = X.toarray()
    unique_celltypes = np.unique(celltype)
    
    for ct in unique_celltypes:
        mask = celltype == ct
        for col in range(X.shape[1]):
            col_values = X[mask, col]
            X[mask, col] = np.random.permutation(col_values)
    
    if return_sparse:
        return sparse.csr_matrix(X)
    else:
        return X

def show_celltypes(s, category, labels, show_other=True):
    ct = deepcopy(s.obs[category].to_numpy())
    ct = np.array([str(cct) for cct in ct])
    # if ct is not in celltypes, replace it with "other"
    mask = ~np.isin(ct, labels) 
    ct[mask] = "other"
    s.obs["show"] = ct
    if show_other:
        sc.pl.umap(s, color="show")
    else:
        sc.pl.umap(s[~mask,:], color=category)

def plot_umap(s, 
              ax=None, title=None, show=True):
    # Get UMAP coordinates and Leiden clusters
    X_umap = s.obsm['X_umap']
    leiden_labels = s.obs['leiden']

    # Create single plot only if ax is not provided
    if ax is None:
        # Create figure with extra width for legend
        fig = plt.figure(figsize=(12, 6))
        # Create main axes that doesn't span full width
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])  # [left, bottom, width, height]
    
    # Plot using the ordered categories and get legend info
    sc.pl.umap(s, color="celltype", ax=ax, show=False, 
               palette='tab20', alpha=1, title=title,
               legend_loc='right margin',  # Move legend to right margin
               legend_fontsize=8,  # Adjust font size if needed
               frameon=True)  # Enable frame
    
    # Add boundary lines for each Leiden cluster
    for cluster in sorted(set(leiden_labels)):
        mask = leiden_labels == cluster
        cluster_points = X_umap[mask]
        
        # Only draw boundary if cluster has points
        if len(cluster_points) >= -1:
            # Calculate distances from center to identify outliers
            center = np.mean(cluster_points, axis=0)
            distances = np.sqrt(np.sum((cluster_points - center)**2, axis=1))
            
            # Remove top 20% outliers
            threshold = np.percentile(distances, 80)
            inlier_mask = distances <= threshold
            filtered_points = cluster_points[inlier_mask]
            
            # Create alpha shape (concave hull) with filtered points
            alpha = 0.3
            hull = alphashape.alphashape(filtered_points, alpha)
            
            # Handle both Polygon and MultiPolygon cases
            if hasattr(hull, 'exterior'):
                # Single polygon case
                boundary = hull.exterior.coords.xy
                ax.plot(boundary[0], boundary[1], 'k-', linewidth=1, alpha=0.5)
                # Use polygon centroid for label
                centroid = hull.centroid
                ax.text(centroid.x, centroid.y, f'C{cluster}',
                       horizontalalignment='center', verticalalignment='center',
                       bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                       fontweight='bold', fontsize=9)
            else:
                # MultiPolygon case
                for polygon in hull.geoms:
                    boundary = polygon.exterior.coords.xy
                    ax.plot(boundary[0], boundary[1], 'k-', linewidth=1, alpha=0.5)
                    # Add label at centroid of each polygon
                    centroid = polygon.centroid
                    ax.text(centroid.x, centroid.y, f'C{cluster}',
                           horizontalalignment='center', verticalalignment='center',
                           bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                           fontweight='bold', fontsize=9)
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax

#############################
# code for future generalization to NB  


def NB_scale(X, residual_cutoff=4):
    
    print("using NB scaling with cutoff" + str(residual_cutoff))
    
    L = np.sum(X, 1)
    theta = np.sum(X, 0)/np.sum(X)
    n = X.shape[0]
    
    mu, dispersion = sctransform(X)
    
    mu_matrix = L.reshape(len(L),1) @ theta.reshape(1, len(theta))
    sigma_matrix = mu_matrix + mu_matrix**2/dispersion
    
    C = X.shape[0]
    # add small number to avoid division by zero
    Y = (X - mu_matrix)/(np.sqrt(sigma_matrix) + 1E-10)
    #ind = np.abs(Y) > np.sqrt(n)
    #Y[ind] = np.sqrt(n)*np.sign(Y[ind])
    ind = np.abs(Y) > residual_cutoff
    Y[ind] = np.abs(norm.rvs(size=np.sum(ind)))*np.sign(Y[ind])
    Y = Y/np.sqrt(C)
    
    return Y

def sctransform(X):
    
      # use scTransform to infer means and dispersions. 
      # scTransform accessed through an R script
   
      in_file = 'temp_in_' + str(np.random.uniform()) + ".csv"
      df = pd.DataFrame(X, 
                        columns=[str(n) for n in range(X.shape[1])])
      df.to_csv(in_file, header=True, index=False)
      out_file = 'temp_out_' + str(np.random.uniform()) + ".csv"
      com = "Rscript sctransform.R " + in_file + " " + out_file
      os.system(com)
      m = pd.read_csv(out_file, header=0)
      os.system("rm " + in_file + " " + out_file)
    
      # sctransform drops genes that have low expression.  For those set
      # mu = 0, dispersion = 1
      G = X.shape[1]
      mean = np.zeros(G)
      dispersion = np.ones(G)
      
      active_genes = m["gene"]
      mean[active_genes] = np.exp(m["log_B"])
      dispersion[active_genes] = m["theta"]
      
      return mean, dispersion

#####################

def PCA(Y, k):
    
    svd = tSVD(n_components=k)

    W = svd.fit_transform(Y)  # U @ S matrix
    return W


########################
# knn

def knn_graph(E, k_nn=20):
    A = compute_knn_graph(E, k_nn=k_nn, return_sparse=True)
    A = (A > 0).astype(int)

    return A


def compute_knn_graph(X, 
                      k_nn=20, 
                      metric='euclidean', 
                      sample_threshold=3000,
                      return_sparse=True):
    
    n_samples = X.shape[0]
    
    if n_samples < sample_threshold:
        # For smaller datasets, use kneighbors_graph
        A = kneighbors_graph(X, 
                            n_neighbors=k_nn,
                            mode='distance',
                            metric=metric,
                            include_self=False)
    else:
        # For larger datasets, use ball_tree
        nn = NearestNeighbors(n_neighbors=k_nn+1,
                                algorithm='ball_tree',
                                metric=metric)
        nn.fit(X)
        
        # Get distances and indices
        distances, indices = nn.kneighbors(X)
        
        # Convert to sparse matrix format
        rows = np.repeat(np.arange(n_samples), k_nn+1)
        cols = indices.ravel()
        data = distances.ravel()
        
        A = csr_matrix((data, (rows, cols)), 
                         shape=(n_samples, n_samples))
        
    if return_sparse:
        return A
    else:
        return A.toarray()

################################
# modularity clustering from a adjacency matrix

import pandas as pd



def cross_community_edge_fractions(A, l):
    unique_labels = np.unique(l)
    data = []

    for i in unique_labels:
        cluster_i = np.where(l == i)[0]
        cluster_not_i = np.where(l != i)[0]
        sub_matrix = A[cluster_i,:]
        all_edges = np.sum(sub_matrix)
        sub_matrix = sub_matrix[:,cluster_not_i]
        external_edges = np.sum(sub_matrix)

        fraction = external_edges / all_edges
        data.append({"Cluster": i, "ExternalEdgeFraction": fraction})

    return pd.DataFrame(data)

def split_modularity(A, l, k):
    n = len(l)
    u_labels = np.unique(l)
    f_c = np.array([np.sum(l==cl) for cl in u_labels])/n
    if len(u_labels) == 1:
        penalty_term = 1
        splitting_term = 0
        return penalty_term, splitting_term, f_c

    penalty_term = np.sum(f_c**2)
    clustering_term = 0
    
    for i, u_label in enumerate(u_labels):
        mask = l == u_label
        non_mask = ~mask
        A_c = A[mask,:][:,non_mask]

        external_average = np.mean(np.sum(A_c, axis=1)/k)
        clustering_term += f_c[i]*external_average

    return penalty_term, clustering_term, f_c
        

def modularity(A, l, resolution):

    m = np.sum(A) # this is 2m in Newmans notation
    Q = 0.0
    communities = np.unique(l)
    
    for c in communities:
        nodes_in_c = np.where(l == c)[0]
        A_c = A[nodes_in_c,:][:,nodes_in_c]
        sum_in = np.sum(A_c)  # Weight of edges within the community
        k_c = np.sum(A[nodes_in_c], axis=1)
        sum_tot = np.sum(k_c)  # Total degree of nodes in the community
        
        Q += (sum_in / m) - resolution*(sum_tot / m) ** 2
    
    return Q



def leiden(W, resolution):
    # Convert sparse matrix to COO format for easy iteration
    if sparse.issparse(W):
        W = W.tocoo()
        # Create graph from edges
        g = ig.Graph(n=W.shape[0], directed=True)
        edges = list(zip(W.row, W.col))
        g.add_edges(edges)
    else:
        # Handle dense matrix case
        g = ig.Graph.Adjacency(W.tolist(), mode="DIRECTED")

    partition = la.find_partition(g, la.RBConfigurationVertexPartition,
                            resolution_parameter = resolution)
    comm = np.array([str(cc) for cc in partition.membership])
    return comm

def leiden_from_graph(g, resolution,
                      start_l=None):
    partition = la.find_partition(g, la.RBConfigurationVertexPartition,
                            resolution_parameter = resolution,
                            initial_membership=start_l)
    return partition.membership
    #comm = np.array([str(cc) for cc in partition.membership])
    #return comm

# deep embeddings give the wrong answer because when I restrict
# A to a subgraph I ignore external edges which leads to 
# underestimating the number of expected edges through the k_i k_j
# term. 
# def leiden_deep(W, resolution):
#     g = ig.Graph.Adjacency(W.tolist(), mode="DIRECTED")
#     l = recursive_leiden(g, resolution)
#     l = np.array([str(ll) for ll in l])
#     return l

# def recursive_leiden(graph, resolution):
#     # Apply Leiden to the current graph
#     partition_type=la.RBConfigurationVertexPartition
#     partition = la.find_partition(graph, partition_type, resolution_parameter=resolution)
    
#     # If the partition is just one community, no further splitting
#     if len(partition) == 1:
#         return [0] * graph.vcount()
    
#     # Otherwise, attempt to recurse into each community
#     membership = [-1] * graph.vcount()
#     community_offset = 0
    
#     for comm_id, comm_vertices in enumerate(partition):
#         subgraph = graph.subgraph(comm_vertices)
#         sub_membership = recursive_leiden(subgraph, resolution)
        
#         # Offset the sub-membership so each subgraph's communities have unique IDs
#         sub_membership = [m + community_offset for m in sub_membership]
        
#         for idx, v in enumerate(comm_vertices):
#             membership[v] = sub_membership[idx]
            
#         community_offset = max(membership) + 1  # Update offset for next community
    
#     return membership

def community_edge_fractions(A, v):
   
    # Identify unique communities and map them to indices 0, 1, ...
    communities = np.unique(v)
    comm_index = {c: idx for idx, c in enumerate(communities)}
    
    # Initialize the fraction matrix
    k = len(communities)
    F = np.zeros((k, k), dtype=float)
    
    # For each pair of communities, count edges from one to the other
    for c_from in communities:
        from_nodes = np.where(v == c_from)[0]
        # Total edges from community c_from
        total_edges_from = np.sum(A[from_nodes], axis=None)
        
        # If no edges originate here, fractions remain zero
        if total_edges_from == 0:
            continue
        
        for c_to in communities:
            to_nodes = np.where(v == c_to)[0]
            # Edges from c_from to c_to
            edges_between = np.sum(A[np.ix_(from_nodes, to_nodes)])
            F[comm_index[c_from], comm_index[c_to]] = edges_between / total_edges_from
    
    return F

import numpy as np

def average_fraction_within_community(A, v):
 
    communities = np.unique(v)
    
    # Dictionary to store average values for each community
    community_averages = {}
    
    for c in communities:
        # Nodes in community c
        nodes_in_c = np.where(v == c)[0]
        community_A = A[nodes_in_c,:][:,nodes_in_c]

        m_I = np.sum(community_A)
        m = np.sum(A[nodes_in_c,:])
        
        # Compute the average k_i for the community
        community_averages[c] = m_I/m
    
    return community_averages


