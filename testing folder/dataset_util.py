import dataset_Zheng as Zheng
import dataset_brain as brain
import dataset_pancreas as pancreas
import dataset_tabula_sapiens as tabula_sapiens

import util
import resolution as resol
import scipy.sparse
import numpy as np
import pandas as pd
import scipy.io

import matplotlib.pyplot as plt
import pdb

import GMM 

def load(dataset_name, convert_names=True):
    if dataset_name == "brain":
        return brain.brain(convert_names=convert_names)
    elif dataset_name == "pancreas":
        return pancreas.pancreas(convert_names=convert_names)
    elif dataset_name == "eye":
        return tabula_sapiens.tabula_sapiens_eye(convert_names=convert_names)
    elif dataset_name == "heart":
        return tabula_sapiens.tabula_sapiens_heart(convert_names=convert_names)
    elif dataset_name == "blood":
        return tabula_sapiens.tabula_sapiens_blood(convert_names=convert_names)
    elif dataset_name == "PBMC":
        return Zheng.Zheng(convert_names=convert_names)
    elif dataset_name == "tongue":
        return tabula_sapiens.tabula_sapiens_tongue(convert_names=convert_names)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def filter(s, 
           label=None,
           min_count=1000, max_count=np.inf,
                   N=None):
    if N is not None and N < s.shape[0]:
        s = s[np.random.choice(s.shape[0], size=N, replace=False),:]

    if label is not None:
        ct = s.obs[label].to_numpy()
        unique_ct, counts = np.unique(ct, return_counts=True)
        valid_ct = unique_ct[(counts > min_count) & (counts < max_count)]
        s = s[s.obs[label].isin(valid_ct),:]

    return s



# Creates a GMM approximation of the datasets PCA
def datasetPCA_GMM(s_in, gmm, k_nn=20):
    
    print("sampling from GMM")
    
    
    s = util.counts2ann(s_in.X, genes=None, 
                        barcodes=None,
                        min_gene_count=0, min_cell_count=0,
                        check_counts=False)
    _, Wg = GMM.sample(gmm)
    s.obsm["X_pca"] = Wg
    s.obs["celltype"] = s_in.obs["celltype"]
    s.obs["leiden"] = s_in.obs["leiden"]
  
    return s

# Creates a GMM approximation of W in the spike under the NMF model
# def NMF_W_GMM(g, gmm, k_nn=20, pca_dim=50):
#     _, bulk = g.spike_and_bulk_theory(homogeneous=False)
#     spike = g.sample_gaussian_spike(gmm)
    
#     s = g.matrix2scanpy(spike + bulk)

#     s.obsm["X_scaled"] = s.X.toarray()
#     util.stabilize(s)
#     print("creating embedding")
#     util.embed(s, pca_dim=pca_dim)
    
#     return s

# Creates NMF approximation or spike,bulk approximations 
# def NMF_approximation(g, approximation_type,
#                            labels,
#                             pca_dim=50, k_nn=20):

    
#     print("creating approximation")
#     spike, bulk = g.spike_and_bulk_theory(homogeneous=False)
#     spikeh, bulkh = g.spike_and_bulk_theory(homogeneous=True)
#     spike_mean, spike_delta = g.center_matrix(spike, labels=labels)
#     if approximation_type == "S":
#         s = g.matrix2scanpy(spike)
#     elif approximation_type == "B":
#         s = g.matrix2scanpy(spike_mean + bulk)
#     elif approximation_type == "SB":
#         s = g.matrix2scanpy(spike + bulk)
#     elif approximation_type == "SBh":
#         s = g.matrix2scanpy(spike + bulkh)
#     elif approximation_type == "sample":
#         s = g.matrix2scanpy(g.sample())
#     else:
#         raise ValueError(f"Approximation type {approximation_type} not found")

#     if approximation_type == "sample":
#         util.scale(s)
#     else:
#         s.obsm["X_scaled"] = s.X.toarray()
        
#     util.stabilize(s)
#     print("creating embedding")
#     util.embed(s, pca_dim=pca_dim)
#     #print("creating graph")
#     #util.form_graph(s, k_nn=k_nn)
#     #util.compute_leiden(s, resolution=resolution)
#     #sc.tl.umap(s)
  
#     return s

# def joint_embedding(s, ss):
#     Xs = s.X
#     Xss = ss.X

#     # check if the number of columns is the same
#     if Xs.shape[1] != Xss.shape[1]:
#         raise ValueError("The number of columns in s and ss must be the same")

#     # row bind Xs and Xss, which are scipy sparse matrices
#     X = scipy.sparse.vstack([Xs, Xss])
#     cell_types = np.concatenate([s.obs['celltype'].to_numpy(), 
#                                  ss.obs['celltype'].to_numpy()])
#     data_types = np.concatenate([np.repeat("data", Xs.shape[0]), 
#                                  np.repeat("simulated", Xss.shape[0])])

#     genes = s.var_names
#     barcodes = np.concatenate([s.obs.index, ss.obs.index])
    
#     s_out = util.counts2ann(X, genes=genes, 
#                             barcodes=barcodes,
#                             min_gene_count=0, min_cell_count=0,
#                             check_counts=False)
#     s_out.obs["celltype"] = cell_types
#     s_out.obs["datatype"] = data_types

#     util.pipeline(s_out, pca_dim=50, k_nn=20, resolution=None)
#     return s_out
    

    

# def celltype_pairs(g):
#     spike_mean, spike_delta, bulk = g.spike_and_bulk_theory(homogeneous=False)
#     cell_types = g.celltype

#     # given the matrix spike_mean and the row labels cell_types, return a dictionary the means across the labels
#     unique_cell_types = np.unique(cell_types)
#     cell_type_means = {}
#     for cell_type in unique_cell_types:
#         cell_type_means[cell_type] = np.mean(spike_mean[cell_types == cell_type], axis=0)

#     # for each entry in the dictionary, find the key that has the closest mean in Euclidean distance and return a dictionary
#     cell_type_pairs = {}
#     for cell_type, mean in cell_type_means.items():
#         closest_cell_type = min(
#             (x for x in cell_type_means if x != cell_type),  # exclude self from comparison
#             key=lambda x: np.linalg.norm(mean - cell_type_means[x])
#         )
#         cell_type_pairs[cell_type] = closest_cell_type

#     return cell_type_pairs

def cutoffs(s, labels, type, 
            min_cluster_size=500,
            k_nn=20, LRI_threshold=0.9, tol=0.02, debug=False):
    if type not in ["global", "local"]:
        raise ValueError("type must be either 'global' or 'local'")

    print("constructing cluster adjacency matrice")
    clusters = resol.construct_clusters(labels)
    if type == "global":
        clustersA = resol.construct_clustersA_global(s.obsm["X_pca"], clusters, k_nn=k_nn)
    elif type == "local":
        clustersA = resol.construct_clustersA_local(s.obsm["X_pca"], clusters, k_nn=k_nn)

    cutoffs = resol.cutoffs(clustersA, 
                            min_cluster_size=min_cluster_size,
                            LRI_threshold=LRI_threshold, tol=tol, debug=debug)
    return cutoffs

# def grid(s, labels, resolutions, type, 
#          min_cluster_size=500,
#          k_nn=20, debug=False):
#     if type not in ["global", "local"]:
#         raise ValueError("type must be either 'global' or 'local'")

#     print("constructing cluster adjacency matrice")
#     clusters = resol.construct_clusters(labels)
#     if type == "global":
#         clustersA = resol.construct_clustersA_global(s.obsm["X_pca"], clusters, k_nn=k_nn)
#     elif type == "local":
#         clustersA = resol.construct_clustersA_local(s.obsm["X_pca"], clusters, k_nn=k_nn)   

#     return resol.grid(clustersA, resolutions, 
#                       min_cluster_size=min_cluster_size,
#                       debug=debug)

def write(s, outfile="temp_dataset.csv"):
    X = s.obsm["X_pca"]
    y = s.obs["leiden"].to_numpy()
    
    data = np.column_stack((y, X))
    columns = ['label'] + [f'PC{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(outfile, index=False)
    