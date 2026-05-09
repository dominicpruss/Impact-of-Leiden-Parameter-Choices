from sklearn.mixture import GaussianMixture

import numpy as np
import pandas as pd

from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import train_test_split
from scipy.stats import chi2

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pdb


# gmm is objectx return by GMM
def sample(gmm):
        gmm_keys = list(gmm.keys())
        n = len(gmm[gmm_keys[0]]["index"])
        k = gmm[gmm_keys[0]]["dimension"]

        Wg = np.zeros((n,k))
        all_cc = np.zeros(n)
        for ct in gmm.keys():
            #print(f"sampling {ct}")
           
            idx = gmm[ct]["index"]
            gmm_results = gmm[ct]["GMM"]
            n_components  = gmm[ct]["n_mixtures"]
            n_samples = np.sum(idx)
            # Sample from mixture model
            # print the probability of each component
            #rint(f"probabilities: {[f'{gmm_results[i]['p']:.3g}' for i in range(n_mixtures)]}")
            
            # Choose components based on their probabilities using vectorized operations
            component_choices = np.random.choice(
                range(n_components),
                size=n_samples,
                p=[gmm_results[i]['p'] for i in gmm_results.keys()]
            )
            all_cc[idx] = component_choices
            
            # Generate all samples for each unique component
            mixture_keys = list(gmm_results.keys())
            for component in np.unique(component_choices):
                # Find indices for this component
                component_indices = np.where(component_choices == component)[0]
                
                mix = mixture_keys[component]
                mean = gmm_results[mix]['mean']
                cov = gmm_results[mix]['cov']
                if cov.ndim == 1:
                    cov = np.diag(cov)
                
                # Generate all samples for this component at once
                Wg_component = np.random.multivariate_normal(
                    mean,
                    cov,
                    size=len(component_indices)
                )
                
                # Find the actual positions where idx is True
                idx_positions = np.where(idx)[0][component_indices]
                # Fill the corresponding rows in Wg using the mapped indices
                Wg[idx_positions,:] = Wg_component
            
        return all_cc,  Wg
    

# Fits GMM for a single cluster
def GMM_X(X, n_mixtures, covariance_type="full"):
    from sklearn.mixture import GaussianMixture
    
    # Fit single GMM with specified number of components
    gmm = GaussianMixture(n_components=n_mixtures, random_state=0,
                          max_iter=1000,
                          covariance_type=covariance_type)
    gmm.fit(X)
    aic = gmm.aic(X)
    bic = gmm.bic(X)
    

    
    # Create dictionary with results
    results = {}
    mixtures = ["mixture" + str(i) for i in range(1, n_mixtures+1)]
    for i,mt in enumerate(mixtures):
        results[mt] = {
            'mean': gmm.means_[i],
            'cov': gmm.covariances_[i],
            'p': gmm.weights_[i]
        }

    return aic, bic,  results



# Fits GMM to W for each cluster specified by labels
def GMM(W, labels, n_mixtures, covariance_type="full",
        min_cluster_size=1000):

        gmm = {}
        for ct in np.unique(labels):
            idx = labels == ct
            if np.sum(idx) < min_cluster_size:
                print(f"skipping GMM for {ct} because it is too small")
                continue
            print(f"fitting GMM for {ct}")
            aic, bic, gmm_X = GMM_X(W[idx, :], n_mixtures, covariance_type)
            gmm[ct] = {"GMM":gmm_X,
                       'index':idx,
                       'n_mixtures':n_mixtures,
                       'dimension':W.shape[1],
                       'aic':aic,
                       'bic':bic}
        return gmm

################################################
# methods for extracting information from GMM
def get_clusters(gmm):
     return list(gmm.keys())
     
def get_cluster_information(gmm, cluster):
     if not cluster in gmm.keys():
          raise ValueError(f"Cluster {cluster} not found in GMM")
     gc = gmm[cluster]
     n_sample = np.sum(gc["index"])
     n_mixtures = gc["n_mixtures"]
     dimension = gc["dimension"]
     aic = gc["aic"]
     bic = gc["bic"]

     return {"n_samples":n_sample,
             "n_mixtures":n_mixtures,
             "mixtures":list(gc["GMM"].keys()),
             "dimension":dimension,
             "aic":aic,
             "bic":bic}

def get_mixture_information(gmm, cluster, mixture):
    if not cluster in gmm.keys():
          raise ValueError(f"Cluster {cluster} not found in GMM")
    gc = gmm[cluster]
    gc_info = get_cluster_information(gmm, cluster)
    if not mixture in gc["GMM"].keys():
          raise ValueError(f"Mixture {mixture} not found in GMM")
    gmixture = gc["GMM"][mixture]
    mean = gmixture["mean"]
    cov = gmixture["cov"]
    p = gmixture["p"]

    n_sample = int(gc_info["n_samples"]*p)
    return {"mean":mean,
            "cov":cov,
            "n":n_sample}
    



# def GMM_to_labels(gmm):
#     n = len(gmm[list(gmm.keys())[0]]["index"])
   
#     labels = np.repeat("other", n)  # Cast to string type
#     for ct in gmm.keys():
#         idx = gmm[ct]["index"].to_numpy()
#         labels[idx] = ct
#     return labels


# Cross validates GMM for a single cluster
# def GMM_validate_X(X, max_n_mixtures=10):
#     aics = []
#     bics = []
#     n_components_range = range(1, max_n_mixtures + 1)
    
#     for k in n_components_range:
#         print(f"Fitting GMM with {k} components")
#         gmm = GaussianMixture(n_components=k, random_state=0, max_iter=1000)
#         gmm.fit(X)
#         aics.append(gmm.aic(X))
#         bics.append(gmm.bic(X))
    
#     # Create DataFrame with results
#     df = pd.DataFrame({
#         "n_components": n_components_range,
#         "AIC": aics,
#         "BIC": bics
#     })
    
#     return df

# # Cross validates GMM for each cluster
# def GMM_validate(W, labels, max_n_mixtures=6):
#         output = []
#         for ct in np.unique(labels):
#             idx = labels == ct
#             if np.sum(idx) < 1000:
#                 continue
#             print(f"cross validating GMM for {ct}, {np.sum(idx)} samples")
#             cX = W[idx, :]
#             df = GMM_validate_X(cX,  max_n_mixtures)
#             df["cluster"] = ct
#             output.append(df)
            
#         return pd.concat(output)


################################################
def visualize_GMM(gmm):
     labels, X = sample(gmm)
    
     PCA_out = {}
     for clusters in gmm.keys():
          print(f"visualizing {clusters}")
          idx = gmm[clusters]["index"]
          X_cluster = X[idx, :]
          mu = np.mean(X_cluster, axis=0)
          X_cluster = X_cluster - mu

          # Apply PCA
          pca = PCA(n_components=2)
          P_cluster = pca.fit_transform(X_cluster)
          PCA_out[clusters] = {'M':P_cluster,
                              'labels':labels[idx]}
     
     # Create a figure with subplots
     n_clusters = len(PCA_out)
     n_cols = min(3, n_clusters)  # Maximum 3 columns
     n_rows = (n_clusters + n_cols - 1) // n_cols  # Ceiling division
     
     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
     if n_clusters == 1:
         axes = np.array([axes])  # Convert to array for consistent indexing
     axes = axes.flatten()
     
     # Create scatter plots for each cluster
     for idx, (cluster, data) in enumerate(PCA_out.items()):
         ax = axes[idx]
         scatter = ax.scatter(data['M'][:, 0], data['M'][:, 1], 
                            c=data['labels'], cmap='tab20', s=6, alpha=0.6)
         ax.set_title(f'Cluster {cluster}')
         ax.set_xlabel('PC1')
         ax.set_ylabel('PC2')
         
         # Add legend if there are multiple components
         unique_labels = np.unique(data['labels'])
         if len(unique_labels) > 1:
             legend = ax.legend(*scatter.legend_elements(),
                              title="Components",
                              loc="best")
     
     # Remove empty subplots if any
     for idx in range(len(PCA_out), len(axes)):
         fig.delaxes(axes[idx])
     
     plt.tight_layout()
     plt.show()
     return None



