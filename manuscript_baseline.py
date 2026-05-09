import dataset_util
import scanpy as sc
import util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import pdb
import os
import pickle
from adjustText import adjust_text
from multiprocessing import Pool

def workfolder():
    # check if "work" folder exists, if not create it
    if not os.path.exists("manuscript_data"):
        os.makedirs("manuscript_data")
    return "manuscript_data"


def get_datasets():
    return ["tongue", "eye", "heart", "PBMC", 
            "brain", "pancreas", "blood"]

def baseline_filename(dataset):
    dir = workfolder() + "/baseline" 
    if not os.path.exists(dir):
        os.makedirs(dir)
    return os.path.join(dir, f"{dataset}_baseline.pkl")

def baseline_figure_filename():
    dir = workfolder() + "/baseline" 
    if not os.path.exists(dir):
        os.makedirs(dir)
    return os.path.join(dir, "baseline_figure.pdf")

def baseline_figure_filename_SI():
    dir = workfolder() + "/baseline" 
    if not os.path.exists(dir):
        os.makedirs(dir)
    return os.path.join(dir, "baseline_figure_SI.pdf")

def pareto_frontier_filename():
    dir = workfolder() + "/baseline" 
    if not os.path.exists(dir):
        os.makedirs(dir)
    return os.path.join(dir, "pareto_frontier.pdf")

################################################################################
def make():
    create_all_baselines(overwrite=False)
    create_baseline_figure(min_cluster_size=1000,
                           number_celltypes=10,
                           overwrite=False)
    create_baseline_SI_figure(min_cluster_size=1000,
                           number_celltypes=10,
                           overwrite=False)
    create_pareto_frontier_figure()

##### figure creation
def create_pareto_frontier_figure():
    save_path = pareto_frontier_filename()
    s = dataset_util.load("heart")
    # sample 1000 cells from s
    s = s[np.random.choice(s.shape[0], 5000, replace=False), :]
    util.pipeline(s, pca_dim=50, resolution=0.1, k_nn=20)
    A = s.obsp["connectivities"]

    res1 = np.arange(0.01, .5, 0.01)
    res2 = np.arange(0.51, 1.0, 0.01)
    res3 = np.arange(1.01, 10, 0.1)
    res4 = np.arange(10.1, 100, 1)
    res5 = np.arange(100.1, 5000, 100)
    res = np.concatenate([res1, res2, res3, res4])
    g = np.repeat(0.0, len(res))
    h = np.repeat(0.0, len(res))
    gamma = np.repeat(0.0, len(res))
    for i, r in enumerate(res):
        print(f"Computing for resolution {r}")
        util.compute_leiden(s, resolution=r)
        l = s.obs["leiden"]
        print(f"Number of clusters: {len(np.unique(l))}")
        ch, cg, _ = util.split_modularity(A, l, 20)
        
        g[i] = cg
        h[i] = ch
        gamma[i] = r

    # put g, h, r into a dataframe
    tol = 1/s.shape[0]
    df = pd.DataFrame({"g": g, "h": h, "r": gamma})
    # sort df based on g in increasing order.  If a particular row has a g value within 1E-6 of the previous, remove the row.
    df = df.sort_values(by="g")
    df = df[~np.isclose(df["g"], df["g"].shift(), atol=tol)]
    
    # Create figure
    fig = plt.figure(figsize=(8, 3))
    
    # plot g vs h, color by r, add a legend
    scatter = plt.scatter(df["g"], df["h"], 
                c=np.log(1+df["r"]), cmap="viridis", s=40)  # Reduced point size to 80% of default (50)
    plt.xlabel('$g_i$', fontsize=14)
    plt.ylabel('$h_i$', fontsize=14)
    plt.xlim(-.02, np.max(df["g"])+0.1)
    plt.ylim(0, 0.6)
    cbar = plt.colorbar(scatter, label='log(1+$\\gamma_i$)')
    max_log_r = 5
    cbar.set_ticks([0, max_log_r/2, max_log_r])
    cbar.set_ticklabels(['0', f'{max_log_r/2:.1f}', f'{max_log_r:.1f}'])
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to free memory

    return df

def create_baseline_figure(min_cluster_size=500,
                           number_celltypes=10,
                           overwrite=False):
    save_path = baseline_figure_filename()
    sd = load_all_baselines()

    # Order datasets alphabetically
    datasets = sorted(["PBMC", "heart", "tongue"])
    sd_s = {}
    for d in datasets:
        print(f"Loading {d} dataset")
        s = sd[d]
        # for readability remove celltypes with low counts
        s = dataset_util.filter(s, label="leiden", 
                                min_count=min_cluster_size)
        s.obs["celltype"] = baseline_figure_categories(s.obs["celltype"], 
                                                        number_celltypes)
        sd_s[d] = s

    # Create a figure with 1x3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Map axes to their positions
    axes = {
        dataset: ax for dataset, ax in zip(datasets, [ax1, ax2, ax3])
    }
    
    # Plot each UMAP and add panel labels
    for i, (dataset, ax) in enumerate(axes.items()):
        ax_out = util.plot_umap(sd_s[dataset], ax=ax, 
                                           title=dataset.upper(), show=False)
        
        # Add panel label with larger font size
        ax.text(-0.1, 1.1, chr(65 + i), transform=ax.transAxes, 
                size=24, weight='bold')
        
        # Make title bigger and uppercase
        ax.set_title(dataset.upper(), fontsize=24)
        
        # Ensure border is visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
    
    # Adjust layout to make room for legends
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Slightly increased right margin for legends
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to free memory
    
    return None

def baseline_figure_categories(l, n=10):
  
    unique, counts = np.unique(l, return_counts=True)
    sorted_indices = np.argsort(-counts)  # Negative for descending order
 
    top_labels = unique[sorted_indices[:n]]
    lp = np.full_like(l, 'other', dtype=object)
    
    mask = np.isin(l, top_labels)
    lp[mask] = l[mask]
    
    return lp

def create_baseline_SI_figure(min_cluster_size=500,
                           number_celltypes=10,
                           overwrite=False):
    save_path = baseline_figure_filename_SI()
    sd = load_all_baselines()

    # Get remaining datasets and order alphabetically
    main_datasets = set(["PBMC", "heart", "tongue"])
    all_datasets = set(get_datasets())
    datasets = sorted(list(all_datasets - main_datasets))  # This will be ['blood', 'brain', 'eye', 'pancreas']
    
    sd_s = {}
    for d in datasets:
        print(f"Loading {d} dataset")
        s = sd[d]
        # for readability remove celltypes with low counts
        s = dataset_util.filter(s, label="leiden", 
                                min_count=min_cluster_size)
        s.obs["celltype"] = baseline_figure_categories(s.obs["celltype"], 
                                                        number_celltypes)
        sd_s[d] = s

    # Create a figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 10))
    
    # Map axes to their positions
    axes = {
        dataset: ax for dataset, ax in zip(datasets, [ax1, ax2, ax3, ax4])
    }
    
    # Plot each UMAP and add panel labels
    for i, (dataset, ax) in enumerate(axes.items()):
        ax_out = util.plot_umap(sd_s[dataset], ax=ax, 
                                           title=dataset.upper(), show=False)
        
        # Add panel label with larger font size
        ax.text(-0.1, 1.1, chr(65 + i), transform=ax.transAxes, 
                size=24, weight='bold')
        
        # Make title bigger and uppercase
        ax.set_title(dataset.upper(), fontsize=24)
        
        # Ensure border is visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
    
    # Adjust layout to make room for legends
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Slightly increased right margin for legends
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to free memory
    
    return None

################################################################################


# Baseline dictionary contains the baseline resolution and scanpy object for each dataset.
# The scanpy object has been filtered and processed with the pipeline.
def create_all_baselines(overwrite=False):
    datasets = get_datasets()
    for d in datasets:
        print(f"Creating baseline for {d} dataset")
        create_baseline(d, overwrite=overwrite)
    return None

def load_all_baselines():
    datasets = get_datasets()
    b = {}
    for d in datasets:
        b[d] = load_baseline(d)
    return b

def load_baseline(dataset):
    infile = baseline_filename(dataset)
    if not os.path.exists(infile):
        print(f"Skipping {dataset} because it does not exist")
        return None
    with open(infile, 'rb') as f:
        b = pickle.load(f)
    return b


def create_baseline(d_name, overwrite=False):
    outfile = baseline_filename(d_name)
    if os.path.exists(outfile) and not overwrite:
        print(f"Skipping {d_name} because it already exists")
        return None

    #_,_,r = me.modularity2elbow(d_name)
    r = 0.1
    print(f"Baseline resolution for {d_name} dataset: {r}")

    print(f"Loading {d_name} dataset")
    s = dataset_util.load(d_name, convert_names=True)
    print(f"Shape of {d_name} dataset: {s.shape}")
    print(f"Running pipeline on {d_name} dataset")
    util.pipeline(s, pca_dim=50, resolution=r, k_nn=20)
  
    s.uns["baseline_resolution"] = r
    with open(outfile, 'wb') as f:
        pickle.dump(s, f)

