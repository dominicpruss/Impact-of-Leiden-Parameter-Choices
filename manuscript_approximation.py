# Compares local r0 for the baseline datasets to the GMM approximations

import dataset_util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os

import manuscript_baseline as mb
import manuscript_GMM as mGMM
from scipy import stats
import multiprocessing as mp
from functools import partial


def cutoff_folder():
    dir = mb.workfolder() + "/" + "cutoff"
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def cutoff_table_filename(dataset, approximation):
    return cutoff_folder() + "/" + "cutoff_" + dataset + "_" + approximation + ".csv"


def make():
    create_all_cutoff_tables(approx=None, 
                             min_cluster_size=500,
                             overwrite=False,
                             parallel=True)
    # create_cutoff_figure(approx=mGMM.get_approximations(),
    #                      outfile="cutoff_data_vs_GMM.pdf",
    #                      min_cluster_size=1000)
    create_n_mixtures_figure(ic="aic", reduce_value=0)  
    
   
def approximation_types():
    G_approximations = mGMM.get_approximations()
    # append both data and sample separately
    G_approximations.extend(["data"])

    return G_approximations

# def create_cutoff_figure(approx,
#                          outfile,
#                          min_cluster_size=1000):
#     outfile = cutoff_folder() + "/" + outfile
#     t = merge_cutoff_tables(approx)
#     t = t[t['n'] >= min_cluster_size]
    
#     # Calculate number of rows and columns for subplots
#     n_plots = len(approx)
#     n_cols = min(3, n_plots)  # Max 3 columns
#     n_rows = (n_plots + n_cols - 1) // n_cols
    
#     # Create figure
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
#     if n_rows * n_cols > 1:
#         axes = axes.flatten()
#     else:
#         axes = [axes]
    
#     # Get unique datasets for coloring
#     datasets = sorted(t['dataset'].unique())
#     colors = plt.cm.tab20(np.linspace(0, 1, len(datasets)))
#     dataset_colors = dict(zip(datasets, colors))
    
#     # Create scatter plots
#     for idx, approximation in enumerate(approx):
#         ax = axes[idx]
#         mask = t['approximation'] == approximation
#         data = t[mask]
        
#         # Plot points for each dataset
#         for dataset in datasets:
#             mask_dataset = data['dataset'] == dataset
#             ax.scatter(data[mask_dataset]['r0_true'], 
#                       data[mask_dataset]['r0_pred'],
#                       c=[dataset_colors[dataset]],
#                       label=dataset,
#                       alpha=0.7,
#                       s=50)
        
#         # Add x=y line
#         lims = [
#             min(data['r0_true'].min(), data['r0_pred'].min()),
#             max(data['r0_true'].max(), data['r0_pred'].max())
#         ]
#         ax.plot(lims, lims, 'k--', alpha=0.5)
        
#         # Add regression line
#         x = data['r0_true'].values
#         y = data['r0_pred'].values
#         slope, intercept = np.polyfit(x, y, 1)
#         ax.plot(x, slope * x + intercept, 'r-', alpha=0.5)
        
#         # Calculate and add R²
#         r_squared = stats.pearsonr(x, y)[0] ** 2
#         ax.text(0.98, 0.02, f'R² = {r_squared:.3f}',
#                 transform=ax.transAxes,
#                 horizontalalignment='right',
#                 verticalalignment='bottom')
        
#         # Labels and title
#         ax.set_xlabel('True r0')
#         ax.set_ylabel('Predicted r0')
#         ax.set_title(f'Approximation: {approximation}')
    
#     # Remove empty subplots if any
#     for idx in range(len(approx), len(axes)):
#         fig.delaxes(axes[idx])
    
#     # Add legend to the right of the figure
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(handles, labels, 
#               bbox_to_anchor=(1.05, 0.5),
#               loc='center left',
#               title='Dataset')
    
#     # Save figure
#     plt.tight_layout()
#     plt.savefig(outfile, bbox_inches='tight')
#     plt.show()
#     plt.close()
    
#     return None

def create_n_mixtures_figure(ic="aic", reduce_value=0):
    outfile = cutoff_folder() + "/" + "aic_nmixtures.pdf"

    tG = mGMM.n_mixtures_selection_table(ic, reduce_value=reduce_value)
    t = merge_cutoff_tables(approximation_types())

    tG["cluster"] = tG["cluster"].astype(str)
    t["cluster"] = t["cluster"].astype(str)
    
    # Merge tG with t to get r0_true and r0_pred values
    merged_df = tG.merge(
        t[['dataset', 'cluster', 'approximation', 'r0_true', 'r0_pred']],
        on=['dataset', 'cluster', 'approximation'],
        how='left'
    )

    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 2.1))
    
    # Left panel (scatter plot)
    n_mixtures_values = sorted(merged_df['n_mixtures'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_mixtures_values)))
    
    for n_mix, color in zip(n_mixtures_values, colors):
        mask = merged_df['n_mixtures'] == n_mix
        ax1.scatter(merged_df[mask]['r0_true'], 
                   merged_df[mask]['r0_pred'],
                   c=[color],
                   label=str(n_mix),
                   alpha=0.7,
                   s=50)
    
    # Add x=y line
    lims = [
        min(merged_df['r0_true'].min(), merged_df['r0_pred'].min()),
        max(merged_df['r0_true'].max(), merged_df['r0_pred'].max())
    ]
    ax1.plot(lims, lims, 'k-', alpha=0.5)
    
    # Add regression line
    x = merged_df['r0_true'].values
    y = merged_df['r0_pred'].values
    slope, intercept = np.polyfit(x, y, 1)
    ax1.plot(x, slope * x + intercept, 'r--', alpha=0.5)
    
    # Calculate and add R²
    r_squared = stats.pearsonr(x, y)[0] ** 2
    ax1.text(0.98, 0.02, f'R² = {r_squared:.3f}',
             transform=ax1.transAxes,
             horizontalalignment='right',
             verticalalignment='bottom')
    
    # Labels for left panel
    ax1.set_xlabel('True $\\gamma_s$', fontsize=14)
    ax1.set_ylabel('Predicted $\\gamma_s$', fontsize=14)
    ax1.legend(title='Number of Mixtures')
    
    # Add panel label A
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, 
             fontsize=16, fontweight='bold')

    # Right panel (bar plot)
    n_mix_counts = merged_df['n_mixtures'].value_counts().sort_index()
    bars = ax2.bar(n_mix_counts.index, n_mix_counts.values, 
                   color=colors[:len(n_mix_counts)])
    
    # Labels for right panel
    ax2.set_xlabel('Number of Mixtures', fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Add panel label B
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, 
             fontsize=16, fontweight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return None

#################################################
# cutoff tables compute the cutoff resolution for different approximations

def create_all_cutoff_tables(approx=None, 
                             min_cluster_size=1000,
                             overwrite=False,
                             parallel=False):
    if approx is None:
        approx = approximation_types()
    datasets = mb.get_datasets()

    if not parallel:
        for d in datasets:
            for a in approx:
                cutoff_filename = cutoff_table_filename(d, a)
                if not os.path.exists(cutoff_filename) or overwrite:
                    create_cutoff_table(d, a, min_cluster_size)
                else:
                    print(f"cutoff table already exists for {d} and {a}")

        return None

    
    # Create a partial function with fixed arguments
    for approximation in approx:
        process_func = partial(parallel_util_process_dataset, 
                             approximation=approximation,
                             min_cluster_size=min_cluster_size,
                             overwrite=overwrite)
        
        # Create a pool of workers and map the datasets
        with mp.Pool() as pool:
            pool.map(process_func, datasets)

def parallel_util_process_dataset(dataset, approximation, min_cluster_size, overwrite):
    cutoff_filename = cutoff_table_filename(dataset, approximation)
    if not os.path.exists(cutoff_filename) or overwrite:
        create_cutoff_table(dataset, approximation, min_cluster_size)

def merge_cutoff_tables(active_approximations):
    base = "data"
    t = load_all_cutoff_tables()
    t = t[['dataset', 'approximation', 'cluster', 'r0','n']]
    
    # Create new dataframe with r0_data and r0_approx
    data_values = t[t['approximation'] == base][['dataset', 'cluster', 'r0']].rename(columns={'r0': 'r0_true'})
    approx_values = t[(t['approximation'] != base) & (t['approximation'].isin(active_approximations))]
    
    tt = approx_values.merge(data_values, on=['dataset', 'cluster'])
    tt = tt.rename(columns={'r0': 'r0_pred'})
    
    return tt




def create_cutoff_table(dataset,
                        approximation,
                        min_cluster_size=1000):
    outfile = cutoff_table_filename(dataset, approximation)
    cs = mb.load_baseline(dataset)
    a = approximation

    labels = cs.obs["leiden"]

    print(f"approximation {a} for {dataset}")
    if a == "data":
        s = cs
    elif a[0:2] == "dG":
        gmm = mGMM.load_GMM(dataset, a) 
        s = dataset_util.datasetPCA_GMM(cs, gmm, k_nn=20)
    else:
        raise ValueError(f"approximation {a} not found")
    
    print("starting cutoff computation")
    df = dataset_util.cutoffs(s, labels, 
                            type="local",
                            min_cluster_size=min_cluster_size,
                            k_nn=20, debug=True)
    df["approximation"] = a
    df["dataset"] = dataset
   
    # save to csv
    df.to_csv(outfile)
    return df

def load_all_cutoff_tables():
    t = []
    for dataset in mb.get_datasets():
        for approximation in approximation_types():
            t.append(load_cutoff_table(dataset, approximation))
    return pd.concat(t)

def load_cutoff_table(dataset, approximation):
    outfile = cutoff_table_filename(dataset, approximation)
    if not os.path.exists(outfile):
        print(f"cutoff table does not exist for {dataset} and {approximation}, skipping")
        return None
    
    return pd.read_csv(outfile)

