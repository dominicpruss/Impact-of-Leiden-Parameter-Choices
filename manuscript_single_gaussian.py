import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import multivariate_normal as mvn
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
import pickle
import os
from scipy.interpolate import CubicSpline
from matplotlib.colors import ListedColormap

import util
import gamma_cutoff_gaussian as gg
import GMM as gmm
import manuscript_GMM as mGMM
import manuscript_baseline as mb

import pdb
from multiprocessing import Pool



def workfolder():
    # check if "work" folder exists, if not create it
    if not os.path.exists("manuscript_data"):
        os.makedirs("manuscript_data")

    if not os.path.exists("manuscript_data/single_gaussian"):
        os.makedirs("manuscript_data/single_gaussian")
    return "manuscript_data/single_gaussian"

def get_n_p_combinations(N):
    ns = [1000, 5000, 10000]
    ps = [20, 50, 100]

    return [(n, p, N) for n in ns for p in ps]


def get_filename_samples(n, p):
    return os.path.join(workfolder(), 
                        f"X_n{n}_p{p}.pkl")

def get_filename_dataset(dataset):
    return os.path.join(workfolder(), 
                        f"{dataset}.pkl")

def get_filename_gamma_comparison_figure_Methods():
    return os.path.join(workfolder(), 
                        f"gamma_comparison_Methods.pdf")

def get_filename_gamma_comparison_figure():
    return os.path.join(workfolder(), 
                        f"gamma_comparison.pdf")

def get_filename_UMAP_gamma_figure():
    return os.path.join(workfolder(), 
                        f"UMAP_gamma.pdf")

def get_gamma_curve_file():
    return os.path.join(workfolder(), "gamma_curve.csv")

def make():
    # make predictions
    create_gamma_curve(N=10000)

    # generate true values
    combinations =  get_n_p_combinations(N=100)
    
    n_processes = np.min([8, len(combinations)])
    print(f"Using {n_processes} processes")
    with Pool(processes=n_processes) as pool:
        pool.map(make_worker_sample, combinations)

    datasets = mb.get_datasets()
    n_processes = np.min([8, len(datasets)])
    print(f"Using {n_processes} processes")
    with Pool(processes=n_processes) as pool:
        pool.map(make_worker_dataset, datasets)

    plot_gamma_comparisons()
    plot_gamma_comparisons_Methods()
    plot_UMAP_gamma_figure()

def make_worker_sample(args):
        n = args[0]
        p = args[1]
        N = args[2]
        print(f"Processing n={n}, p={p}")
        generate_sample_file(n, p, N=N)
        return 0

def make_worker_dataset(args):
    dataset = args
    print(f"Processing dataset={dataset}")
    generate_dataset_file(dataset)
    return 0

#################################################
def single_gaussian_example_Discussion(n1=1E9, n2 = 1E5, p=50):
    gamma_pred = create_predictor_from_gamma_curve()
    rho1 = np.sqrt(np.log(n1))/(2*np.sqrt(p))
    rho2 = np.sqrt(np.log(n2))/(2*np.sqrt(p))

    gamma1 = gamma_pred(rho1)
    gamma2 = gamma_pred(rho2)

    print(f"gamma1={gamma1}, gamma2={gamma2}")

def plot_UMAP_gamma_figure(n=10000, p=50):
    filename = get_filename_UMAP_gamma_figure()
    # Create custom colormap with red and blue
    colors = ['red', 'blue']
    custom_cmap = ListedColormap(colors)

    # Create figure with four panels
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    
    # Function to create UMAP plot for given d[0] value
    def create_umap_plot(ax, d0, panel_label):
        gamma_pred = create_predictor_from_gamma_curve()
        d = np.repeat(1.0, p)
        d[0] = d0
        Sigma = np.diag(d)
        X = mvn.rvs(np.repeat(0.0, p), Sigma, size=n)

        rho = np.sqrt(np.log(n))*d[0]/(2*np.sqrt(p-1))
        gamma = gamma_pred(rho)
        print(f"d[0]={d0}, rho={rho}, gamma={gamma}")

        s = util.counts2ann(X)
        s.obsm["X_pca"] = X
        util.form_graph(s, k_nn=20)
        sc.tl.umap(s)
        umap_coor = s.obsm["X_umap"]

        dg = 0.05
        g1 = gamma-dg
        g2 = gamma+dg

        util.compute_leiden(s, resolution=g1)
        l1 = s.obs["leiden"]
        util.compute_leiden(s, resolution=g2)
        l2 = s.obs["leiden"]

        # Left panel
        scatter1 = ax[0].scatter(umap_coor[:, 0], umap_coor[:, 1], 
                              c=l1.astype('category').cat.codes,
                              cmap=custom_cmap, alpha=0.8, s=1.3)
        ax[0].set_title(f'γ = {g1:.2f}', fontsize=12)
        ax[0].set_xlabel('UMAP1')
        ax[0].set_ylabel('UMAP2')
        
        # Right panel
        scatter2 = ax[1].scatter(umap_coor[:, 0], umap_coor[:, 1],
                              c=l2.astype('category').cat.codes,
                              cmap=custom_cmap, alpha=0.8, s=1.3)
        ax[1].set_title(f'γ = {g2:.2f}', fontsize=12)
        ax[1].set_xlabel('UMAP1')
        ax[1].set_ylabel('UMAP2')
        
        # Add legend only to right panel
        unique_labels2 = sorted(l2.unique())
        legend2 = ax[1].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor=colors[i % len(colors)], 
                                               markersize=10, label=f'Cluster {label}')
                                    for i, label in enumerate(unique_labels2)],
                            title='Clusters',
                            bbox_to_anchor=(1.05, 1),
                            loc='upper left')
        
        # Add panel labels
        ax[0].text(-0.15, 1.05, panel_label, transform=ax[0].transAxes, 
                   fontsize=24, fontweight='bold')
        ax[1].text(-0.15, 1.05, chr(ord(panel_label) + 1), transform=ax[1].transAxes, 
                   fontsize=24, fontweight='bold')
        
        plt.savefig(filename)
        return fig

    # Create top row (d[0]=1)
    create_umap_plot(axes[0], 1, 'A')
    
    # Create bottom row (d[0]=5)
    create_umap_plot(axes[1], 5, 'C')
    
    plt.tight_layout()
    return fig

def plot_gamma_comparisons_Methods():
    filename = get_filename_gamma_comparison_figure_Methods()
  
    # Load both sample and dataset data
    df_sample = load_all_sample_files()
    df_dataset = load_all_dataset_files()
    df_dataset = df_dataset[df_dataset["n"] > 100]

    gamma_pred = create_predictor_from_gamma_curve()
    df_sample['gamma_order'] = gamma_pred(df_sample['rho'])
    df_dataset['gamma_order'] = gamma_pred(df_dataset['rho'])

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # Define comparisons and their labels
    comparisons = [
        ('gamma_app1', 'Step 1'),
        ('gamma_app2', 'Step 2'),
        ('gamma_app3', 'Step 3'),
        ('gamma_app4', 'Step 4'),
        ('gamma_app5', 'Step 5'),
    ]
    
    # Plot sample data (top row)
    for idx, (gamma_type, title) in enumerate(comparisons):
        ax = axes[0, idx]
        
        # Create scatter plot
        ax.scatter(df_sample['gamma'], df_sample[gamma_type], 
                  color='blue', alpha=0.95, s=10)
        
        # Add diagonal line for reference
        lims = [
           min(df_sample['gamma'].min(), df_sample[gamma_type].min()),
           max(df_sample['gamma'].max(), df_sample[gamma_type].max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        
        # Calculate regression line
        slope, intercept = np.polyfit(df_sample['gamma'], df_sample[gamma_type], 1)
        regression_line = slope * df_sample['gamma'] + intercept
        
        # Add thicker regression line to the plot
        ax.plot(df_sample['gamma'], regression_line, 'r-', 
                alpha=0.7, linewidth=5)
        
        # Customize plot
        ax.set_xlabel('True $\\gamma_s$', fontsize=18)
        ax.set_ylabel(f'Predicted $\\gamma_s$', fontsize=18)
        ax.set_title(f'{title} (Simulations)', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect(0.7)
        
        # Add panel label
        ax.text(-0.15, 1.05, chr(65 + idx), transform=ax.transAxes, 
                fontsize=24, fontweight='bold')
    
    # Plot dataset data (bottom row)
    for idx, (gamma_type, title) in enumerate(comparisons):
        ax = axes[1, idx]
        
        # Create scatter plot
        ax.scatter(df_dataset['gamma'], df_dataset[gamma_type], 
                  color='blue', alpha=0.95, s=10)
        
        # Add diagonal line for reference
        lims = [
           min(df_dataset['gamma'].min(), df_dataset[gamma_type].min()),
           max(df_dataset['gamma'].max(), df_dataset[gamma_type].max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        
        # Calculate regression line
        slope, intercept = np.polyfit(df_dataset['gamma'], df_dataset[gamma_type], 1)
        regression_line = slope * df_dataset['gamma'] + intercept
        
        # Add thicker regression line to the plot
        ax.plot(df_dataset['gamma'], regression_line, 'r-', 
                alpha=0.7, linewidth=5)
        
        # Customize plot
        ax.set_xlabel('True $\\gamma_s$', fontsize=18)
        ax.set_ylabel(f'Predicted $\\gamma_s$', fontsize=18)
        ax.set_title(f'{title} (Datasets)', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect(0.7)
        
        # Add panel label
        ax.text(-0.15, 1.05, chr(70 + idx), transform=ax.transAxes, 
                fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename)
    return fig

def plot_gamma_comparisons():
    filename = get_filename_gamma_comparison_figure()
  
    # Load both sample and dataset data
    df_sample = load_all_sample_files()
    df_dataset = load_all_dataset_files()
    df_dataset = df_dataset[df_dataset["n"] > 100]

    gamma_pred = create_predictor_from_gamma_curve()
    df_sample['gamma_order'] = gamma_pred(df_sample['rho'])
    df_dataset['gamma_order'] = gamma_pred(df_dataset['rho'])
    
    df_theory = load_gamma_curve()

    # Compute global min/max values for consistent axis limits
    x_min = 0
    x_max = 0.8
    y_min = 0
    y_max = 0.8

    # Create figure with constrained layout
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1])
    
    # Plot sample data (top left)
    ax = plt.subplot(gs[0, 0])
    gamma_type = 'gamma_app5'
    title = 'Approximation 5'
    
    # Create scatter plot
    ax.scatter(df_sample['gamma'], df_sample[gamma_type], 
              color='blue', alpha=0.95, s=10)
    
    # Add diagonal line for reference
    ax.plot([0, 0.8], [0, 0.8], 'k--', alpha=0.5)
    
    # Calculate regression line
    slope, intercept = np.polyfit(df_sample['gamma'], df_sample[gamma_type], 1)
    regression_line = slope * df_sample['gamma'] + intercept
    
    # Add thicker regression line to the plot
    ax.plot(df_sample['gamma'], regression_line, 'r-', 
            alpha=0.7, linewidth=5)
    
    # Customize plot
    ax.set_xlabel('True $\\gamma_s$', fontsize=18)
    ax.set_ylabel('Predicted $\\gamma_s$', fontsize=18)
    ax.set_title('Simulations', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect(0.7)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    
    # Add panel label
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, 
            fontsize=24, fontweight='bold')
    
    # Plot dataset data (top right)
    ax = plt.subplot(gs[0, 1])
    
    # Create scatter plot
    ax.scatter(df_dataset['gamma'], df_dataset[gamma_type], 
              color='blue', alpha=0.95, s=10)
    
    # Add diagonal line for reference
    ax.plot([0, 0.8], [0, 0.8], 'k--', alpha=0.5)
    
    # Calculate regression line
    slope, intercept = np.polyfit(df_dataset['gamma'], df_dataset[gamma_type], 1)
    regression_line = slope * df_dataset['gamma'] + intercept
    
    # Add thicker regression line to the plot
    ax.plot(df_dataset['gamma'], regression_line, 'r-', 
            alpha=0.7, linewidth=5)
    
    # Customize plot
    ax.set_xlabel('True $\\gamma_s$', fontsize=18)
    ax.set_ylabel('Predicted $\\gamma_s$', fontsize=18)
    ax.set_title('Datasets', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect(0.7)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    
    # Add panel label
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, 
            fontsize=24, fontweight='bold')
    
    # Plot large n approximation for samples (bottom left)
    ax = plt.subplot(gs[1, 0])
    
    # Plot points for samples
    ax.scatter(df_sample['rho_large_n'], 
              df_sample['gamma'],
              color='green',
              alpha=0.95,
              s=10,
              zorder=1,
              label='True $\\gamma_s$')
    
    # Plot theoretical curve on top (higher z-order)
    ax.scatter(df_theory['rho'], df_theory['gamma'], 
              color='black', alpha=0.7, s=10,
              label='Predicted $\\gamma_s$')
    
    # Customize plot
    ax.set_xlabel('ρ', fontsize=18)
    ax.set_ylabel('$\\gamma_s$', fontsize=18)
    ax.set_title('Simulations', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0, 10))
    ax.set_ylim((y_min, y_max))
    ax.legend(fontsize=14, markerscale=1.625)
    
    # Add panel label
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, 
            fontsize=24, fontweight='bold')
    
    # Plot large n approximation for datasets (bottom right)
    ax = plt.subplot(gs[1, 1])
    
    # Plot points for datasets
    ax.scatter(df_dataset['rho'], 
              df_dataset['gamma'],
              color='green',
              alpha=0.95,
              s=10,
              zorder=1,
              label='True $\\gamma_s$')
    
    # Plot theoretical curve on top (higher z-order)
    ax.scatter(df_theory['rho'], df_theory['gamma'], 
              color='black', alpha=0.7, s=10,
              label='Predicted $\\gamma_s$',
              zorder=2)
    
    # Customize plot
    ax.set_xlabel('ρ', fontsize=18)
    ax.set_ylabel('$\\gamma_s$', fontsize=18)
    ax.set_title('Datasets', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0, 10))
    ax.set_ylim((y_min, y_max))
    ax.legend(fontsize=14, markerscale=1.625)
    
    plt.savefig(filename)
    return fig




#################################################
# functions for simulated/sampled Gaussians


def load_all_sample_files():
    combinations = get_n_p_combinations(0)
    df = []
    for np in combinations:
        n = np[0]
        p = np[1]
        df.append(load_sample_file(n, p, return_df=True))
    return pd.concat(df)

def load_sample_file(n, p, return_df=True):
    filename = get_filename_samples(n, p)
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return None
    with open(filename, 'rb') as f:
        samples = pickle.load(f)
    if return_df:
        df = pd.DataFrame(samples)
        # Remove the 's' column
        df = df.drop('s', axis=1)
        return df
    else:
        return samples
    


def generate_sample_file(n, p, N=100):
    d0 = np.linspace(5, 100, N)
    filename = get_filename_samples(n, p)
    if os.path.exists(filename):
        print(f"File {filename} already exists")
        return 0
    
    samples = []
    for i in range(N):
        print(f"generating sample {i+1} of {N}")
        d = np.random.uniform(1, 5, size=p)
        d[0] = d0[i]
        Sigma = np.diag(d)
        if n < 5000:
            info = compute_gamma_information(n, p, Sigma, include_approximations=True)
        else:
            info = compute_gamma_information(n, p, Sigma, include_approximations=False)
        samples.append(info)
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)
   



#################################################
def load_all_dataset_files():
    datasets = mb.get_datasets()
    df = []
    for dataset in datasets:
        df.append(load_dataset_file(dataset))
    return pd.concat(df)    

def load_dataset_file(dataset, return_df=True):
    filename = get_filename_dataset(dataset)
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return None
    with open(filename, 'rb') as f:
        df = pickle.load(f)
    if return_df:
        df = pd.DataFrame(df)
        # Remove the 's' column
        df = df.drop('s', axis=1)
       
    return df

# functions for computing true gamma for the dataset gaussians
def generate_all_dataset_files():
    datasets = mb.get_datasets()
    
    n_processes = len(datasets)
    print(f"Using {n_processes} processes")
    with Pool(processes=n_processes) as pool:
        pool.map(generate_dataset_file, datasets)

# def investigate_dataset(dataset, cluster, mixture):
#     g = mGMM.load_GMM(dataset, "dG8_full")
#     gm = gmm.get_mixture_information(g, cluster, mixture)
#     mu = gm["mean"]
#     sigma = gm["cov"]
#     # compute the eigenvalues of sigma
#     d = np.sqrt(np.linalg.eigvals(sigma))
#     n = gm["n"]
#     X = mvn.rvs(0*mu, sigma, size=n)
#     p = X.shape[1]

#     return n, p, np.sort(d), X



def generate_dataset_file(dataset):
    filename = get_filename_dataset(dataset)
    if os.path.exists(filename):
        print(f"File {filename} already exists")
        return 0
    
    gamma_predictor = create_predictor_from_gamma_curve()
    
    out = []
    g = mGMM.load_GMM(dataset, "dG8_full")
    clusters = gmm.get_clusters(g)
    for cluster in clusters:
        c_info = gmm.get_cluster_information(g, cluster)
        mixtures = c_info["mixtures"]
        for mixture in mixtures:
            print(f"cluster {cluster}, mixture {mixture}")
            cm_info = gmm.get_mixture_information(g, 
                                                  cluster, 
                                                  mixture)
            mu = cm_info["mean"]
            sigma = cm_info["cov"]
            n = cm_info["n"]
            p = sigma.shape[0]
            if n < 100:
                continue

            d = compute_gamma_information(n, p, sigma, include_approximations=True)
            print(d)
            out.append(d)

    df = pd.DataFrame(out)
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
    return df
 
#################################################
# functions for producing the predicted gamma based on order statistics
def create_predictor_from_gamma_curve():
    df = load_gamma_curve()
    rho = df['rho'].values
    gamma = df['gamma'].values
    
    spline = CubicSpline(rho, gamma)
    return spline

def load_gamma_curve():
    df = pd.read_csv(get_gamma_curve_file(), header=0)
    return df

def create_gamma_curve(N=10000):
    filename = get_gamma_curve_file()
    if os.path.exists(filename):
        print(f"File {filename} already exists")
        return None
    rho = np.arange(0, 20, .01)
    gamma = np.zeros(len(rho))
    
    # Unpack results into gamma array
    for i in range(len(rho)):
        print(f"rho={rho[i]}")
        gamma[i] = gg.estimate_gamma_orderstatistics(rho[i], N=N)

    df = pd.DataFrame({"rho": rho, "gamma": gamma})
    df.to_csv(filename, header=True, index=False)
    return df


#################################################
# utility functions 

# d is the diagonal of the covariance matrix that generated X
# approximations
# app1 : assume left singular vectors of X are normal
# app2 : replace chi2 with a normal
# app3 : use order statistics with rho computed using true singular values
# app4 : use order statistics with rho computed using sample singular values
# app5 : use order statistics with rho computed using large n limit sv's.
def compute_gamma_information(n, p, Sigma, include_approximations=False):
    X = mvn.rvs(0.0*np.zeros(p), Sigma, size=n)
    X2 = mvn.rvs(0.0*np.zeros(p), Sigma, size=n)

    gamma_pred = create_predictor_from_gamma_curve()
    A = form_A(X, k_nn=10)

    _,s, _ = linalg.svd(X, full_matrices=False)
    _,s_sample, _ = linalg.svd(X2, full_matrices=False)
    rho = gg.get_rho(n, s)
    rho_sample = gg.get_rho(n, s_sample)

    rho_large_n = gg.rho_large_n(n, Sigma)
   
    gamma_app3 = gamma_pred(rho).item()
    gamma_app4 = gamma_pred(rho_sample).item()
    gamma_app5 = gamma_pred(rho_large_n).item()
    if include_approximations:
        print("computing gamma approximation 1")
        gamma_app1 = gg.estimate_gamma(s, n, N=1000, Y_approximation="eigenvector")
        print("computing gamma approximation 2")
        gamma_app2 = gg.estimate_gamma(s, n, N=5000, Y_approximation="chisq2")
    else:
        gamma_app1 = None
        gamma_app2 = None

    print("computing true gamma")
    gamma = true_gamma(A)

    info = {'n':n,
              'p':p,
              's':s,
              'rho_large_n':rho_large_n,
              'rho_sample':rho_sample,
              'rho':rho,
              'gamma_app1':gamma_app1,
              'gamma_app2':gamma_app2,
              'gamma_app3':gamma_app3,
              'gamma_app4':gamma_app4,
              'gamma_app5':gamma_app5,
              'gamma':gamma}
    
    return info

def sample_X(n, p, d=None):
    if d is None:
        d = np.repeat(1, p)
    return np.random.multivariate_normal(mean=np.zeros(p), 
                                  cov=np.diag(d), size=n)


def form_A(X, k_nn=10):
    # Create k-nearest neighbors graph
    print("computing graph")
    A = kneighbors_graph(X, n_neighbors=k_nn, mode='connectivity')

    return A

def true_gamma(A, tol=1E-2, verbose=False):
    gammaL = 0
    gammaR = 1
    
    # Binary search for minimum gamma that gives non-trivial clustering
    while gammaR - gammaL > tol:
        gamma_mid = (gammaL + gammaR) / 2
        labels = util.leiden(A, gamma_mid)
        if verbose:
            nl = len(np.unique(labels))
            print(f"gL {gammaL}, gM {gamma_mid}, gR {gammaR}, labels: {nl}")
        
        # Check if clustering is non-trivial (more than one unique value)
        if len(np.unique(labels)) > 1:
            gammaR = gamma_mid
        else:
            gammaL = gamma_mid
            
    return (gammaL + gammaR)/2  # Return the smallest gamma that gives non-trivial clustering

