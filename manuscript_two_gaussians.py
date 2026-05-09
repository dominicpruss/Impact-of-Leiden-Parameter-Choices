import numpy as np
import pandas as pd
import scanpy as sc
from scipy import linalg
import matplotlib.pyplot as plt
import pickle
import os
from scipy.interpolate import CubicSpline

import gamma_isotropic2 as gi2
import util
import pdb
from multiprocessing import Pool

def workfolder():
    # check if "work" folder exists, if not create it
    if not os.path.exists("manuscript_data"):
        os.makedirs("manuscript_data")

    if not os.path.exists("manuscript_data/two_gaussians"):
        os.makedirs("manuscript_data/two_gaussians")
    return "manuscript_data/two_gaussians"

def make():
    generate_all_sample_files()
    plot_UMAP_gamma_G2_figure()
    plot_gamma_comparison_figure()
    plot_gamma_comparison_figure_Methods()
    plot_tradeoff_figure()

def get_n_f_p_combinations():
    ns = [1000, 5000, 10000]
    ps = [20, 50, 100]
    fs = [0.5, 0.75, 0.95]

    return [(n, f, p) for n in ns for p in ps for f in fs]


def get_filename_samples(n, f, p):
    return os.path.join(workfolder(), 
                        f"X_n{n}_f{f}_p{p}.csv")


def get_gamma_comparison_figure_filename():
    return os.path.join(workfolder(), "gamma_comparison_G2.pdf")

def get_gamma_comparison_figure_filename_Methods():
    return os.path.join(workfolder(), "gamma_comparison_G2_Methods.pdf")

def get_UMAP_gamma_figure_filename():
    return os.path.join(workfolder(), "UMAP_gamma_G2_figure.pdf")


########################################################
def plot_UMAP_gamma_G2_figure(n=10000, f=0.95, p=50, mu=2.0):
    filename = get_UMAP_gamma_figure_filename()
    n1 = int(n*f)
    n2 = n - n1
    
    print(f"generating sample for mu={mu}")
    X, z = gi2.sample_X(mu, n1=n1, n2=n2, p=p)
    lambd = mu

    s_large_n = np.repeat(1.0, p)
    s1_large_n, alpha2_large_n = gi2.spike_statistics(lambd, 0)
    s_large_n[0] = s1_large_n
    rho_large_n = gi2.compute_rho(n, s_large_n)

    # print rho and gamma
    gamma = gi2.estimate_gamma_orderstatistics(rho_large_n, 
                                                    np.sqrt(alpha2_large_n), 
                                                    f, N=1000)
    
    print(f"rho={rho_large_n}, gamma={gamma}")

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

    # Create custom colormap with red and blue
    from matplotlib.colors import ListedColormap
    colors = ['red', 'blue']
    custom_cmap = ListedColormap(colors)

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Left panel - colored by l1
    scatter1 = ax1.scatter(umap_coor[:, 0], umap_coor[:, 1], 
                          c=l1.astype('category').cat.codes,
                          cmap=custom_cmap, alpha=0.8, s=1.3)
    ax1.set_title(f'γ = {g1:.2f}', fontsize=12)
    ax1.set_xlabel('UMAP1')
    ax1.set_ylabel('UMAP2')
    
    # Right panel - colored by l2
    scatter2 = ax2.scatter(umap_coor[:, 0], umap_coor[:, 1],
                          c=l2.astype('category').cat.codes,
                          cmap=custom_cmap, alpha=0.8, s=1.3)
    ax2.set_title(f'γ = {g2:.2f}', fontsize=12)
    ax2.set_xlabel('UMAP1')
    ax2.set_ylabel('UMAP2')
    
    # Add legend only to right panel
    unique_labels2 = sorted(l2.unique())
    legend2 = ax2.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=colors[i % len(colors)], 
                                           markersize=10, label=f'Cluster {label}')
                                for i, label in enumerate(unique_labels2)],
                        title='Clusters',
                        bbox_to_anchor=(1.05, 1),
                        loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename)
    return fig

def plot_gamma_comparison_figure_Methods():
    filename = get_gamma_comparison_figure_filename_Methods()
    df_sample = load_all_sample_files()
    #df_sample = df_sample[df_sample["p"] == p]
    
    # Create figure with 5 panels
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Define comparisons and their labels
    comparisons = [
        ('gamma_app1', 'Step 1'),
        ('gamma_app2', 'Step 2'),
        ('gamma_app3', 'Step 3'),
        ('gamma_app4', 'Step 4'),
        ('gamma_app5', 'Step 5'),
    ]
    
    # Plot each comparison
    for idx, (gamma_type, title) in enumerate(comparisons):
        ax = axes[idx]
        
        # Filter out None values
        mask = ~df_sample[gamma_type].isna()
        df_plot = df_sample[mask]
        
        # Create scatter plot
        ax.scatter(df_plot['gamma'], df_plot[gamma_type], 
                  color='blue', alpha=0.95, s=10)
        
        # Add diagonal line for reference
        lims = [
           min(df_plot['gamma'].min(), df_plot[gamma_type].min()),
           max(df_plot['gamma'].max(), df_plot[gamma_type].max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        
        # Calculate regression line
        slope, intercept = np.polyfit(df_plot['gamma'], df_plot[gamma_type], 1)
        regression_line = slope * df_plot['gamma'] + intercept
        
        # Add thicker regression line to the plot
        ax.plot(df_plot['gamma'], regression_line, 'r-', 
                alpha=0.7, linewidth=5)
        
        # Customize plot
        ax.set_xlabel('True $\\gamma_s$', fontsize=18)
        ax.set_ylabel('Predicted $\\gamma_s$', fontsize=18)
        ax.set_title(f'{title}', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect(0.7)
        
        # Add panel label
        ax.text(-0.15, 1.05, chr(65 + idx), transform=ax.transAxes, 
                fontsize=24, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename)
    return fig

def plot_gamma_comparison_figure(p=50, n=1000):
    filename = get_gamma_comparison_figure_filename()
    df_sample = load_all_sample_files()
    
    # Get unique f values and sort them
    f_values = sorted(df_sample['f'].unique())
    
    # Create figure with constrained layout
    fig = plt.figure(figsize=(5*len(f_values), 10), constrained_layout=True)
    gs = plt.GridSpec(2, len(f_values), figure=fig, height_ratios=[1.5, 1])
    
    app = "gamma_app5"
    
    # First row: Gamma comparisons - use all data
    x_min = 0
    x_max = 0.8
    y_min = 0
    y_max = 0.8
    
    for j, f in enumerate(f_values):
        ax = plt.subplot(gs[0, j])
        df_subset = df_sample[df_sample['f'] == f]
        
        # Plot scatter points
        ax.scatter(df_subset['gamma'], df_subset[app], 
                  color='blue', alpha=0.6, s=20)
        
        # Add 45-degree line
        ax.plot([0, 0.8], [0, 0.8], 'k--', alpha=0.5)
        
        # Add regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df_subset['gamma'], df_subset[app])
        x_reg = np.array([0, 0.8])
        y_reg = slope * x_reg + intercept
        ax.plot(x_reg, y_reg, 'r-', alpha=0.7, linewidth=5)
        
        # Customize plot
        ax.set_xlabel('True $\\gamma_s$', fontsize=18)
        ax.set_ylabel('Predicted $\\gamma_s$', fontsize=18)
        ax.set_title(f'q = {f:.2f}', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_aspect(0.7)
        
        # Set consistent axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add panel label
        ax.text(-0.15, 1.05, chr(65 + j), transform=ax.transAxes, 
                fontsize=24, fontweight='bold')
    
    # Second row: RMT approximation - filter by n and p
    x_min = 0
    x_max = 4
    y_min = 0
    y_max = 0.8
    
    # Filter data for bottom panel
    df_bottom = df_sample[df_sample["p"] == p]
    df_bottom = df_bottom[df_bottom["n"] == n]
    
    for j, f in enumerate(f_values):
        ax = plt.subplot(gs[1, j])
        df_subset = df_bottom[df_bottom['f'] == f]
        
        # Sort by mu and remove any duplicate mu values
        df_subset = df_subset.sort_values('mu')
        df_subset = df_subset.drop_duplicates(subset=['mu'])
        
        # Plot true gamma as scatter points
        ax.scatter(df_subset['mu'], df_subset["gamma"], 
                 color='green', alpha=0.6, s=20,
                 label='True $\\gamma_s$')
        ax.scatter(df_subset['mu'], df_subset[app], 
                  color='black', alpha=0.6, s=10,
                  label='Predicted $\\gamma_s$')
        
        # Perform local regression
        from statsmodels.nonparametric.smoothers_lowess import lowess
        # Use a smaller frac value for more local smoothing
        smoothed = lowess(df_subset[app], df_subset['mu'], 
                        frac=0.1, it=3)
        
        # Customize plot
        ax.set_xlabel('μ', fontsize=18)
        ax.set_ylabel('$\\gamma_s$', fontsize=18)
        ax.set_title(f'q = {f:.2f}', fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, markerscale=1.625)
        
        # Set consistent axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Add panel label
        ax.text(-0.15, 1.05, chr(68 + j), transform=ax.transAxes, 
                fontsize=24, fontweight='bold')
    
    plt.savefig(filename)
    return fig



########################################################
def generate_all_sample_files():
    combinations =  get_n_f_p_combinations()
    
    n_processes = np.min([8, len(combinations)])
    print(f"Using {n_processes} processes")
    with Pool(processes=n_processes) as pool:
        pool.map(generate_all_sample_files_worker, combinations)

def generate_all_sample_files_worker(args):
        n = args[0]
        f = args[1]
        p = args[2]
        print(f"Processing n={n}, f={f}, p={p}")
        generate_sample_file(n, f, p)
        return 0

def load_all_sample_files():
    combinations =  get_n_f_p_combinations()
    dfs = []
    for n, f, p in combinations:
        dfs.append(load_sample_file(n, f, p))
    return pd.concat(dfs)

def load_sample_file(n, f, p):
    filename = get_filename_samples(n, f, p)
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return None
    
    d = pd.read_csv(filename)
    return d
    

def generate_sample_file(n, f, p, n_mu_samples=101, overwrite=False):
    filename = get_filename_samples(n, f, p)
    if os.path.exists(filename) and not overwrite:
        print(f"File {filename} already exists")
        return 0
    
    #n1 = int(n*f)
    #n2 = n - n1
    mu_all = np.linspace(0, 5, n_mu_samples)
    
    samples = []
    for mu in mu_all:
        info = generate_sample(n, p, f, mu)
        samples.append(info)

    d = pd.DataFrame(samples)   
    d.to_csv(filename, index=False)


def generate_sample(n, p, f, mu):
    n1 = int(n*f)
    n2 = n - n1
    
    print(f"generating sample for mu={mu}")
    X, z = gi2.sample_X(mu, n1=n1, n2=n2, p=p)
    z = z/np.sqrt(np.sum(z**2))
    U, s, _ = linalg.svd(X, full_matrices=False)
    rho = gi2.compute_rho(n, s)
    alpha = np.abs(np.sum(U[:,0]*z))

    lambd = np.abs(mu)
    
    s_RMT = np.repeat(1.0, p)
    s1_RMT, alpha2_RMT = gi2.spike_statistics(lambd, p/n)
    s_RMT[0] = s1_RMT
    rho_RMT = gi2.compute_rho(n, s_RMT)

    s_large_n = np.repeat(1.0, p)
    s1_large_n, alpha2_large_n = gi2.spike_statistics(lambd, 0)
    s_large_n[0] = s1_large_n
    rho_large_n = gi2.compute_rho(n, s_large_n)
    

    if n < 5000:
        gamma_app1 = gi2.estimate_gamma(s, alpha, n, f, N=1000, Y_approximation="eigenvector")
        gamma_app2 = gi2.estimate_gamma(s, alpha, n, f, N=1000, Y_approximation="chisq2")  
    else:
        gamma_app1 = None
        gamma_app2 = None

    gamma_app3 = gi2.estimate_gamma_orderstatistics(rho, alpha, f, N=1000)
    gamma_app4 = gi2.estimate_gamma_orderstatistics(rho_RMT, np.sqrt(alpha2_RMT), f, N=1000)
    gamma_app5 = gi2.estimate_gamma_orderstatistics(rho_large_n, np.sqrt(alpha2_large_n), f, N=1000)
    
    gamma = gi2.cutoff_gamma(X, k_nn=10)
       
   
    info = {'n':n,
              'p':p,
              'f':f,
              'mu':mu,
              'rho_large_n':rho_large_n,
              'rho_RMT':rho_RMT,
              'rho':rho,
              'alpha2':alpha**2,
              'alpha2_RMT':alpha2_RMT,
              'alpha2_large_n':alpha2_large_n,
              'gamma_app1':gamma_app1,
              'gamma_app2':gamma_app2,
              'gamma_app3':gamma_app3,
              'gamma_app4':gamma_app4,
              'gamma_app5':gamma_app5,
              'gamma':gamma}
    return info 
   

#########################################################
# When I try to predict gamma using approximation 3
# for the dataset gaussian mixture, I get a very bad
# prediction.   
# def gather_gaussians(dataset, cluster):
#     out = []
#     cluster = str(cluster)

#     g = mGMM.load_GMM(dataset, "dG8_full")
#     c_info = gmm.get_cluster_information(g, cluster)
#     mixtures = c_info["mixtures"]
#     for mixture in mixtures:
#         print(f"cluster {cluster}, mixture {mixture}")
#         cm_info = gmm.get_mixture_information(g, 
#                                                 cluster, 
#                                                 mixture)
#         mu = cm_info["mean"]
#         sigma = cm_info["cov"]
#         n = cm_info["n"]
#         p = sigma.shape[0]
#         if n < 100:
#             continue

#         c_info = {'dataset':dataset,
#                 'cluster':cluster,
#                 'mixture':mixture,
#                 'mu':mu,
#                 'sigma':sigma,
#                 'n':n,
#                 'p':p}
#         out.append(c_info)
#     return out

# def generate_all_pair_gammas(dataset, cluster):
#     d = gather_gaussians(dataset, cluster)
#     out = []

#     for i in range(len(d)):
#         for j in range(i+1, len(d)):
#             d1 = d[i]
#             d2 = d[j]
#             mu1 = d1["mu"]
#             sigma1 = d1["sigma"]
#             mu2 = d2["mu"]
#             sigma2 = d2["sigma"]
#             n1 = d1["n"]
#             n2 = d2["n"]
#             mix1 = d1["mixture"]
#             mix2 = d2["mixture"]
#             print("processing pair", mix1, mix2)
#             info = generate_gaussian_pair_gammas(mu1, sigma1, n1,
#                                                  mu2, sigma2, n2)
            
#             # Add additional fields to info
#             info['mix1'] = mix1
#             info['mix2'] = mix2
#             info['dataset'] = dataset
#             info['cluster'] = cluster
            
#             out.append(info)
#     df = pd.DataFrame(out)
#     return df



# def generate_gaussian_pair_gammas(mu1, sigma1, n1, mu2, sigma2, n2):
  
#     X1 = np.random.multivariate_normal(mu1, sigma1, n1)
#     X2 = np.random.multivariate_normal(mu2, sigma2, n2)
#     X = np.concatenate([X1, X2], axis=0)

#     A = mSG.form_A(X, k_nn=10)
#     true_gamma = mSG.true_gamma(A)

#     z = np.concatenate((np.repeat(1, n1), np.repeat(-1, n2)))
#     z = z/np.sqrt(np.sum(z**2))
#     U, s, _ = linalg.svd(X, full_matrices=False)
#     rho = gi2.compute_rho(n1+n2, s)
#     alpha = np.abs(np.sum(U[:,0]*z))
#     f = n1/(n1+n2)

#     gamma_app3 = gi2.estimate_gamma_orderstatistics(rho, alpha, f, N=1000)

#     out_info = {'gamma':true_gamma,
#                 'gamma_app4':gamma_app3}
#     return out_info

########################################################


# Used in tradeoff figure
def generate_isotropic2_mu_gamma_curve(plot=True, N=5000):
    isotropic2_mu_gamma_filename = workfolder() + "/isotropic2_mu_gamma_curve.csv"
    if os.path.exists(isotropic2_mu_gamma_filename):
        df = pd.read_csv(isotropic2_mu_gamma_filename)
    else:

        n = 1000
        p = 50
        f = 0.5

        mus = np.linspace(.1, 10, N)  # Changed to 1000 points for better smoothing
        out = []

        for mu in mus:
            print(f"processing mu={mu}")
            lambd = mu
            s_RMT = np.repeat(1.0, p)
            s1_RMT, alpha2_RMT = gi2.spike_statistics(lambd, 0)
            s_RMT[0] = s1_RMT
            rho_RMT = gi2.compute_rho(n, s_RMT)

            gamma_app5 = gi2.estimate_gamma_orderstatistics(rho_RMT, 
                                                            np.sqrt(alpha2_RMT), 
                                                            f, N=1000)
            out.append({'mu':mu, 'gamma':gamma_app5})

        df = pd.DataFrame(out)
        df.to_csv(isotropic2_mu_gamma_filename, index=False)

    if plot:
        plt.plot(df["mu"], df["gamma"])
        plt.show()

    return df
  


       

    