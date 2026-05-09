import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import multivariate_normal as mvn
from scipy import linalg
import matplotlib.pyplot as plt
import pickle
import os
from scipy.interpolate import CubicSpline
import alphashape

import gamma_isotropic2 as gi
import gamma_cutoff_gaussian as gc
import manuscript_single_gaussian as msG
import manuscript_two_gaussians as mtG
import util
import pdb
from multiprocessing import Pool

def workfolder():
    # check if "work" folder exists, if not create it
    if not os.path.exists("manuscript_data"):
        os.makedirs("manuscript_data")

    if not os.path.exists("manuscript_data/joint"):
        os.makedirs("manuscript_data/joint")
    return "manuscript_data/joint"

def make():
    plot_tradeoff_figure()
    plot_tradeoff_UMAP_figure()


def get_tradeoff_figure_filename():
    return os.path.join(workfolder(), "tradeoff.pdf")


def get_UMAP_tradeoff_figure_filename():
    return os.path.join(workfolder(), "UMAP_tradeoff_figure.pdf")


########################################################
def plot_tradeoff_UMAP_figure(gamma_isotropic=0.3, 
                              gamma_single=0.2):
    # set random seed 
    np.random.seed(123) 
    filename = get_UMAP_tradeoff_figure_filename()
    d = create_joint_table()
    idx = np.argmin(np.abs(d["gamma"] - gamma_isotropic))
    mu = d.iloc[idx]["mu"]
    print("isotropic pair")
    print(d.iloc[idx])

    idx = np.argmin(np.abs(d["gamma"] - gamma_single))
    rho_single = d.iloc[idx]["rho"]
    d_value = d.iloc[idx]["d_value"]
    print("single normal")
    print(d.iloc[idx])


    # create the isotropic pair
    X2, z = gi.sample_X(mu, n1=500, n2=500, p=50)
    true_gamma_isotropic = true_gamma(X2, k_nn=10)
    print(f"isotropic pair true gamma_l={true_gamma_isotropic}, gamma_g={2*true_gamma_isotropic}")

    # create the single normal
    p = 50
    n = 1000
    mu_big = np.repeat(0.0, p)
    mu_big[0] = 3.2*np.sqrt(d_value)
    d = np.repeat(1.0, p)
    d[0] = d_value
    Sigma = np.diag(d)  
    X1 = mvn.rvs(mu_big, Sigma, size=n)
    true_gamma_single = true_gamma(X1, k_nn=10)
    print(f"single normal true gamma_l={true_gamma_single}, gamma_g={2*true_gamma_single}")


    X = np.concatenate([X1, X2], axis=0)
    z = np.concatenate([np.repeat("single", 1000),
                        np.repeat("pair1", 500),
                        np.repeat("pair2", 500)])
    print(X.shape)
    s = util.counts2ann(X)
    s.obsm["X_pca"] = X
    util.form_graph(s, k_nn=20)
    sc.tl.umap(s)
    umap_coor = s.obsm["X_umap"]

    dg = 0.05
    g1 = 2*(true_gamma_single - dg)
    g2 = 2*(true_gamma_single + true_gamma_isotropic)/2
    g3 = 2*(true_gamma_isotropic + dg)

    util.compute_leiden(s, resolution=g1)
    l1 = s.obs["leiden"]
    util.compute_leiden(s, resolution=g2)
    l2 = s.obs["leiden"]
    util.compute_leiden(s, resolution=g3)
    l3 = s.obs["leiden"]

    # Create custom colormap with red and blue
    from matplotlib.colors import ListedColormap
    colors = ['red', 'blue', "green", "brown"]
    custom_cmap = ListedColormap(colors)

    # Define color mapping and unique labels
    unique_z = sorted(np.unique(z))
    color_dict = {'single': 'green', 'pair1': 'red', 'pair2': 'blue'}

    # Create figure with 1x3 panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # First panel - colored by l1
    scatter1 = ax1.scatter(umap_coor[:, 0], umap_coor[:, 1], 
                          c=[color_dict[label] for label in z],
                          alpha=0.8, s=6.0)
    ax1.set_title(f'γ = {g1:.2f}', fontsize=16)
    ax1.set_xlabel('UMAP1')
    ax1.set_ylabel('UMAP2')
    ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes, fontsize=18, fontweight='bold')
    
    # Add cluster boundaries and labels for panel A
    for cluster in sorted(set(l1)):
        mask = l1 == cluster
        cluster_points = umap_coor[mask]
        
        # Only draw boundary if cluster has points
        if len(cluster_points) >= 3:  # Need at least 3 points for a boundary
            # Calculate distances from center to identify outliers
            center = np.mean(cluster_points, axis=0)
            distances = np.sqrt(np.sum((cluster_points - center)**2, axis=1))
            
            # Remove top 20% outliers
            threshold = np.percentile(distances, 80)
            inlier_mask = distances <= threshold
            filtered_points = cluster_points[inlier_mask]
            
            # Create alpha shape (concave hull)
            alpha = 0.3
            hull = alphashape.alphashape(filtered_points, alpha)
            
            # Handle both Polygon and MultiPolygon cases
            if hasattr(hull, 'exterior'):
                # Single polygon case
                boundary = hull.exterior.coords.xy
                ax1.plot(boundary[0], boundary[1], 'k-', linewidth=2, alpha=0.7)
                # Use polygon centroid for label
                centroid = hull.centroid
                ax1.text(centroid.x, centroid.y, f'C{cluster}',
                       horizontalalignment='center', verticalalignment='center',
                       bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.5'),
                       fontweight='bold', fontsize=14)
            else:
                # MultiPolygon case
                for polygon in hull.geoms:
                    boundary = polygon.exterior.coords.xy
                    ax1.plot(boundary[0], boundary[1], 'k-', linewidth=2, alpha=0.7)
                    # Add label at centroid of each polygon
                    centroid = polygon.centroid
                    ax1.text(centroid.x, centroid.y, f'C{cluster}',
                           horizontalalignment='center', verticalalignment='center',
                           bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.5'),
                           fontweight='bold', fontsize=14)
    
    # Second panel - colored by l2
    scatter2 = ax2.scatter(umap_coor[:, 0], umap_coor[:, 1],
                          c=[color_dict[label] for label in z],
                          alpha=0.8, s=6.0)
    ax2.set_title(f'γ = {g2:.2f}', fontsize=16)
    ax2.set_xlabel('UMAP1')
    ax2.set_ylabel('UMAP2')
    ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes, fontsize=18, fontweight='bold')
    
    # Add cluster boundaries and labels for panel B
    for cluster in sorted(set(l2)):
        mask = l2 == cluster
        cluster_points = umap_coor[mask]
        
        # Only draw boundary if cluster has points
        if len(cluster_points) >= 3:  # Need at least 3 points for a boundary
            # Calculate distances from center to identify outliers
            center = np.mean(cluster_points, axis=0)
            distances = np.sqrt(np.sum((cluster_points - center)**2, axis=1))
            
            # Remove top 20% outliers
            threshold = np.percentile(distances, 80)
            inlier_mask = distances <= threshold
            filtered_points = cluster_points[inlier_mask]
            
            # Create alpha shape (concave hull)
            alpha = 0.3
            hull = alphashape.alphashape(filtered_points, alpha)
            
            # Handle both Polygon and MultiPolygon cases
            if hasattr(hull, 'exterior'):
                # Single polygon case
                boundary = hull.exterior.coords.xy
                ax2.plot(boundary[0], boundary[1], 'k-', linewidth=2, alpha=0.7)
                # Use polygon centroid for label
                centroid = hull.centroid
                ax2.text(centroid.x, centroid.y, f'C{cluster}',
                       horizontalalignment='center', verticalalignment='center',
                       bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.5'),
                       fontweight='bold', fontsize=14)
            else:
                # MultiPolygon case
                for polygon in hull.geoms:
                    boundary = polygon.exterior.coords.xy
                    ax2.plot(boundary[0], boundary[1], 'k-', linewidth=2, alpha=0.7)
                    # Add label at centroid of each polygon
                    centroid = polygon.centroid
                    ax2.text(centroid.x, centroid.y, f'C{cluster}',
                           horizontalalignment='center', verticalalignment='center',
                           bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.5'),
                           fontweight='bold', fontsize=14)
    
    # Third panel - colored by l3
    scatter3 = ax3.scatter(umap_coor[:, 0], umap_coor[:, 1],
                          c=[color_dict[label] for label in z],
                          alpha=0.8, s=6.0)
    ax3.set_title(f'γ = {g3:.2f}', fontsize=16)
    ax3.set_xlabel('UMAP1')
    ax3.set_ylabel('UMAP2')
    ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes, fontsize=18, fontweight='bold')
    
    # Add cluster boundaries and labels for panel C
    for cluster in sorted(set(l3)):
        mask = l3 == cluster
        cluster_points = umap_coor[mask]
        
        # Only draw boundary if cluster has points
        if len(cluster_points) >= 3:  # Need at least 3 points for a boundary
            # Calculate distances from center to identify outliers
            center = np.mean(cluster_points, axis=0)
            distances = np.sqrt(np.sum((cluster_points - center)**2, axis=1))
            
            # Remove top 20% outliers
            threshold = np.percentile(distances, 80)
            inlier_mask = distances <= threshold
            filtered_points = cluster_points[inlier_mask]
            
            # Create alpha shape (concave hull)
            alpha = 0.3
            hull = alphashape.alphashape(filtered_points, alpha)
            
            # Handle both Polygon and MultiPolygon cases
            if hasattr(hull, 'exterior'):
                # Single polygon case
                boundary = hull.exterior.coords.xy
                ax3.plot(boundary[0], boundary[1], 'k-', linewidth=2, alpha=0.7)
                # Use polygon centroid for label
                centroid = hull.centroid
                ax3.text(centroid.x, centroid.y, f'C{cluster}',
                       horizontalalignment='center', verticalalignment='center',
                       bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.5'),
                       fontweight='bold', fontsize=14)
            else:
                # MultiPolygon case
                for polygon in hull.geoms:
                    boundary = polygon.exterior.coords.xy
                    ax3.plot(boundary[0], boundary[1], 'k-', linewidth=2, alpha=0.7)
                    # Add label at centroid of each polygon
                    centroid = polygon.centroid
                    ax3.text(centroid.x, centroid.y, f'C{cluster}',
                           horizontalalignment='center', verticalalignment='center',
                           bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', boxstyle='round,pad=0.5'),
                           fontweight='bold', fontsize=14)
    
    # Add legend for colors on the far right
    legend = fig.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                                          markerfacecolor=color_dict[label], 
                                          markersize=13, label=label)
                               for label in unique_z],
                       bbox_to_anchor=(1.01, 0.5),
                       loc='center left',
                       fontsize=13)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    return mu, d_value

    
def plot_tradeoff_figure():
    filename = get_tradeoff_figure_filename()
    df = create_joint_table()
    df = df[df["mu"] <= 2]
  
    # Use LOESS smoothing with smaller fraction for better local resolution
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(df['d_value'], df['mu'], frac=0.075, it=3)  # Reduced from 0.15 to 0.05
    
    # Create interpolation function from smoothed data
    from scipy.interpolate import interp1d
    smooth_func = interp1d(smoothed[:, 0], smoothed[:, 1], 
                           kind='cubic', bounds_error=False, 
                           fill_value='extrapolate') 

    # Create figure with specified size
    plt.figure(figsize=(8, 4.2))  # Width 8 inches, height 70% of default

    # Plot scatter points first, colored by gamma
    scatter = plt.scatter(df["mu"], df["d_value"], c=df["gamma"], cmap='viridis', alpha=0.3, s=10)
    plt.colorbar(scatter, label='γ')
    
    # Fill area below the line
    plt.fill_between(df["mu"], 0, smooth_func(df["mu"]), color='grey', alpha=0.3)
    
    # Plot smoothed line on top with thicker line
    plt.plot(df["mu"], smooth_func(df["mu"]), color='red', linewidth=2)
    
    # Add point at specific point
    plt.plot(1.5, 17, 'o', color='black', markersize=8)
    
    # Add labels
    # labels should be mu and d1
    plt.xlabel('μ', fontsize=14)
    plt.ylabel('$\\sigma^2$', fontsize=14)
    
    # Set axes limits
    plt.ylim(0, 30)
    
    # Save the plot to file
    plt.savefig(filename)
    
    # Return the figure for display
    return plt.gcf()


########################################################
def create_joint_table():
    n = 1000
    p = 50
    logn = np.sqrt(np.log(n))

    d1 = msG.load_gamma_curve()
    # generated with n=1000, p=50, q=0.5
    d2 = mtG.generate_isotropic2_mu_gamma_curve(False)
    # d2 has mu and gamma.  For each gamma value find the closest
    # gamma value in d1 and add the mu value to the table
    out = []
    for i in range(len(d2)):
        mu = d2.iloc[i]["mu"]
        gamma = d2.iloc[i]["gamma"]
        idx = np.argmin(np.abs(d1["gamma"] - gamma))
        gamma_diff = np.abs(d1.iloc[idx]["gamma"] - gamma)
        if gamma_diff < 0.05:
            rho = d1.iloc[idx]["rho"]
            d_value = 2*rho*np.sqrt(p-1)/logn
            out.append({'mu':mu, 'gamma':gamma, 'rho':rho, 'd_value':d_value})
    df = pd.DataFrame(out)
    return df

def true_gamma(X, k_nn=10):
    A = gi.form_A(X, k_nn=k_nn)
    gamma = gi.true_gamma(A)
    return gamma
    
    