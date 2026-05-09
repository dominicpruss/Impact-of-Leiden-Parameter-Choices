import dataset_util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pdb
import os


import manuscript_baseline as mb

def globalvslocal_table_filename(dataset, type):
    if not type in ["LRI", "split"]:
        raise ValueError("type must be either 'LRI' or 'split'")

    dir = mb.workfolder() + "/globalvslocal"
    if not os.path.exists(dir):
        os.makedirs(dir)

    outfile = dir + "/" + dataset + "_" + type + "_globalvslocal.csv"
    return outfile

def globalvslocal_figure_filename():
    dir = mb.workfolder() + "/globalvslocal"
    if not os.path.exists(dir):
        os.makedirs(dir)

    outfile = dir + "/globalvslocal.pdf"
    return outfile

def make():
    create_all_gl_tables()
    create_gl_figure()

def create_gl_figure(save_path='globalvslocal.pdf',
                            dist_to_diagonal_threshold=np.inf):
    save_path = globalvslocal_figure_filename()
    df = load_gl_tables()
    
    true_cutoff = 3
    predicted_cutoff = 3
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(16, 10))
    
    # Create gridspec with no space between panels
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.1, wspace=0.1,
                         left=0.1, right=0.9, bottom=0.1, top=0.9)
    
    # Create axes for each panel
    ax1 = fig.add_subplot(gs[0, 0])  # Top left - split scatter
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left - split vertical lines
    ax3 = fig.add_subplot(gs[0, 1])  # Top right - LRI scatter
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom right - LRI vertical lines
    
    datasets = mb.get_datasets()
    # Define 8 distinct bright colors
    tableau8 = [
        (31, 119, 180),   # blue
        (255, 127, 14),   # orange
        (44, 160, 44),    # green
        (214, 39, 40),    # red
        (148, 103, 189),  # purple
        (140, 86, 75),    # brown
        (227, 119, 194),  # pink
        (23, 190, 207)    # cyan
    ]
    # Scale RGB values to [0, 1] range
    tableau8 = [(r/255., g/255., b/255.) for r, g, b in tableau8]
    
    colors = {dataset: tableau8[i] for i, dataset in enumerate(datasets)}
    
    # Function to create scatter plot
    def create_scatter_plot(ax, df_type, title):
        df_subset = df[df["type"] == df_type].copy()
        df_subset["predicted"] = df_subset["pred_global_r0"]
        df_subset["true"] = df_subset["global_r0"]
        df_subset["f"] = df_subset["cluster_frequency"]
        
        all_predicted = []
        all_true = []
        for dataset, color in colors.items():
            mask = (df_subset['dataset'] == dataset) & (df_subset['predicted'] < predicted_cutoff) & (df_subset['true'] < true_cutoff)
            data = df_subset[mask]
            
            # Collect all points for regression
            all_predicted.extend(data['predicted'])
            all_true.extend(data['true'])
            
            # Create scatter plot with PBMC in all caps
            label = dataset.upper() if dataset == "pbmc" else dataset.capitalize()
            ax.scatter(data['true'], data['predicted'], 
                      color=color, 
                      alpha=0.8, 
                      s=64,
                      label=label)
            
            # Add cluster labels only for points far from diagonal
            for _, row in data.iterrows():
                dist_to_diagonal = abs(row['true'] - row['predicted']) / np.sqrt(2)
                if dist_to_diagonal > dist_to_diagonal_threshold:
                    ax.text(row['true'], row['predicted'] + 0.05,
                            row['cluster'],
                            color=color,
                            fontsize=14,
                            ha='center',
                            va='bottom')
        
        # Add diagonal line
        ax.plot([0, true_cutoff], [0, predicted_cutoff], 'k:', alpha=0.5)
        
        # Add regression line and R-squared
        all_predicted = np.array(all_predicted)
        all_true = np.array(all_true)
        
        # Calculate regression line
        slope, intercept = np.polyfit(all_true, all_predicted, 1)
        reg_line = slope * np.array([0, true_cutoff]) + intercept
        
        # Calculate R-squared
        y_pred = slope * all_true + intercept
        r_squared = np.corrcoef(all_true, all_predicted)[0,1]**2
        
        # Plot regression line
        ax.plot([0, true_cutoff], reg_line, 'k-', alpha=0.5)
        
        # Add R-squared text
        ax.text(0.95, 0.05, f'R² = {r_squared:.3f}', 
                transform=ax.transAxes,
                fontsize=12,
                ha='right',
                va='bottom')
        
        # Customize panel
        ax.set_xlabel('')  # Remove x-label from top panel
        ax.set_ylabel('Predicted $\\gamma_s$')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, markerscale=1)
        ax.set_aspect('equal')
        ax.set_xlim(-0.1, true_cutoff + 0.1)
        ax.set_ylim(-0.1, predicted_cutoff + 0.1)
    
    # Function to create vertical lines plot
    def create_vertical_lines(ax, df_type):
        df_subset = df[df["type"] == df_type].copy()
        df_subset["predicted"] = df_subset["pred_global_r0"]
        df_subset["true"] = df_subset["global_r0"]
        df_subset["f"] = df_subset["cluster_frequency"]
        
        max_f = 0
        for dataset, color in colors.items():
            mask = (df_subset['dataset'] == dataset) & (df_subset['predicted'] < predicted_cutoff)
            data = df_subset[mask]
            max_f = max(max_f, data['f'].max())
            
            # Plot vertical lines from 0 to f for each point
            for _, row in data.iterrows():
                ax.plot([row['true'], row['true']], [0, row['f']],
                        color=color, alpha=0.6, linewidth=2)
        
        # Customize panel
        ax.set_xlabel('True $\\gamma_s$')
        ax.set_ylabel('Cluster frequency')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, true_cutoff + 0.1)
        ax.set_ylim(0, max_f * 1.1)
    
    # Create all four panels
    create_scatter_plot(ax1, "split", "Split")
    create_vertical_lines(ax2, "split")
    create_scatter_plot(ax3, "LRI", "Rand Index")
    create_vertical_lines(ax4, "LRI")
    
    # Add panel labels
    ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes, size=14, weight='bold')
    ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes, size=14, weight='bold')
    ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes, size=14, weight='bold')
    ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes, size=14, weight='bold')
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to free memory
    
    return None

##### table creation
def check():
    df = load_gl_tables()
    df1 = df[df["type"] == "LRI"]
    df2 = df[df["type"] == "split"]

    # order both dataframes by dataset, cluster
    df1 = df1.sort_values(by=["dataset", "cluster"])
    df2 = df2.sort_values(by=["dataset", "cluster"])

    # check that cluster_frequency is the same for both dataframes by plotting
    plt.scatter(df1["cluster_frequency"], df2["cluster_frequency"])
    plt.show()

def create_all_gl_tables():
    for d in mb.get_datasets():
        print(f"Creating global vs local table for {d}")
        create_gl_table(d, type="LRI", LRI_threshold=0.8)
        create_gl_table(d, type="split", LRI_threshold=(1 - 1E-6))

# The global vs local table is a dataframe comparing the local and global resolutions
# of the baseline clustering for a dataset.
def create_gl_table(dataset, type,
                    min_cluster_size=500,
                    LRI_threshold=0.8,
                    tol=0.005,
                    overwrite=False):
    save_path = globalvslocal_table_filename(dataset, type)
    if os.path.exists(save_path) and not overwrite:
        print(f"Table already exists for {dataset}")
        return
    
    print("creating table " + save_path)
    print(f"Loading {dataset} dataset")
    s = mb.load_baseline(dataset)

    print(f"Computing {dataset} global cutoffs")
    dfg = dataset_util.cutoffs(s, s.obs["leiden"], type="global",
                                min_cluster_size=min_cluster_size,
                                LRI_threshold=LRI_threshold, 
                                tol=tol)
    print(f"Computing {dataset} local cutoffs")
    dfl = dataset_util.cutoffs(s, s.obs["leiden"], type="local",
                                min_cluster_size=min_cluster_size,
                                LRI_threshold=LRI_threshold,
                                tol=tol)
    clusters = np.array([str(s) for s in dfg["cluster"]])
    cd = {"dataset": np.repeat(dataset, len(clusters)), 
                'cluster': clusters,
                "local_r0": dfl["r0"].to_numpy(),
                "global_r0": dfg["r0"].to_numpy(),
                "cluster_frequency": dfl["cluster_frequency"].to_numpy(),
                "n_partitions": dfl["n_partitions"].to_numpy(),
                "LRI": dfl["LRI"].to_numpy(),
                "min_freq": dfl["min_freq"].to_numpy()}
    
    df = pd.DataFrame(cd)
    df.to_csv(save_path, index=False)
    return df

def load_gl_tables():
    df_all = []
    for d in mb.get_datasets():
        for type in ["LRI", "split"]:
            df = load_gl_table(d, type)
            df_all.append(df)
      
    df_joint = pd.concat(df_all, axis=0, ignore_index=True)
    return df_joint

def load_gl_table(dataset, type):
    cpath = globalvslocal_table_filename(dataset, type)

    if os.path.exists(cpath):
        #print(f"Loading {dataset} table")
        df = pd.read_csv(cpath)
        df["pred_global_r0"] = df["local_r0"]/df["cluster_frequency"]
        df["type"] = type
        return df
    else:
        print(f"No table found for {dataset}")
        return None



