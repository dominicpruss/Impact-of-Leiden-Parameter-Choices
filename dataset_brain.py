import scanpy as sc
import pandas as pd
import anndata as ad
import pdb

import numpy as np
import os
import matplotlib.pyplot as plt

import util

# I use cluster_id as "celltype"
# I store the original celltype in "brain_celltype"
def brain(overwrite=False, convert_names=True):
    
    tissue = "brain"
    base_f = "../data/brain atlas - cerebellar vermis/"
    in_f = base_f + tissue + ".h5ad"
    u_count_f = base_f + tissue + ".anndata"

    names_f = base_f + "brain_names.csv"
    
    if not os.path.isfile(u_count_f) or overwrite:
      s = sc.read_h5ad(in_f)

      idx = (s.obs["total_UMIs"] < 10**4.4) & (s.obs["total_UMIs"] > 10**3.2)
      s = s[idx,:]
      
      # make n_genes between 1200 and 2500
      idx = (s.obs["total_genes"] < 10**3.8) & (s.obs["total_genes"] > 10**3)
      s = s[idx,:]
      
      idx = s.obs["fraction_mitochondrial"] < 0.04
      s = s[idx,:]
      
      Xr = s.X
      gene_counts = Xr.sum(axis=0).flatten()
      idx = gene_counts > 50
      s = s[:,idx]

      print("subsampling brain!")
      s = s[np.random.choice(s.shape[0], 30000, 
                             replace=False), :]
      s.obs["celltype"] = [str(s) for s in s.obs["cell_type"]]

      s = util.counts2ann(s.X, 
                            genes=s.var_names, 
                            barcodes=s.obs_names, 
                            min_gene_count=0, 
                            min_cell_count=0,
                            obs=s.obs)
  
      #s.obs["celltype"] = s.obs["cluster_id"].to_numpy()
      #s.obs["brain_celltype"] = s.obs["cell_type"].to_numpy()
      #s.obs["celltype"] = [str(s) for s in s.obs["cell_type"]]
      util.normalize(s)
      print(s.X.sum(axis=1))
      s = util.gene_selection(s, n_genes=1000)
      
      s.write(u_count_f, compression="gzip")  
        
    s = ad.read_h5ad(u_count_f)

    if convert_names:
        ct = s.obs["celltype"].to_numpy()
        d = pd.read_csv(names_f)
        dic = dict(zip(d["celltype"].to_numpy(), 
                       d["celltype_new"].to_numpy()))
        
        new_ct = np.array([dic[c] for c in ct])
        s.obs["celltype"] = new_ct
    
    return s


