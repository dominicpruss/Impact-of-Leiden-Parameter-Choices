import scanpy as sc
import pandas as pd
import anndata as ad
import pdb

from scipy.stats import randint

import numpy as np
import os

import util


def Zheng(overwrite=False,
          convert_names=False):
    
    base_f = "../data/Zheng_dataset/"
    u_count_f = base_f + "Zheng.anndata"
    
    metadata_f = base_f + '68k_pbmc_barcodes_annotation.tsv'
    names_f = base_f + "Zheng_names.csv"
  
    if not os.path.isfile(u_count_f) or overwrite:
      m = pd.read_csv(metadata_f, header=0, sep="\t")
      # X is initially genes by cells
      s = sc.read_10x_mtx(base_f)
      s.obs["cell_type"] = m['celltype'].to_numpy()
      
      mito_gene = s.var_names.str.startswith("MT-")
    
      # Zheng specific processing
      s.obs['count'] = np.sum(s.X, axis=1)
      s.obs['pct_mt'] = (s.X @ (1*mito_gene))/s.obs['count']
      
      pct_mt = s.obs["pct_mt"].to_numpy()
      s = s[pct_mt < 0.05,:]
      
  
      cell_counts = s.obs['count'].to_numpy()
      s = s[(cell_counts > 1000) & (cell_counts < 3000),:]
      gene_counts = np.sum(s.X, axis=0)
      s = s[:,gene_counts > 50]

      print("subsampling Zheng!")
      s = s[np.random.choice(s.shape[0], 30000, 
                             replace=False), :]
      
      s.obs["celltype"] = [str(s) for s in s.obs["cell_type"]]
      s = util.counts2ann(s.X, 
                          genes=s.var_names, 
                          barcodes=s.obs_names, 
                          min_gene_count=50, 
                          min_cell_count=500,
                          obs=s.obs)
      #s.obs["celltype"] = s.obs["cell_type"].to_numpy()
      s = util.normalize(s)
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
      
     
     