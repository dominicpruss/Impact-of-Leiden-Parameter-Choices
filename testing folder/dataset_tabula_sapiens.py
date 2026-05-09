import scanpy as sc
import pandas as pd
import anndata as ad
import pdb

import numpy as np
import os

import util



def tabula_sapiens_blood(overwrite=False,
                         convert_names=True):
    
    tissue = "blood"
    base_f = "../data/tabula_sapiens/"
    in_f = base_f + tissue + ".h5ad"
    u_count_f = base_f + tissue + ".anndata"

    names_f = base_f + tissue + "_names.csv"
    
    if not os.path.isfile(u_count_f) or overwrite:
      s = sc.read_h5ad(in_f)

      idx = (s.obs["total_counts"] < 10**4.5)
      s = s[idx,:]
      # make n_genes between 1200 and 2500
      idx = (s.obs["n_genes_by_counts"] < 10**3.85) & (s.obs["n_genes_by_counts"] > 10**2.3)
      s = s[idx,:]

      idx = s.obs["pct_counts_mt"] < 14
      s = s[idx,:]

      Xr = s.X
      gene_counts = Xr.sum(axis=0).flatten()
      idx = gene_counts > 50
      s = s[:,idx]
        
      print("subsampling blood!")
      s = s[np.random.choice(s.shape[0], 30000, 
                             replace=False), :]
      s.obs["celltype"] = [str(s) for s in s.obs["cell_type"]]

      s = util.counts2ann(s.X, 
                            genes=s.var_names, 
                            barcodes=s.obs_names, 
                            min_gene_count=0, 
                            min_cell_count=0,
                            obs=s.obs)
      s.obs["celltype"] = s.obs["cell_type"].to_numpy()
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



def tabula_sapiens_heart(overwrite=False,
                         convert_names=True):
    
    tissue = "heart"
    base_f = "../data/tabula_sapiens/"
    in_f = base_f + tissue + ".h5ad"
    u_count_f = base_f + tissue + ".anndata"

    names_f = base_f + tissue + "_names.csv"
    
    if not os.path.isfile(u_count_f) or overwrite:
      s = sc.read_h5ad(in_f)

      idx = (s.obs["total_counts"] < 10**4.9)
      s = s[idx,:]
      
      idx = (s.obs["n_genes_by_counts"] < 10**3.8) & (s.obs["n_genes_by_counts"] > 10**3)
      s = s[idx,:]

      idx = s.obs["pct_counts_mt"] < 20
      s = s[idx,:]

      Xr = s.X
      gene_counts = Xr.sum(axis=0).flatten()
      idx = gene_counts > 50
      s = s[:,idx]
        
      s.obs["celltype"] = [str(s) for s in s.obs["cell_type"]]

      s = util.counts2ann(s.X, 
                            genes=s.var_names, 
                            barcodes=s.obs_names, 
                            min_gene_count=0, 
                            min_cell_count=0,
                            obs=s.obs)
      s.obs["celltype"] = s.obs["cell_type"].to_numpy()
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



def tabula_sapiens_eye(overwrite=False,
                         convert_names=True):
    
    tissue = "eye"
    base_f = "../data/tabula_sapiens/"
    in_f = base_f + tissue + ".h5ad"
    u_count_f = base_f + tissue + ".anndata"

    names_f = base_f + tissue + "_names.csv"
    
    if not os.path.isfile(u_count_f) or overwrite:
      s = sc.read_h5ad(in_f)
      
      idx = (s.obs["total_counts"] < 10**4.9)
      s = s[idx,:]

      # make n_genes between 1200 and 2500
      idx = (s.obs["n_genes_by_counts"] < 10**3.8) & (s.obs["n_genes_by_counts"] > 10**2.6)
      s = s[idx,:]


      idx = s.obs["pct_counts_mt"] < 17
      s = s[idx,:]

      Xr = s.X
      gene_counts = Xr.sum(axis=0).flatten()
      idx = gene_counts > 50
      s = s[:,idx]
        
      s.obs["celltype"] = [str(s) for s in s.obs["cell_type"]]

      s = util.counts2ann(s.X, 
                            genes=s.var_names, 
                            barcodes=s.obs_names, 
                            min_gene_count=0, 
                            min_cell_count=0,
                            obs=s.obs)
     
      s.obs["celltype"] = s.obs["cell_type"].to_numpy()
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




def tabula_sapiens_tongue(overwrite=False,
                         convert_names=False):
    
    tissue = "tongue"
    base_f = "../data/tabula_sapiens/"
    in_f = base_f + tissue + ".h5ad"
    u_count_f = base_f + tissue + ".anndata"

    names_f = base_f + tissue + "_names.csv"
    
    if not os.path.isfile(u_count_f) or overwrite:
      s = sc.read_h5ad(in_f)
      
      idx = (s.obs["total_counts"] < 10**4.9)
      s = s[idx,:]

      # make n_genes between 1200 and 2500
      idx = (s.obs["n_genes_by_counts"] < 10**4) & (s.obs["n_genes_by_counts"] > 10**2.9)
      s = s[idx,:]


      idx = s.obs["pct_counts_mt"] < 17
      s = s[idx,:]

      Xr = s.X
      gene_counts = Xr.sum(axis=0).flatten()
      idx = gene_counts > 50
      s = s[:,idx]
        
      s.obs["celltype"] = [str(s) for s in s.obs["cell_type"]]

      s = util.counts2ann(s.X, 
                            genes=s.var_names, 
                            barcodes=s.obs_names, 
                            min_gene_count=0, 
                            min_cell_count=0,
                            obs=s.obs)
     
      s.obs["celltype"] = s.obs["cell_type"].to_numpy()
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

