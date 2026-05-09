

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pickle
import pdb
import os

import manuscript_baseline as mb
from scipy import stats
import seaborn as sns

import GMM
import multiprocessing as mp
from functools import partial

def GMM_folder():
    dir = mb.workfolder() + "/" + "GMM"
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def gmm_filename(dataset, approximation):
    return GMM_folder() + "/" + "GMM_" + dataset + "_" + approximation + ".pkl"

def GMM_figure_filename():
    return GMM_folder() + "/" + "GMM_figure.pdf"

def make(overwrite=False):
    # dG6 is the 6 mixture GMM approximation of the datasets PCA
    approximations = get_approximations()
    create_all_GMM(approximations, overwrite=overwrite)


# approximation types
# for the dataset approximation:
# dG#_type where # is the number of mixtures and type is either "full" or "diag",
# "tied"


# I restrict to full because diag and tied do not give good fits for cutoff
def get_approximations():
    a = ["dG" + str(i) + "_full" for i in np.arange(1,11)]
    return a

def parse_approximation(approximation):
    # check if it is a valid approximation
    if not approximation in get_approximations():
        raise ValueError(f"Approximation {approximation} not found")

    # check if "full or "diag is in approximation
    if "full" in approximation:
        covariance_type = "full"
    elif "diag" in approximation:
        covariance_type = "diag"
    elif "tied" in approximation:
        covariance_type = "tied"
    else:
        raise ValueError(f"Approximation {approximation} not found")

    # parse the approximation
    prefix = approximation.split("_")[0]
    if prefix[0] == "d":
        type = "data"
        n_mixtures = int(prefix[2:])
    else:
        raise ValueError(f"Approximation {approximation} not found")
    covariance_type = approximation.split("_")[1]
    return type, n_mixtures, covariance_type  

#################################################
def n_mixtures_selection_table(ic, reduce_value=0):
    if ic not in ["aic", "bic"]:
        raise ValueError(f"ic {ic} not found")
    
    datasets = mb.get_datasets()
    approximations = get_approximations()
    t = []
    for d in datasets:
        for a in approximations:
            gmms_for_dataset = load_GMM(d, a)
            type, n_mixtures, covariance_type = parse_approximation(a)

            if gmms_for_dataset is None:
                continue
            for cluster in gmms_for_dataset.keys():
                aic = gmms_for_dataset[cluster]["aic"]
                bic = gmms_for_dataset[cluster]["bic"]

                n_mixtures = gmms_for_dataset[cluster]["n_mixtures"]
                n_samples = np.sum(gmms_for_dataset[cluster]["index"])
                dct = {"dataset": d, 
                       "type": type, 
                       "approximation": a,
                       "n_mixtures": n_mixtures,
                       "n_samples": n_samples,
                       "covariance_type": covariance_type,
                       "cluster": cluster,
                       "aic": aic, 
                       "bic": bic}
                t.append(dct)
    df = pd.DataFrame(t)

    # Find best n_mixtures for each dataset, type, cluster combination based on AIC
    best_nm = df.groupby(['dataset', 'type', 'cluster'])['aic'].idxmin()  
    best_nm_df = df.loc[best_nm][['dataset', 'type', 'cluster', 'n_mixtures']]
    best_nm_df = best_nm_df.rename(columns={'n_mixtures': 'nm_best'})

    # Merge back with original dataframe
    df = df.merge(best_nm_df, on=['dataset', 'type', 'cluster'])
    
    # Calculate adjusted n_mixtures and filter
    df['adjusted_nm'] = df.apply(lambda x: max(1, x['nm_best'] - reduce_value), axis=1)
    min_bic_df = df[df['n_mixtures'] == df['adjusted_nm']]

    min_bic_df = min_bic_df.drop(columns=["nm_best", "adjusted_nm"])

    return min_bic_df

def GMM_p_table(approximations):
    datasets = mb.get_datasets()
    t = []
    for d in datasets:
        for a in approximations:
            gmms_for_dataset = load_GMM(d, a)
            if gmms_for_dataset is None:
                continue
            for cluster in gmms_for_dataset.keys():
                gmm = gmms_for_dataset[cluster]["GMM"]
                total_samples = np.sum(gmms_for_dataset[cluster]["index"])
                for mixture in gmm.keys():
                    p = gmm[mixture]["p"]
                    dct = {"dataset": d, 
                            "approximation": a, 
                            "cluster": cluster, 
                            "mixture": mixture, 
                            "p": p,
                            "n_samples": p*total_samples}
                    t.append(dct)

    df = pd.DataFrame(t)
    
    # Convert cluster to numeric for sorting, then back to original type
    df['cluster'] = pd.to_numeric(df['cluster'])
    df = df.sort_values(['dataset', 'approximation', 'cluster', 'mixture'])
    df['cluster'] = df['cluster'].astype(str)
    
    return df


def create_all_GMM(approximations, overwrite=False):
    datasets = mb.get_datasets()
    # Reverse nested loops: iterate over approximations first, then datasets
    for approximation in approximations:
        # Create a partial function with fixed approximation and overwrite parameters
        create_GMM_partial = partial(create_GMM, approximation=approximation, overwrite=overwrite)
        
        # Create a pool of workers
        with mp.Pool() as pool:
            # Map the partial function to all datasets
            pool.map(create_GMM_partial, datasets)

# load all datasets GMM for a given approximation
def load_all_GMM(approximation):
    datasets = mb.get_datasets()
    gmms = {}   
    for dataset in datasets:
        gmms[dataset] = load_GMM(dataset, approximation)
        
    return gmms

def create_GMM(dataset, approximation, overwrite=False):
    outfile = gmm_filename(dataset, approximation)
    if os.path.exists(outfile):
        print(f"GMM already exists for {dataset} with {approximation}, skipping")
        return None

    type, n_mixtures, covariance_type = parse_approximation(approximation)
    print(f"Creating GMM for {dataset} with {n_mixtures} mixtures and {covariance_type} covariance")
    s = mb.load_baseline(dataset)
    labels = s.obs["leiden"]

    if type == "data":
        X = s.obsm["X_pca"]
        gmm = GMM.GMM(X, labels, n_mixtures, covariance_type)
    else:
        raise ValueError(f"Approximation {approximation} not found")

    pickle.dump(gmm, open(outfile, 'wb'))
    return gmm

def load_GMM(dataset, approximation):
    outfile = gmm_filename(dataset, approximation)
    if not os.path.exists(outfile):
        print(f"GMM does not exist for {dataset} with {approximation}, skipping")
        return None
    return pickle.load(open(outfile, 'rb'))
