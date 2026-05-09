import multiprocessing as mp
import dataset_brain
import dataset_pancreas
import dataset_tabula_sapiens
import dataset_Zheng

import dataset_util

import pandas as pd
import numpy as np

def get_datasets():
    return ["tongue", "eye", "heart", "PBMC", 
            "brain", "pancreas", "blood"]

def get_sizes():
    sizes = {}
    sizes["brain"] = load_brain().shape
    sizes["pancreas"] = load_pancreas().shape
    sizes["tabula_eye"] = load_tabula_eye().shape
    sizes["tabula_heart"] = load_tabula_heart().shape
    sizes["tabula_tongue"] = load_tabula_tongue().shape
    sizes["zheng"] = load_zheng().shape
    return pd.DataFrame(sizes)

def get_library_sizes():
    sizes = {}
    for d in mb.get_datasets():
        s = dataset_util.load(d)
        L = s.X.toarray().sum(axis=1)
        # create a dictionary with the quantiles of L
        sizes[d] = {'mean': L.mean(), 'median': np.median(L), 'q1': np.percentile(L, 25), 'q3': np.percentile(L, 75)}
    
    # convert sizes to a DataFrame
    return pd.DataFrame(sizes)

def load_brain():
    return dataset_brain.brain()

def load_pancreas():
    return dataset_pancreas.pancreas()

def load_tabula_eye():
    return dataset_tabula_sapiens.tabula_sapiens_eye()

def load_tabula_heart():
    return dataset_tabula_sapiens.tabula_sapiens_heart()

def load_tabula_tongue():
    return dataset_tabula_sapiens.tabula_sapiens_tongue()

def load_zheng():
    return dataset_Zheng.Zheng()

def create_brain():
    print("Making brain dataset")
    dataset_brain.brain(overwrite=True)

def create_pancreas():
    print("Making pancreas dataset")
    dataset_pancreas.pancreas(overwrite=True)

def create_tabula_eye():
    print("Making tabula sapiens eye dataset")
    dataset_tabula_sapiens.tabula_sapiens_eye(overwrite=True)

def create_tabula_heart():
    print("Making tabula sapiens heart dataset")
    dataset_tabula_sapiens.tabula_sapiens_heart(overwrite=True)

def create_zheng():
    print("Making Zheng dataset")
    dataset_Zheng.Zheng(overwrite=True)

def create_tabula_tongue():
    print("Making tabula sapiens tongue dataset")
    dataset_tabula_sapiens.tabula_sapiens_tongue(overwrite=True)

def make():
    create_brain()
    create_pancreas()
    create_tabula_eye()
    create_tabula_heart()
    create_tabula_tongue()
    create_zheng()

