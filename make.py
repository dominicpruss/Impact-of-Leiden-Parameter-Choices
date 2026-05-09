import dataset_make
import manuscript_baseline as mb
import manuscript_globalvslocal as mgl
import manuscript_GMM as mg
import manuscript_approximation as ma
import manuscript_single_gaussian as ms
import manuscript_two_gaussians as mtg
import manuscript_joint as mj

#def make_datasets():
#    dataset_make.make()

def make_analysis_files():
    mb.make()
    mgl.make()
    mg.make()
    ma.make()
    ms.make()
    mtg.make()
    mj.make()
