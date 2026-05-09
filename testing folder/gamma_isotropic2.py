import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from scipy import linalg
import util

from sklearn.neighbors import kneighbors_graph

from scipy import linalg
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm

import gamma_cutoff_gaussian as gcg


def sample_X(mu, n1=1000, n2=1000, p=50):    
    I = np.diag(np.ones(p))
    mu_p = np.zeros(p)
    mu_p[0] = mu

    X1 = np.random.multivariate_normal(mean=mu_p, cov=I, size=n1)
    X2 = np.random.multivariate_normal(mean=-mu_p, cov=I, size=n2)
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((np.repeat(1, n1), np.repeat(-1, n2)))
    return X, y

def compute_rho(n, s):
    s0 = s[0]
    s1 = s[1:]
    s1 = np.sort(s1)
    s1 = s1[::-1]
    rho = np.sqrt(np.log(n))/(2*np.sqrt(np.sum(s1**4/s0**4)))
    return rho



def compute_rho_large_n(n, p, mu):
    log2n = np.sqrt(2*np.log(n))
    rho = log2n/(2*np.sqrt(2*(p-1)))*(1 + mu**2/4)
    return rho


def estimate_gamma(s, alpha, n, f, N, Y_approximation):
    if not Y_approximation in ["eigenvector", "chisq2"]:
        raise ValueError("Y_approximation must be one of: eigenvector, chisq2")

    rho = compute_rho(n, s)
    slog2n = np.sqrt(np.log(n))

    probs = np.array([1-f, f])
    outcomes = np.array([-1, 1])

    alpha2 = alpha**2
    var_value = 1 - alpha2

    g = np.zeros(N)
    z_values = np.zeros(N)
    for i in range(N):
        w_sign = np.random.choice(outcomes, size=n, p=probs)
        z_sign = np.random.choice(outcomes, size=1, p=probs)

        G_w = np.random.normal(0, 1, size=(n,2))
        G_z = np.random.normal(0, 1, size=2)
        
        w = (w_sign*alpha + np.sqrt(var_value)*G_w[:,1])
        z = (z_sign*alpha + np.sqrt(var_value)*G_z[1])

        if Y_approximation == "eigenvector":
            Y = gcg.sample_Y(s, N=n)
        elif Y_approximation == "chisq2":
            Y = np.random.normal(0, 1, size=n)


        z_values[i] = z.item()

        d = rho*(w - z)**2 + slog2n*Y
        # = sv**2*(w - z)**2 + noise_total
       
        i_min = np.argmin(d)
        gg = 1.0*(np.sign(w[i_min]) != np.sign(z))
        g[i] = gg.item()

    # I need the fraction in each cluster
    frac = np.mean(z_values > 0)
    ff = 1 - frac**2 - (1-frac)**2

    return np.mean(g)/ff



def estimate_gamma_orderstatistics(rho, alpha, f, N=300):

    #mu_norm = np.sqrt(np.sum(mu**2))
    #spike_rho = mu_norm/2

    #rho = compute_rho(n, p, mu, c, f)
    #_, alpha2 = spike_statistics(spike_rho, c)
    #alpha = np.sqrt(alpha2)
    alpha2 = alpha**2
    var_value = 1 - alpha2
    
    probs = np.array([1-f, f])
    outcomes = np.array([-1, 1])

    min_f = np.min([f, 1-f])
    n = np.max([1000, int(500.0/min_f)])

    g = np.zeros(N)
    z_values = np.zeros(N)
    
    for i in range(N):
        w_sign = np.random.choice(outcomes, size=n, p=probs)
        z_sign = np.random.choice(outcomes, size=1, p=probs)

        G_w = np.random.normal(0, 1, size=(n,2))
        G_z = np.random.normal(0, 1, size=2)
        
        w = (w_sign*alpha + np.sqrt(var_value)*G_w[:,1])
        z = (z_sign*alpha + np.sqrt(var_value)*G_z[1])

        #G_n = np.random.normal(0, 1, size=(n,p-1))
        #noise = np.sqrt(2)*G_n*base_s
        #noise_total = np.sum(noise**2, axis=1)
        # noise_total = noise_total - 2*np.sum(base_s**2)
        # noise_total = noise_total/(2*np.sqrt(2*np.sum(base_s**4)))
        # plt.hist(noise_total, bins=50)
        # plt.show()

        Y = np.random.exponential(1/np.linspace(1,n,n))
        Y = np.cumsum(Y)
        Y = Y[np.random.choice(n, size=n, replace=False)]

        z_values[i] = z.item()

        d = rho*(w - z)**2 + Y
        # = sv**2*(w - z)**2 + noise_total
       
        i_min = np.argmin(d)
        gg = 1.0*(np.sign(w[i_min]) != np.sign(z))
        g[i] = gg.item()

    # I need the fraction in each cluster
    frac = np.mean(z_values > 0)
    ff = 1 - frac**2 - (1-frac)**2

    interp = norm.cdf(alpha/np.sqrt(1 - alpha2 + 1E-10))
    estimated_frac = f*interp + (1-f)*(1-interp)
    estimate_ff = 1 - estimated_frac**2 - (1-estimated_frac)**2
    #print("alpha", alpha, "alpha2", alpha2, "interp", interp)
    #print("estimated_ff", estimate_ff, "true_ff", ff)

    ff = estimate_ff
    print("ff", ff)
    print("g", np.mean(g))
 
    return np.mean(g)/ff

def cutoff_gamma(X, k_nn=10):
    A = form_A(X, k_nn)
    return true_gamma(A)

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


def spike_statistics(lambd, c):
    # see Benayach-Georges and Nadakuditi, equations (8)-(10)
    if lambd <= (c**.25):
        sv = 1 + np.sqrt(c)
        alpha2 = 0
        return sv, alpha2

    l_BGN = (1+lambd**2)*(c+lambd**2)/lambd**2
    l_BGN = np.sqrt(l_BGN)
    #alpha_BGN = 1 - c*(1 + rho**2)/rho**2/(rho**2 + c)
    alpha_BGN = 1 - (c + lambd**2)/(lambd**2)/(1 + lambd**2)

    sv = l_BGN
    alpha2 = alpha_BGN
    return sv, alpha2

