import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb
import scipy
from scipy import linalg


# distance ~ rho*(w - z)^2 + Y_scale*Y
#  where rho = Y_scale/(2*sqrt(sigma2))

# def get_statistics(s, n):
#     sm1 = s[1:]
#     sigma2 = 2*np.sum((sm1/s[0])**4)
   
#     logn = np.log(n)
#     Y_scale = np.sqrt(2*logn)
#     rho = Y_scale/(2*np.sqrt(sigma2))
#     return sigma2, Y_scale, rho

def get_rho(n, s):
    sm1 = s[1:]
    sigma2 = np.sum((sm1/s[0])**4)
   
    logn = np.log(n)
    rho = np.sqrt(logn)/(2*np.sqrt(sigma2))
    return rho

# assumes that the singular values converge to the variances 
# d are the eigenvalues of the covariance matrix
def rho_large_n(n, Sigma):
    d = np.linalg.eigvals(Sigma)
    d = np.sort(d)
    d = d[::-1]

    log2n = np.sqrt(np.log(n))
    d1 = d[0]
    d2 = d[1:]
    r = np.sum((d2**2)/(d1**2))
    rho = log2n/(2*np.sqrt(r))
    return rho

# Extract n, sigma2, Y_scale, rho from X
# def get_statistics_from_matrix(X):
#     n = X.shape[0]
#     _, s, _ = linalg.svd(X, full_matrices=False)
#     return get_statistics(s, n)

# def get_Y_coefficients(s):
#     sm1 = s[1:]
#     coef = (sm1/s[0])**2
#     return coef



# s are the singular values of a sample matrix X from a Gaussian distribution
def sample_Y(s, N=100):
    sm1 = s[1:]
    coef = sm1**2
    sigma = np.sqrt(2*np.sum(sm1**4))
    pm1 = len(coef)

    Y = np.zeros(N)
    for i in range(N):
        norms = np.random.normal(0, 1, size=pm1)
        Y_pre = coef*(norms**2 - 1)/sigma
        Y[i] = np.sum(Y_pre)
    
    # if plot:
    #     # Calculate mean and standard deviation for normal approximation
    #     mu = np.mean(Y)
    #     sigma = np.std(Y)
    #     # Create histogram
    #     plt.hist(Y, density=True, alpha=0.7, bins='auto', label='Empirical')
    
    #     # Generate points for normal curve
    #     x = np.linspace(min(Y), max(Y), 100)
    #     normal_curve = scipy.stats.norm.pdf(x, mu, sigma)
        
    #     # Plot normal approximation
    #     plt.plot(x, normal_curve, 'r-', label='Normal Approximation')
    #     plt.legend()
    #     plt.title('Distribution of Y with Normal Approximation')
    #     plt.show()
    return Y


def estimate_gamma(s, n, N, Y_approximation):
    if not Y_approximation in ["eigenvector", "chisq2"]:
        raise ValueError("Y_approximation must be one of: eigenvector, chisq2")

    rho = get_rho(n, s)
    Y_scale = np.sqrt(2*np.log(n))

    g = np.zeros(N)
    for i in range(N):
        w = np.random.normal(0, 1, size=n)
        z = np.random.normal(0, 1, size=1)
        if Y_approximation == "chisq2":
            Y = np.random.normal(0, 1, size=n)
        elif Y_approximation == "eigenvector":
            Y = sample_Y(s, N=n)
        

        #d = (w - z)**2 + 2*np.sqrt(s2)*Y
        #d = l2n/(2*np.sqrt(s2))*(w - z)**2 + l2n*Y
        d = rho*(w - z)**2 + Y_scale*Y
        i_min = np.argmin(d)
        gg = 1.0*(np.sign(w[i_min]) != np.sign(z))
        g[i] = gg.item()

    return 2*np.mean(g)


def estimate_gamma_orderstatistics(rho, n=5000, N=300):

    g = np.zeros(N)
    for i in range(N):
        w = np.random.normal(0, 1, size=n)
        z = np.random.normal(0, 1, size=1)
        
        rates = np.linspace(1, n, n)
        Y = np.random.exponential(1/rates)
        Y = np.cumsum(Y)
        Y = Y[np.random.choice(n, size=n, replace=False)]
    
        d = rho*(w - z)**2 + Y
        i_min = np.argmin(d)
        gg = 1.0*(np.sign(w[i_min]) != np.sign(z))
        g[i] = gg.item()

    return 2*np.mean(g)
