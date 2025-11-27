#
# Assessment 1 - setup for Part A
#
# Support code for: Stochastic Approximation & Bayesian Modelling in Practice
#
from scipy import stats

from lab_03_setup import *


def compute_posterior(PHI, y, alpha, sigma_2):
    M = PHI.shape[1]
    beta = 1 / sigma_2
    H = beta * (PHI.T @ PHI) + alpha * np.eye(M)
    SIGMA = np.linalg.inv(H)  # Assumes H is invertible!
    Mu = beta * (SIGMA @ (PHI.T @ y))
    #
    return Mu, SIGMA


def compute_log_marginal_scipy(PHI, y, alph, sigma_2):
    # This should work (mostly...),
    # but is not ideal in the common case where N > M
    #
    N, M = PHI.shape
    #
    # Compute the NxN covariance
    #
    C = sigma_2 * np.eye(N) + (PHI @ PHI.T) / alph
    #
    # Call scipy
    #
    logML = stats.multivariate_normal.logpdf(y.T, mean=None, cov=C, allow_singular=True)
    #
    return logML


def compute_log_marginal_woodbury(PHI, y, alph, sigma_2):
    #
    # Exploit the shape of C and the fact that M << N (usually)
    # - computation scales with O(M^3), not O(N^3)
    #
    N, M = PHI.shape
    beta = 1 / sigma_2
    Mu, SIGMA = compute_posterior(PHI, y, alph, sigma_2)
    #
    # Constant factor
    #
    logML = -N*np.log(2*np.pi)
    #
    # log determinant factor (log|C|), calculated using MxM SIGMA
    #
    (sgn, log_det) = np.linalg.slogdet(SIGMA)
    #
    if sgn < 0:
        # This usually means that SIGMA is not positive-definite
        raise np.linalg.LinAlgError("log_det sign is negative - something is wrong!")
    #
    # This follows from matrix determinant identities
    #
    logML += log_det + N*np.log(beta) + M*np.log(alph)
    #
    # data term (t'Ct) - this is derived from applying the Woodbury identity
    # and then doing some rearranging and substitution
    #
    logML -= beta * (y.T @ (y - PHI @ Mu))
    #
    logML = logML[0, 0] / 2.0
    #
    return logML
