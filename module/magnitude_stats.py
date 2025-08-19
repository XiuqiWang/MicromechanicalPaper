# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 11:54:57 2025

@author: WangX3
"""

import numpy as np

def magnitude_stats(A, B):
    """
    Compute the mean, standard deviation, and standard error of
    R = sqrt(A_mean**2 + B_mean**2)

    Parameters
    ----------
    A : array_like
        Samples of variable A.
    B : array_like
        Samples of variable B.

    Returns
    -------
    R_mean : float
        Mean value of R.
    R_std : float
        Standard deviation of R.
    R_stderr : float
        Standard error of R.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    # Means of A and B
    A_mean = np.mean(A)
    B_mean = np.mean(B)

    # Compute R from means
    R = np.sqrt(A_mean**2 + B_mean**2)

    # Propagate uncertainty using partial derivatives
    n = len(A)
    cov_AB = np.cov(A, B, ddof=1)[0, 1]
    var_R = ((A_mean / R)**2 * np.var(A, ddof=1) +
             (B_mean / R)**2 * np.var(B, ddof=1) +
             2 * (A_mean * B_mean / R**2) * cov_AB)
    
    R_std = np.sqrt(var_R)
    R_stderr = R_std / np.sqrt(n)

    return R, R_std, R_stderr
