# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 10:50:44 2025

@author: WangX3
"""

import numpy as np

def ratio_stats(A, B, degrees=False):
    """
    Compute mean, standard deviation, and standard error for R = A_mean / B_mean.
    
    Parameters
    ----------
    A, B : array-like
        Input data arrays (must have same length).
    degrees : bool, optional
        If True, returns results multiplied by 180/pi (mainly useful if this ratio
        represents a tangent or similar angular quantity).
    
    Returns
    -------
    R_mean, R_std, R_se : float
        Mean, standard deviation, and standard error of the ratio.
    """
    A = np.asarray(A)
    B = np.asarray(B)

    # Means
    A_mean = np.mean(A)
    B_mean = np.mean(B)

    # Standard deviations with sample correction (ddof=1)
    A_std = np.std(A, ddof=1)
    B_std = np.std(B, ddof=1)

    # Ratio mean
    R_mean = A_mean / B_mean

    # Error propagation for std
    dR_dA = 1 / B_mean
    dR_dB = -A_mean / (B_mean**2)
    R_std = np.sqrt((dR_dA * A_std)**2 + (dR_dB * B_std)**2)

    # Standard error of the ratio
    R_se = R_std / np.sqrt(len(A))
    R_se = np.array(R_se, dtype=float)
    R_se[R_se == 0] = np.nan

    return R_mean, R_std, R_se
