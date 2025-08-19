# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 10:43:27 2025

@author: WangX3
"""

import numpy as np

def atan_ratio_stats(A, B):
    """
    Compute R = arctan(mean(A)/mean(B)) with std and standard error via error propagation.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    nA, nB = len(A), len(B)

    A_mean = np.mean(A)
    B_mean = np.mean(B)

    std_A = np.std(A, ddof=1)
    std_B = np.std(B, ddof=1)

    se_A = std_A / np.sqrt(nA)
    se_B = std_B / np.sqrt(nB)

    x = A_mean / B_mean
    R_mean = np.arctan(x)

    dR_dA = 1 / ((1 + x**2) * B_mean)
    dR_dB = -A_mean / ((1 + x**2) * B_mean**2)

    R_se = np.sqrt((dR_dA**2) * se_A**2 + (dR_dB**2) * se_B**2)
    R_se = np.array(R_se, dtype=float)
    R_se[R_se == 0] = np.nan
    R_std = np.sqrt((dR_dA**2) * std_A**2 + (dR_dB**2) * std_B**2)

    return np.rad2deg(R_mean), np.rad2deg(R_std), np.rad2deg(R_se)
