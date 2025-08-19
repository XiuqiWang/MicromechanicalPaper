# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 11:37:51 2025

@author: WangX3
"""
from .flux_weighted_mean_speed import flux_weighted_mean_speed
import numpy as np

# derive Tsal-f(Uim)
def BinTsalUim(velocities_im, theta_im, Tsals, mp_sals, vel_bins):
    velocities_im = np.array(velocities_im)
    velocities_imx = np.array(velocities_im)*np.cos(np.deg2rad(np.array(theta_im)))
    velocities_imz = np.array(velocities_im)*np.sin(np.deg2rad(np.array(theta_im)))
    mp_sal = np.array(mp_sals)
    Tsals = np.array(Tsals)
    
    # Allocate Tsals into velocity ranges
    Tsal_mean, Tsal_stderr = [],[]
    Uim_mean = []
    for i in range(len(vel_bins)-1):
        # Find indices of velocities within the current range
        indices = (velocities_im >= vel_bins[i]) & (velocities_im < vel_bins[i + 1])
        idx = np.where(indices)[0]
        idx = np.array(idx)
        if np.any(idx):  # Check if there are elements in this range
            Uimx_in_bin = velocities_imx[indices]
            Uimz_in_bin = velocities_imz[indices]
            weightsim_in_bin = mp_sal[indices]/Tsals[indices]
            Uim_m, uim_se, _, _ = flux_weighted_mean_speed(Uimx_in_bin, Uimz_in_bin, weightsim_in_bin)
            Uim_mean.append(Uim_m)
            Tsal_in_bin = Tsals[indices]
            mp_in_bin = mp_sal[indices]
            Tsal_m, Tsal_se = mass_weighted_mean_se_forT(Tsal_in_bin, mp_in_bin)
            Tsal_mean.append(Tsal_m)
            Tsal_stderr.append(Tsal_se)
        else:
            Uim_mean.append(np.nan)
            Tsal_mean.append(np.nan)
            Tsal_stderr.append(np.nan)
            
    Uim_Tsal = (vel_bins[:-1] + vel_bins[1:]) / 2 

    return np.array(Tsal_mean), np.array(Tsal_stderr), np.array(Uim_mean)

def mass_weighted_mean_se_forT(T, m):
    T = np.asarray(T, dtype=float)
    m = np.asarray(m, dtype=float)

    # mass weights (nonnegative), valid T>0
    w = m
    valid = np.isfinite(T) & np.isfinite(w) & (w > 0) & (T > 0)

    T = T[valid]
    w = w[valid]
    lam = 1.0 / T

    W  = w.sum()
    W2 = np.sum(w**2)

    # Weighted mean of lambda = 1/T
    lambda_bar = np.sum(w * lam) / W

    # Unbiased weighted variance of lambda
    denom = W - (W2 / W)
    if denom <= 0:
        se_lambda = np.nan
        n_eff = 0.0
    else:
        var_lambda = np.sum(w * (lam - lambda_bar)**2) / denom
        n_eff = (W**2) / W2
        se_lambda = np.sqrt(var_lambda / n_eff)

    # Harmonic mean of T and its SE via delta method:
    # T_harm = 1/lambda_bar,  Var(T_harm) ≈ (1/lambda_bar^4) Var(lambda_bar)
    T_harm = 1.0 / lambda_bar
    se_T = se_lambda / (lambda_bar**2)
    
    # Convert exact/near-zero SE to NaN if requested
    if np.isfinite(se_T) and (se_T == 0.0 or np.isclose(se_T, 0.0)):
        se_T = np.nan

    return T_harm, se_T