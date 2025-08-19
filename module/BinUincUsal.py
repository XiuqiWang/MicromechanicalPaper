# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:52:30 2025

@author: WangX3
"""
from .flux_weighted_mean_speed import flux_weighted_mean_speed
import numpy as np

# derive Uim-f(Usal); UD-f(Usal)
def BinUincUsal(velocities_im,velocities_dep,Vsals,Vreps,thetaims,thetaDs,mp_sals,mp_reps,Tim,Tdep,velsal_bins):
    Usals = np.concatenate([Vsals, Vreps])
    velocities_imx = np.array(velocities_im)*np.cos(np.deg2rad(np.array(thetaims)))
    velocities_imz = np.array(velocities_im)*np.sin(np.deg2rad(np.array(thetaims)))
    velocities_depx = np.array(velocities_dep)*np.cos(np.deg2rad(np.array(thetaDs)))
    velocities_depz = np.array(velocities_dep)*np.sin(np.deg2rad(np.array(thetaDs)))
    mp_sal = np.array(mp_sals)
    mp_rep = np.array(mp_reps)
    Tim = np.array(Tim)
    Tdep = np.array(Tdep)
    # weights_im = mp_sal*velocities_im
    # weights_dep = mp_rep*velocities_dep
    weights_sal = np.concatenate([mp_sal*Tim, mp_rep*Tdep])
    
    # Allocate CORs into velocity ranges
    Uim_mean, Uim_stderr, UD_mean, UD_stderr = [],[],[],[]
    Usal_mean = []
    for i in range(len(velsal_bins)-1):
        # Find indices of velocities within the current range
        indices = (Usals >= velsal_bins[i]) & (Usals < velsal_bins[i + 1])
        idx = np.where(indices)[0]
        idx = np.array(idx)
      
        if np.any(idx):  # Check if there are elements in this range
            indices_in_im = idx[idx < len(velocities_im)]
            Uimx_in_bin = velocities_imx[indices_in_im]
            Uimz_in_bin = velocities_imz[indices_in_im]
            weightsim_in_bin = mp_sal[indices_in_im]/Tim[indices_in_im]
            Uim_m, uim_se, _, _ = flux_weighted_mean_speed(Uimx_in_bin, Uimz_in_bin, weightsim_in_bin)
            Uim_mean.append(Uim_m)
            Uim_stderr.append(uim_se)
            
            indices_in_D = idx[idx >= len(velocities_im)] - len(velocities_im)
            UDx_in_bin = velocities_depx[indices_in_D]
            UDz_in_bin = velocities_depz[indices_in_D]
            weightsdep_in_bin = mp_rep[indices_in_D]/Tdep[indices_in_D]
            UD_m, uD_se, _, _ = flux_weighted_mean_speed(UDx_in_bin, UDz_in_bin, weightsdep_in_bin)
            UD_mean.append(UD_m)
            UD_stderr.append(uD_se)
            
            Usal_in_bin = Usals[indices]
            weights_sal_in_bin = weights_sal[indices]
            Usal_m, Usal_se, _, _ = mass_in_flight_weighted_Usal(Usal_in_bin, weights_sal_in_bin)
            Usal_mean.append(Usal_m)
        else:
            Uim_mean.append(np.nan)
            Uim_stderr.append(np.nan)
            UD_mean.append(np.nan)
            UD_stderr.append(np.nan)
            
    Usal_median = (velsal_bins[:-1] + velsal_bins[1:]) / 2 

    return np.array(Uim_mean), np.array(Uim_stderr), np.array(UD_mean), np.array(UD_stderr), np.array(Usal_mean)


def mass_in_flight_weighted_Usal(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)

    # Mask invalid data
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.nan, np.nan, np.nan, 0.0

    v = values[mask]
    w = weights[mask]

    W = np.sum(w)
    mean = np.sum(w * v) / W

    # Weighted variance (unbiased)
    denom = W - (np.sum(w**2) / W)
    if denom <= 0:
        var = 0.0
        n_eff = 1.0
    else:
        var = np.sum(w * (v - mean)**2) / denom
        n_eff = (W**2) / np.sum(w**2)

    std = np.sqrt(var)
    se = std / np.sqrt(n_eff) if n_eff > 0 else np.nan
    
    # Convert exact/near-zero SE to NaN if requested
    if np.isfinite(se) and (se == 0.0 or np.isclose(se, 0.0)):
        se = np.nan

    return mean, se, std, n_eff
