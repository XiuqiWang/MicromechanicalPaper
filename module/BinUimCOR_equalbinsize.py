# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:35:07 2025

@author: WangX3
"""
import numpy as np
from .atan_ratio_stats import atan_ratio_stats
from .ratio_stats import ratio_stats

def BinUimCOR_equalbinsize(velocities_im,velocities_re,thetares,vel_bins):
    # Sort velocities and corresponding CORs
    sorted_indices = np.argsort(velocities_im)
    velocities_im = np.array(velocities_im)[sorted_indices]
    Ures = np.array(velocities_re)[sorted_indices]
    thetares = np.array(thetares)[sorted_indices]
    Ures_x = Ures*np.cos(np.deg2rad(thetares))
    Ures_z = Ures*np.sin(np.deg2rad(thetares))
    
    # Allocate CORs into velocity ranges
    cor_num = []
    cor_means, cor_stds, cor_stderrs = [],[],[]
    Ure_mean, Ure_std, Ure_stderr = [],[],[]
    thetare_mean, thetare_stds, thetare_stderrs = [],[],[]
    # Usal_mean,Uimx_mean = [],[]
    for i in range(len(vel_bins)-1):
        # Find indices of velocities within the current range
        indices = (velocities_im >= vel_bins[i]) & (velocities_im < vel_bins[i + 1])
        if np.any(indices):  # Check if there are elements in this range
            Uim_in_bin = velocities_im[indices]
            Ure_in_bin = Ures[indices]
            Urex_in_bin = Ures_x[indices]
            Urez_in_bin = Ures_z[indices]
            cor_num.append(len(Uim_in_bin))
            # theta_re
            thetare, thetare_std, thetare_se = atan_ratio_stats(Urez_in_bin, Urex_in_bin)
            thetare_mean.append(thetare)
            thetare_stds.append(thetare_std)
            thetare_stderrs.append(thetare_se)
            # cor
            cor, cor_std, cor_se = ratio_stats(Ure_in_bin, Uim_in_bin)
            cor_means.append(cor)
            cor_stds.append(cor_std)
            cor_stderrs.append(cor_se)
        else:
            cor_means.append(np.nan)  # No data in this bin
            cor_stds.append(np.nan)
            cor_stderrs.append(np.nan)
            thetare_mean.append(np.nan)
            thetare_stds.append(np.nan)
            thetare_stderrs.append(np.nan)
            cor_num.append(0)
    
    Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2 
    
    return np.array(cor_means), np.array(cor_stds), np.array(cor_stderrs), np.array(cor_num),np.array(thetare_mean), np.array(thetare_stds), np.array(thetare_stderrs), Uplot