# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:49:59 2025

@author: WangX3
"""
import numpy as np
from .ratio_stats import ratio_stats

def BinThetaimCOR_equalbinsize(thetas,Uims, Ures,theta_bins):
# Sort velocities and corresponding CORs
    sorted_indices = np.argsort(thetas)
    thetas = np.array(thetas)[sorted_indices]
    Uims = np.array(Uims)[sorted_indices]
    Ures = np.array(Ures)[sorted_indices]
    
    # Allocate CORs into velocity ranges
    cor_means, cor_stds, cor_stderrs = [],[],[]
    counts = []  # 新增：存储每个 bin 的数据点数量
    for i in range(len(theta_bins)-1):
        # Find indices of velocities within the current range
        indices = (thetas >= theta_bins[i]) & (thetas < theta_bins[i + 1])
        if np.any(indices):  # Check if there are elements in this range
            Uim_in_bin = Uims[indices]
            Ure_in_bin = Ures[indices]
            cor, cor_std, cor_se = ratio_stats(Ure_in_bin, Uim_in_bin)
            cor_means.append(cor)
            cor_stds.append(cor_std)
            cor_stderrs.append(cor_se)
            counts.append(len(Uim_in_bin))
        else:
            cor_means.append(np.nan)  # No data in this bin
            cor_stds.append(np.nan)
            cor_stderrs.append(np.nan)
            counts.append(0)
    
    thetaplot = (theta_bins[:-1] + theta_bins[1:]) / 2 
    
    return cor_means, cor_stds, counts, thetaplot