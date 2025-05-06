# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:49:59 2025

@author: WangX3
"""
import numpy as np
import pandas as pd

def BinThetaimCOR_equalbinsize(thetas,cors,theta_bins):
# Sort velocities and corresponding CORs
    sorted_indices = np.argsort(thetas)
    thetas = np.array(thetas)[sorted_indices]
    cors = np.array(cors)[sorted_indices]
    
    # Allocate CORs into velocity ranges
    cor_means = []
    cor_stds = []
    counts = []  # 新增：存储每个 bin 的数据点数量
    for i in range(len(theta_bins)-1):
        # Find indices of velocities within the current range
        indices = (thetas >= theta_bins[i]) & (thetas < theta_bins[i + 1])
        if np.any(indices):  # Check if there are elements in this range
            cor_in_bin = cors[indices]
            cor_means.append(np.mean(cor_in_bin))
            cor_stds.append(np.std(cor_in_bin))
            counts.append(len(cor_in_bin))
        else:
            cor_means.append(np.nan)  # No data in this bin
            cor_stds.append(np.nan)
            counts.append(0)
    
    Uplot = (theta_bins[:-1] + theta_bins[1:]) / 2 
    cor_stderr = cor_stds/np.sqrt(counts)
    
    print('N_im:', counts)
    
    return cor_means,cor_stderr,Uplot