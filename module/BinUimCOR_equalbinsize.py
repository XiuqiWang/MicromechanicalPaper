# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:35:07 2025

@author: WangX3
"""
import numpy as np
import pandas as pd

def BinUimCOR_equalbinsize(velocities,cors,thetares,vel_bins):
    # Sort velocities and corresponding CORs
    sorted_indices = np.argsort(velocities)
    velocities = np.array(velocities)[sorted_indices]
    cors = np.array(cors)[sorted_indices]
    theta = np.array(thetares)[sorted_indices]
    
    # Allocate CORs into velocity ranges
    cor_num = []
    cor_means = []
    cor_stds = []
    cor_stderrs = []
    thetare_mean, thetare_stderr = [],[]
    for i in range(len(vel_bins)-1):
        # Find indices of velocities within the current range
        indices = (velocities >= vel_bins[i]) & (velocities < vel_bins[i + 1])
        if np.any(indices):  # Check if there are elements in this range
            cor_in_bin = cors[indices]
            thetare_in_bin = theta[indices]
            cor_num.append(len(cor_in_bin))
            cor_means.append(np.mean(cor_in_bin))
            cor_stds.append(np.std(cor_in_bin))
            cor_stderrs.append(np.std(cor_in_bin)/np.sqrt(len(cor_in_bin)))
            thetare_mean.append(np.mean(thetare_in_bin))
            thetare_stderr.append(np.std(thetare_in_bin)/np.sqrt(len(thetare_in_bin)))
        else:
            cor_means.append(np.nan)  # No data in this bin
            cor_stds.append(np.nan)
            cor_stderrs.append(np.nan)
            cor_num.append(0)
            thetare_mean.append(np.nan)
            thetare_stderr.append(np.nan)
    
    Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2 
    print('Nim:', cor_num)
    
    return cor_means,cor_stds,cor_stderrs,thetare_mean,thetare_stderr,Uplot