# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:40:08 2025

@author: WangX3
"""
import numpy as np
import pandas as pd

def BinThetaimCOR(thetas,cors,theta_bins):
    if not thetas:
        mean_cor, std_cor, Uplot = [],[],[]
    else:
        data = pd.DataFrame({'theta': thetas, 'cor': cors})
        data = data.sort_values(by='theta').reset_index(drop=True)
    
        # Get bin edges using quantiles
        quantiles = np.linspace(0, 1, theta_bins + 1)
        bin_edges = np.quantile(data['theta'], quantiles)
    
        data['bin'] = pd.cut(data['theta'], bins=bin_edges, include_lowest=True)

        mean_cor = data.groupby('bin')['cor'].mean()
        std_cor = data.groupby('bin')['cor'].std()
    
        Uplot = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return mean_cor, std_cor, Uplot
    
    
    
    
    # # Sort velocities and corresponding CORs
    # sorted_indices = np.argsort(thetas)
    # thetas = np.array(thetas)[sorted_indices]
    # cors = np.array(cors)[sorted_indices]
    
    # # Allocate CORs into velocity ranges
    # cor_means = []
    # cor_stds = []
    # for i in range(len(theta_bins)-1):
    #     # Find indices of velocities within the current range
    #     indices = (thetas >= theta_bins[i]) & (thetas < theta_bins[i + 1])
    #     if np.any(indices):  # Check if there are elements in this range
    #         cor_in_bin = cors[indices]
    #         cor_means.append(np.mean(cor_in_bin))
    #         cor_stds.append(np.std(cor_in_bin))
    #     else:
    #         cor_means.append(np.nan)  # No data in this bin
    #         cor_stds.append(np.nan)
    
    # Uplot = (theta_bins[:-1] + theta_bins[1:]) / 2 
    # return cor_means,cor_stds,Uplot