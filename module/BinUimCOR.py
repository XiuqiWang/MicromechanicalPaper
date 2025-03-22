# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:35:07 2025

@author: WangX3
"""
import numpy as np
import pandas as pd

def BinUimCOR(velocities,cors,num_bins):
    if not velocities:
        mean_cor, std_cor, Uplot = [],[],[]
    else:
        # Combine data into a DataFrame and sort by impact velocity
        data = pd.DataFrame({'velocity': velocities, 'cor': cors})
        data = data.sort_values(by='velocity').reset_index(drop=True)
    
        # Get bin edges using quantiles
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(data['velocity'], quantiles)
    
        data['bin'] = pd.cut(data['velocity'], bins=bin_edges, include_lowest=True)

        mean_cor = data.groupby('bin')['cor'].mean()
        std_cor = data.groupby('bin')['cor'].std()
    
        Uplot = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Display the results
    # print("Bin Edges:", bin_edges)
    # print("\nMean COR in each bin:\n", mean_cor)
    # print("\nStandard Deviation of COR in each bin:\n", std_cor)
    #bin_counts = data['bin'].value_counts().sort_index()
    #print("\nNumber of data points in each bin:\n", bin_counts)
    
    return mean_cor, std_cor, Uplot
    
    # Sort velocities and corresponding CORs
    # sorted_indices = np.argsort(velocities)
    # velocities = np.array(velocities)[sorted_indices]
    # cors = np.array(cors)[sorted_indices]
    
    # Allocate CORs into velocity ranges
    # cor_means = []
    # cor_stds = []
    # for i in range(len(vel_bins)-1):
    #     # Find indices of velocities within the current range
    #     indices = (velocities >= vel_bins[i]) & (velocities < vel_bins[i + 1])
    #     if np.any(indices):  # Check if there are elements in this range
    #         cor_in_bin = cors[indices]
    #         cor_means.append(np.mean(cor_in_bin))
    #         cor_stds.append(np.std(cor_in_bin))
    #     else:
    #         cor_means.append(np.nan)  # No data in this bin
    #         cor_stds.append(np.nan)
    
    # Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2 
    # return cor_means,cor_stds,Uplot