# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:46:03 2025

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
    print("Bin Edges:", bin_edges)
    print("\nMean COR in each bin:\n", mean_cor)
    print("\nStandard Deviation of COR in each bin:\n", std_cor)
    bin_counts = data['bin'].value_counts().sort_index()
    print("\nNumber of data points in each bin:\n", bin_counts)
    
    return mean_cor, std_cor, Uplot