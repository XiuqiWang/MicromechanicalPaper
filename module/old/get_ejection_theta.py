# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 12:03:56 2025

@author: WangX3
"""

import numpy as np
import pandas as pd

def get_ejection_theta(matched_thetaim_all, matched_NE_all, matched_UE_all, num_bins):
    if not matched_thetaim_all:
        mean_NE, mean_UE, std_UE, UNEplot =[],[],[],[]
    else:
        # Combine data into a DataFrame and sort by impact velocity
        data = pd.DataFrame({'Thetaim': matched_thetaim_all, 'NE': matched_NE_all, 'UE': matched_UE_all})
        data = data.sort_values(by='Thetaim').reset_index(drop=True)
    
        # Get bin edges using quantiles
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(data['Thetaim'], quantiles)
    
        data['bin'] = pd.cut(data['Thetaim'], bins=bin_edges, include_lowest=True)
    
        #print(data.columns)  # 检查DataFrame的列名
        # for name, group in data.groupby('bin'):
        #     print(name, group['UE'].values)
            
        mean_NE = data.groupby('bin')['NE'].mean()
        data['bin'] = data['bin'].astype(str)
    
        # 对每个bin中的子列表进行均值计算
        mean_UE = data.groupby('bin')['UE'].apply(lambda x: np.nan if x.empty or all(len(i) == 0 for i in x) 
        else np.concatenate([i for i in x if len(i) > 0]).mean())
        std_UE = data.groupby('bin')['UE'].apply(lambda x: np.nan if x.empty or all(len(i) == 0 for i in x) 
        else np.concatenate([i for i in x if len(i) > 0]).std())
        ThetaNEplot = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return mean_NE, mean_UE, std_UE, ThetaNEplot 