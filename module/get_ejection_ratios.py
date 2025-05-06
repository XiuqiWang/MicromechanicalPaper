# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:50:46 2025

@author: WangX3
"""
import numpy as np
import pandas as pd

def get_ejection_ratios(matched_Vim_all, matched_NE_all, matched_UE_all, num_bins):
    if not matched_Vim_all:
        mean_NE, mean_UE, std_UE, UNEplot =[],[],[],[]
    else:
        # Combine data into a DataFrame and sort by impact velocity
        data = pd.DataFrame({'Vim': matched_Vim_all, 'NE': matched_NE_all, 'UE': matched_UE_all})
        data = data.sort_values(by='Vim').reset_index(drop=True)
    
        # Get bin edges using quantiles
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(data['Vim'], quantiles)
    
        data['bin'] = pd.cut(data['Vim'], bins=bin_edges, include_lowest=True)
    
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
        UNEplot = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return mean_NE, mean_UE, std_UE, UNEplot 
    
    
        