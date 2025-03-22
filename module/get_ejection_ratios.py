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
    
    
        # ejection_numbers = []
        # ejection_velocities = []
        # for sublist in matched_ejection_list:
        #     ejection_numbers.append(len(sublist[0]))
        #     ejection_velocities.append(sublist[1])
        
        # # Convert to numpy arrays for easier manipulation
        # impact_velocities = np.array(impact_velocities)
        # ejection_numbers = np.array(ejection_numbers)
        # ejection_velocities = np.array(ejection_velocities)
        
        # # Create a mask where ejection_numbers is non-zero
        # #non_zero_mask = ejection_numbers != 0

        # # Apply the mask to filter elements in all arrays
        # filtered_ejection_numbers = ejection_numbers#[non_zero_mask]
        # filtered_impact_velocities = impact_velocities#[non_zero_mask]
        # filtered_ejection_velocities = ejection_velocities#[non_zero_mask]
    
        # ejection_ratios,VE_mean,VE_std = [],[],[]
        # number = []
        # for i in range(len(vel_bins)-1):
        #     # Find the indices of the velocities within the current bin range
        #     bin_mask = (filtered_impact_velocities >= vel_bins[i]) & (filtered_impact_velocities < vel_bins[i + 1])
        #     # Calculate the ejection ratio (ejection number / total impact number)
        #     if np.sum(bin_mask) > 0:
        #         ejection_ratio_mean = np.sum(filtered_ejection_numbers[bin_mask]) / np.sum(bin_mask)#np.mean(ejection_numbers[bin_mask])#
        #         #ejection_ratio_std = np.std(ejection_numbers[bin_mask])
        #         impact_num = np.sum(bin_mask)
        #         flattened_elements = list(itertools.chain.from_iterable(filtered_ejection_velocities[bin_mask]))
        #         VE_mean.append(np.mean(flattened_elements)) #flatten into array
        #         VE_std.append(np.std(flattened_elements))
        #         number.append(np.size(flattened_elements))
        #     else:
        #         ejection_ratio_mean = np.NAN
        #         impact_num = np.NAN
        #         VE_mean.append(np.NAN)
        #         VE_std.append(np.NAN)
        #     ejection_ratios.append(ejection_ratio_mean)#, impact_num])#, ejection_ratio_std])
        
        # Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2 
        
        
        # print('number of ejections in each group:', number)
        
        # return Uplot, ejection_ratios, VE_mean, VE_std