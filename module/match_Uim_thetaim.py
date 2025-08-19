# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 10:35:12 2025

@author: WangX3
"""
import numpy as np
from .atan_ratio_stats import atan_ratio_stats

def match_Uim_thetaim(Vim, thetaim, vel_bins):
    # Sort velocities and corresponding CORs
    sorted_indices = np.argsort(Vim)
    velocities = np.array(Vim)[sorted_indices]
    angles = np.array(thetaim)[sorted_indices]
    Vx = velocities*np.cos(np.deg2rad(angles))
    Vz = velocities*np.sin(np.deg2rad(angles))
    
    # Allocate CORs into velocity ranges
    mean_thetaim, std_thetaim, stderr_thetaim = [],[],[]
    counts = []  # 新增：存储每个 bin 的数据点数量
    for i in range(len(vel_bins)-1):
        # Find indices of velocities within the current range
        indices = (velocities >= vel_bins[i]) & (velocities < vel_bins[i + 1])
        if np.any(indices):  # Check if there are elements in this range
            theta_in_bin = angles[indices]
            Vx_in_bin = Vx[indices]
            Vz_in_bin = Vz[indices]
            theta, theta_std, theta_se = atan_ratio_stats(Vz_in_bin, Vx_in_bin)
            mean_thetaim.append(theta)
            std_thetaim.append(theta_std)
            stderr_thetaim.append(theta_se)
            counts.append(len(Vx_in_bin))  # number of data points in each bin
        else:
            mean_thetaim.append(np.nan)  # No data in this bin
            std_thetaim.append(np.nan)
            stderr_thetaim.append(np.nan)
            counts.append(0)
    
    Uthetaplot = (vel_bins[:-1] + vel_bins[1:]) / 2 
    counts = np.array(counts)
    
    return np.array(mean_thetaim), np.array(stderr_thetaim), counts, np.array(Uthetaplot)
    
    
    
    
    # if not Vim:
    #     mean_thetaim, std_thetaim, Uthetaplot =[],[],[]
    # else:
    #     # Combine data into a DataFrame and sort by impact velocity
    #     data = pd.DataFrame({'Vim': Vim, 'thetaim': thetaim})
    #     data = data.sort_values(by='Vim').reset_index(drop=True)
    
    #     # Get bin edges using quantiles
    #     quantiles = np.linspace(0, 1, vel_bins + 1)
    #     bin_edges = np.quantile(data['Vim'], quantiles)
    
    #     data['bin'] = pd.cut(data['Vim'], bins=bin_edges, include_lowest=True)
    
    #     #print(data.columns)  # 检查DataFrame的列名
    #     # for name, group in data.groupby('bin'):
    #     #     print(name, group['UE'].values)
            
    #     mean_thetaim = data.groupby('bin')['thetaim'].mean()
    #     std_thetaim = data.groupby('bin')['thetaim'].std()
    #     data['bin'] = data['bin'].astype(str)
    
    #     Uthetaplot = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # return mean_thetaim, std_thetaim, Uthetaplot
    
    
    
    
    

    