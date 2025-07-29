# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:52:30 2025

@author: WangX3
"""

import numpy as np
import pandas as pd

# derive Uim-f(Usal); UD-f(Usal)
def BinUincUsal(velocities_im,velocities_dep,Vsals,Vreps,thetaims,thetaDs,velsal_bins):
    velocities_inc = np.concatenate([velocities_im, velocities_dep])
    theta_inc = np.concatenate([thetaims, thetaDs])
    Usals = np.concatenate([Vsals, Vreps])
    velocities_im = np.array(velocities_im)
    velocities_dep = np.array(velocities_dep)
    
    # Allocate CORs into velocity ranges
    Uim_num,UD_num = [],[]
    Uim_mean, Uim_stderr, UD_mean, UD_stderr = [],[],[],[]
    Usal_mean = []
    for i in range(len(velsal_bins)-1):
        # Find indices of velocities within the current range
        indices = (Usals >= velsal_bins[i]) & (Usals < velsal_bins[i + 1])
        idx = np.where(indices)[0]
        idx = np.array(idx)
        # print('indices', indices)
        print('length idx', len(idx))
        if np.any(idx):  # Check if there are elements in this range
            # Uinc_in_bin = velocities_inc[indices]
            indices_in_im = idx[idx < len(velocities_im)]
            # print('indices_in_im:', indices_in_im)
            print('length indices_in_im', len(indices_in_im))
            Uim_in_bin = velocities_im[indices_in_im]
            # thetaim_in_bin = thetaims[indices_in_im]
            indices_in_D = idx[idx >= len(velocities_im)] - len(velocities_im)
            UD_in_bin = velocities_dep[indices_in_D]
            # thetaD_in_bin = thetaDs[indices_in_D]
            # Uim_num.append(len(Uim_in_bin))
            Uim_mean.append(np.mean(Uim_in_bin))
            # UD_num.append(len(UD_in_bin))
            UD_mean.append(np.mean(UD_in_bin))
            # Uincx_mean.append(np.mean(Uinc_in_bin*np.cos(theta_in_bin/180*np.pi)))
            # Uinc_mean.append(np.mean(Uinc_in_bin))
            Uim_stderr.append(np.std(Uim_in_bin)/np.sqrt(len(Uim_in_bin)))
            UD_stderr.append(np.std(UD_in_bin)/np.sqrt(len(UD_in_bin)))
        else:
            # Uim_num.append(0)
            Uim_mean.append(np.nan)
            Uim_stderr.append(np.nan)
            UD_mean.append(np.nan)
            UD_stderr.append(np.nan)
            # Uincx_mean.append(np.nan)
            # Uinc_mean.append(np.nan)
            
    Usal_mean = (velsal_bins[:-1] + velsal_bins[1:]) / 2 

    return np.array(Uim_mean), np.array(Uim_stderr), np.array(UD_mean), np.array(UD_stderr), np.array(Usal_mean)