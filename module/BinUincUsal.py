# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:52:30 2025

@author: WangX3
"""

import numpy as np
import pandas as pd

def BinUincUsal(velocities_im,velocities_dep,Vsals,Vreps,thetaims,thetaDs,vel_bins):
    # Sort velocities and corresponding CORs
    # sorted_indices = np.argsort(velocities_im)
    # velocities_im = np.array(velocities_im)[sorted_indices]
    # velocities_dep = np.array(velocities_dep)[sorted_indices]
    # Usals = np.array(Vsals)[sorted_indices]
    # Ureps = np.array(Vreps)[sorted_indices]
    # thetaims = np.array(thetaims)[sorted_indices]
    # thetaDs = np.array(thetaDs)[sorted_indices]
    velocities_inc = np.array(velocities_im + velocities_dep)
    theta_inc = np.array(thetaims + thetaDs)
    Usals = np.array(Vsals + Vreps)
    
    # Allocate CORs into velocity ranges
    # cor_num = []
    Usal_mean,Uincx_mean = [],[]
    for i in range(len(vel_bins)-1):
        # Find indices of velocities within the current range
        indices = (velocities_inc >= vel_bins[i]) & (velocities_inc < vel_bins[i + 1])
        if np.any(indices):  # Check if there are elements in this range
            Uinc_in_bin = velocities_inc[indices]
            Usal_in_bin = Usals[indices]
            theta_in_bin = theta_inc[indices]
            # cor_num.append(len(cor_in_bin))
            Usal_mean.append(np.mean(Usal_in_bin))
            Uincx_mean.append(np.mean(Uinc_in_bin*np.cos(theta_in_bin/180*np.pi)))
        else:
            # cor_num.append(0)
            Usal_mean.append(np.nan)
            Uincx_mean.append(np.nan)
    
    # Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2 
    # print('Nim:', cor_num)
    
    return np.array(Usal_mean), np.array(Uincx_mean)