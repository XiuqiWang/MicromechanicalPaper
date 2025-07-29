# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:52:30 2025

@author: WangX3
"""

import numpy as np
import pandas as pd

def BinUincUsal(velocities_im,velocities_dep,Vsals,Vreps,thetaims,thetaDs,vel_bins):
    velocities_inc = np.array(velocities_im + velocities_dep)
    theta_inc = np.array(thetaims + thetaDs)
    Usals = np.array(Vsals + Vreps)
    
    # Allocate CORs into velocity ranges
    Uinc_num = []
    Usal_mean, Usal_stderr, Uincx_mean = [],[],[]
    for i in range(len(vel_bins)-1):
        # Find indices of velocities within the current range
        indices = (velocities_inc >= vel_bins[i]) & (velocities_inc < vel_bins[i + 1])
        if np.any(indices):  # Check if there are elements in this range
            Uinc_in_bin = velocities_inc[indices]
            Usal_in_bin = Usals[indices]
            theta_in_bin = theta_inc[indices]
            Uinc_num.append(len(Uinc_in_bin))
            Usal_mean.append(np.mean(Usal_in_bin))
            Uincx_mean.append(np.mean(Uinc_in_bin*np.cos(theta_in_bin/180*np.pi)))
            # Uinc_mean.append(np.mean(Uinc_in_bin))
            Usal_stderr.append(np.std(Usal_in_bin)/np.sqrt(len(Uinc_in_bin)))
        else:
            Uinc_num.append(0)
            Usal_mean.append(np.nan)
            Usal_stderr.append(np.nan)
            Uincx_mean.append(np.nan)
            # Uinc_mean.append(np.nan)
            
    Uinc_mean = (vel_bins[:-1] + vel_bins[1:]) / 2 

    return np.array(Usal_mean), np.array(Usal_stderr), np.array(Uincx_mean), np.array(Uinc_mean)