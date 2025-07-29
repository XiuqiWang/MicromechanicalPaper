# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:35:07 2025

@author: WangX3
"""
import numpy as np
import pandas as pd

def BinUimCOR_equalbinsize(velocities_im,velocities_re,thetares,vel_bins):
    # Sort velocities and corresponding CORs
    sorted_indices = np.argsort(velocities_im)
    velocities_im = np.array(velocities_im)[sorted_indices]
    Ures = np.array(velocities_re)[sorted_indices]
    thetares = np.array(thetares)[sorted_indices]
    # cors = np.array(cors)[sorted_indices]
    # Usals = np.array(Vsals)[sorted_indices]
    # thetaims = np.array(thetaims)[sorted_indices]
    
    
    # Allocate CORs into velocity ranges
    cor_num = []
    cor_means = []
    # cor_stds = []
    # cor_stderrs = []
    Ure_mean, Ure_stderr = [],[]
    thetare_mean, thetare_stderr = [],[]
    # Usal_mean,Uimx_mean = [],[]
    for i in range(len(vel_bins)-1):
        # Find indices of velocities within the current range
        indices = (velocities_im >= vel_bins[i]) & (velocities_im < vel_bins[i + 1])
        if np.any(indices):  # Check if there are elements in this range
            Uim_in_bin = velocities_im[indices]
            Ure_in_bin = Ures[indices]
            thetare_in_bin = thetares[indices]
            # cor_in_bin = cors[indices]
            # Usal_in_bin = Usals[indices]
            # thetaim_in_bin = thetaims[indices]
            cor_num.append(len(Uim_in_bin))
            Ure_mean.append(np.mean(Ure_in_bin))
            Ure_stderr.append(np.std(Ure_in_bin)/np.sqrt(len(Ure_in_bin)))
            thetare_mean.append(np.mean(thetare_in_bin))
            thetare_stderr.append(np.std(thetare_in_bin)/np.sqrt(len(thetare_in_bin)))
            cor_means.append(np.mean(Ure_in_bin)/np.mean(Uim_in_bin))
            # Usal_mean.append(np.mean(Usal_in_bin))
            # Uimx_mean.append(np.mean(Uim_in_bin*np.cos(thetaim_in_bin/180*np.pi)))
            # cor_means.append(np.mean(cor_in_bin))
            # cor_stds.append(np.std(cor_in_bin))
            # cor_stderrs.append(np.std(cor_in_bin)/np.sqrt(len(cor_in_bin)))
        else:
            cor_means.append(np.nan)  # No data in this bin
            # cor_stds.append(np.nan)
            # cor_stderrs.append(np.nan)
            cor_num.append(0)
            Ure_mean.append(np.nan)
            Ure_stderr.append(np.nan)
            thetare_mean.append(np.nan)
            thetare_stderr.append(np.nan)
            # Usal_mean.append(np.nan)
            # Uimx_mean.append(np.nan)
    
    Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2 
    print('Nim:', cor_num)
    
    return np.array(cor_means),np.array(cor_num),np.array(thetare_mean),np.array(thetare_stderr),Uplot