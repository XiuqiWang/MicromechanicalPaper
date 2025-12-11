# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 16:09:37 2025

@author: WangX3
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:54:13 2025

@author: WangX3
"""
import numpy as np
import pandas as pd
import itertools
from .ratio_stats import ratio_stats
from .atan_ratio_stats import atan_ratio_stats
from .magnitude_stats import magnitude_stats

def BinNEUEmass(matched_Vim_all, VD_all, msal, mrep, matched_NE_all, matched_UE_all, matched_ThetaE_all, mE, velimde_bins):
        # Convert to numpy arrays for easier manipulation
        impact_velocities = np.array(matched_Vim_all)
        deposition_velocities = np.array(VD_all)
        msal = np.array(msal)
        mrep = np.array(mrep)
        ejection_numbers = np.array(matched_NE_all)
        ejection_velocities = np.array(matched_UE_all)
        ejection_angles = np.array(matched_ThetaE_all)
        mE = np.array(mE)
    
        ejection_ratios,UE_mean,UE_stds,UE_stderr = [],[],[],[]
        number = []
        impact_num, deposition_num = [],[]
        thetaEmean, thetaEstds, thetaEstderr = [],[],[]
        for i in range(len(velimde_bins)-1):
            # Find the indices of the velocities within the current bin range
            bin_mask = (impact_velocities >= velimde_bins[i]) & (impact_velocities < velimde_bins[i + 1])
            bin_mask_de = (deposition_velocities >= velimde_bins[i]) & (deposition_velocities < velimde_bins[i + 1])
            # Calculate the ejection ratio (ejection number / total impact number)
            if np.sum(bin_mask) > 0:
                flatten_mE = list(itertools.chain.from_iterable(mE[bin_mask]))
                ejection_ratio_mean = np.sum(flatten_mE) / (np.sum(msal[bin_mask]) + np.sum(mrep[bin_mask_de]))
                impact_num.append(np.sum(bin_mask))
                deposition_num.append(np.sum(bin_mask_de))
                flattened_elements = list(itertools.chain.from_iterable(ejection_velocities[bin_mask]))
                flattend_thetaE = list(itertools.chain.from_iterable(ejection_angles[bin_mask]))
                UEx = flattened_elements*np.cos(np.deg2rad(flattend_thetaE))
                UEz = flattened_elements*np.sin(np.deg2rad(flattend_thetaE))
                UE, UE_std, UE_se = magnitude_stats(UEx, UEz)
                UE_mean.append(UE)
                UE_stds.append(UE_std)
                UE_stderr.append(UE_se)
                thetaE, thetaEstd, thetaEse = atan_ratio_stats(UEz, UEx)
                thetaEmean.append(thetaE)
                thetaEstds.append(thetaEstd)
                thetaEstderr.append(thetaEse)
                number.append(np.size(flattened_elements))
            else:
                ejection_ratio_mean = np.nan
                impact_num.append(0)
                deposition_num.append(0)
                UE_mean.append(np.nan)
                UE_stds.append(np.nan)
                UE_stderr.append(np.nan)
                number.append(0)
                thetaEmean.append(np.nan)
                thetaEstds.append(np.nan)
                thetaEstderr.append(np.nan)
                
            ejection_ratios.append(ejection_ratio_mean)#, impact_num])#, ejection_ratio_std])
        
        Uplot = (velimde_bins[:-1] + velimde_bins[1:]) / 2 
        
        print('number of incidence:', [a+b for a,b in zip(impact_num,deposition_num)])
        print('number of ejections:', number)
        
        
        return ejection_ratios, UE_mean, UE_stds, UE_stderr, thetaEmean, thetaEstds, thetaEstderr, number, Uplot