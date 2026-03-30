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

def get_ejection_ratios_equalbinsize(matched_Vim_all, VD_all, matched_NE_all, matched_UE_all, matched_ThetaE_all, velimde_bins):#matched_EE_all
        # Convert to numpy arrays for easier manipulation
        impact_velocities = np.array(matched_Vim_all)
        deposition_velocities = np.array(VD_all)
        ejection_numbers = np.array(matched_NE_all)
        ejection_velocities = np.array(matched_UE_all)
        ejection_angles = np.array(matched_ThetaE_all)
    
        ejection_ratios,UE_mean,UE_stds,UE_stderr = [],[],[],[]
        number = []
        impact_num, deposition_num = [],[]
        thetaEmean, thetaEstds, thetaEstderr = [],[],[]
        momentum_ratio_mean, momentum_ratio_std = [], []
        for i in range(len(velimde_bins)-1):
            # Find the indices of the velocities within the current bin range
            bin_mask = (impact_velocities >= velimde_bins[i]) & (impact_velocities < velimde_bins[i + 1])
            bin_mask_de = (deposition_velocities >= velimde_bins[i]) & (deposition_velocities < velimde_bins[i + 1])
            # Calculate the ejection ratio (ejection number / total impact number)
            if np.sum(bin_mask) > 0:
                ejection_ratio_mean = np.sum(ejection_numbers[bin_mask]) / (np.sum(bin_mask) + np.sum(bin_mask_de))
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
                #check momentum partitioning
                # print('UE', ejection_velocities[bin_mask])
                # print('Uinc', impact_velocities[bin_mask])
                ratios_UE_Uim = [np.sum(u) / uim for u, uim in zip(ejection_velocities[bin_mask], impact_velocities[bin_mask])]
                # print('ratios_UE_Uim', ratios_UE_Uim)
                momentum_ratio_mean.append(np.mean(ratios_UE_Uim))
                momentum_ratio_std.append(np.std(ratios_UE_Uim))
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
                momentum_ratio_mean.append(np.nan)
                momentum_ratio_std.append(np.nan)
                
            ejection_ratios.append(ejection_ratio_mean)#, impact_num])#, ejection_ratio_std])
        
        Uplot = (velimde_bins[:-1] + velimde_bins[1:]) / 2 
        
        n_inc = [a+b for a,b in zip(impact_num,deposition_num)]
        print('number of incidence:', n_inc)
        print('number of ejections:', number)
        
        
        return ejection_ratios, UE_mean, UE_stds, UE_stderr, thetaEmean, thetaEstds, thetaEstderr, number, Uplot, n_inc, momentum_ratio_mean, momentum_ratio_std