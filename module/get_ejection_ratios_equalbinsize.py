# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:54:13 2025

@author: WangX3
"""
import numpy as np
import pandas as pd
import itertools

def get_ejection_ratios_equalbinsize(matched_Vim_all, VD_all, matched_NE_all, matched_UE_all, velimde_bins):#matched_EE_all
        # Convert to numpy arrays for easier manipulation
        impact_velocities = np.array(matched_Vim_all)
        deposition_velocities = np.array(VD_all)
        ejection_numbers = np.array(matched_NE_all)
        ejection_velocities = np.array(matched_UE_all)
        # ejection_energy = np.array(matched_EE_all)
        
        # Create a mask where ejection_numbers is non-zero
        #non_zero_mask = ejection_numbers != 0

        # # Apply the mask to filter elements in all arrays
        # filtered_ejection_numbers = ejection_numbers#[non_zero_mask]
        # filtered_impact_velocities = impact_velocities#[non_zero_mask]
        # filtered_ejection_velocities = ejection_velocities#[non_zero_mask]
    
        ejection_ratios,VE_mean,VE_std = [],[],[]
        number = []
        impact_num, deposition_num = [],[]
        EE_mean = []
        for i in range(len(velimde_bins)-1):
            # Find the indices of the velocities within the current bin range
            bin_mask = (impact_velocities >= velimde_bins[i]) & (impact_velocities < velimde_bins[i + 1])
            bin_mask_de = (deposition_velocities >= velimde_bins[i]) & (deposition_velocities < velimde_bins[i + 1])
            # Calculate the ejection ratio (ejection number / total impact number)
            if np.sum(bin_mask) > 0:
                ejection_ratio_mean = np.sum(ejection_numbers[bin_mask]) / (np.sum(bin_mask) + np.sum(bin_mask_de))#np.mean(ejection_numbers[bin_mask])#
                #ejection_ratio_std = np.std(ejection_numbers[bin_mask])
                impact_num.append(np.sum(bin_mask))
                deposition_num.append(np.sum(bin_mask_de))
                flattened_elements = list(itertools.chain.from_iterable(ejection_velocities[bin_mask]))
                VE_mean.append(np.mean(flattened_elements)) #flatten into array
                VE_std.append(np.std(flattened_elements))
                # flattened_EE = list(itertools.chain.from_iterable(ejection_energy[bin_mask]))
                # EE_mean.append(np.mean(flattened_EE)/(np.sum(bin_mask) + np.sum(bin_mask_de)))
                number.append(np.size(flattened_elements))
            else:
                ejection_ratio_mean = np.NAN
                impact_num.append(0)
                deposition_num.append(0)
                VE_mean.append(np.NAN)
                VE_std.append(np.NAN)
                # EE_mean.append(np.NAN)
                number.append(0)
            ejection_ratios.append(ejection_ratio_mean)#, impact_num])#, ejection_ratio_std])
        
        Uplot = (velimde_bins[:-1] + velimde_bins[1:]) / 2 
        
        # print('number of impact:', impact_num)
        # print('number of deposition:', deposition_num)
        # print('number of incidence:', [a+b for a,b in zip(impact_num,deposition_num)])
        print('number of ejections:', number)
        std_error = [a/np.sqrt(b) for a,b in zip(VE_std, number)]
        # print('standard error', std_error)
        
        
        return ejection_ratios, VE_mean, VE_std, std_error, Uplot, number