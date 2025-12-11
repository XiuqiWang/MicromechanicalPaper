# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:16:18 2025

@author: WangX3
"""
import numpy as np
import pandas as pd
import itertools

def get_ejection_theta_equalbinsize(matched_thetaim_all, matched_NE_all, matched_UE_all, thetaim_bins):
        # Convert to numpy arrays for easier manipulation
        impact_angles = np.array(matched_thetaim_all)
        ejection_numbers = np.array(matched_NE_all)
        ejection_velocities = np.array(matched_UE_all)
        
        # Create a mask where ejection_numbers is non-zero
        #non_zero_mask = ejection_numbers != 0

        # # Apply the mask to filter elements in all arrays
        # filtered_ejection_numbers = ejection_numbers#[non_zero_mask]
        # filtered_impact_velocities = impact_velocities#[non_zero_mask]
        # filtered_ejection_velocities = ejection_velocities#[non_zero_mask]
    
        ejection_ratios,VE_mean,VE_std = [],[],[]
        number = []
        for i in range(len(thetaim_bins)-1):
            # Find the indices of the velocities within the current bin range
            bin_mask = (impact_angles >= thetaim_bins[i]) & (impact_angles < thetaim_bins[i + 1])
            # Calculate the ejection ratio (ejection number / total impact number)
            if np.sum(bin_mask) > 0:
                ejection_ratio_mean = np.sum(ejection_numbers[bin_mask]) / np.sum(bin_mask)#np.mean(ejection_numbers[bin_mask])#
                #ejection_ratio_std = np.std(ejection_numbers[bin_mask])
                impact_num = np.sum(bin_mask)
                flattened_elements = list(itertools.chain.from_iterable(ejection_velocities[bin_mask]))
                VE_mean.append(np.mean(flattened_elements)) #flatten into array
                VE_std.append(np.std(flattened_elements))
                number.append(np.size(flattened_elements))
            else:
                ejection_ratio_mean = np.NAN
                impact_num = np.NAN
                VE_mean.append(np.NAN)
                VE_std.append(np.NAN)
            ejection_ratios.append(ejection_ratio_mean)#, impact_num])#, ejection_ratio_std])
        
        Thetaplot = (thetaim_bins[:-1] + thetaim_bins[1:]) / 2 
        
        
        print('number of ejections in each group:', number)
        
        return ejection_ratios, VE_mean, VE_std, Thetaplot