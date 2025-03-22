# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:46:26 2025

@author: WangX3
"""
import numpy as np

def BinUimUd(impact_velocities,deposition_velocities,num_bins):
    if not impact_velocities and not deposition_velocities:
        Pr, Uplot = [],[]
    else:
        # Combine all velocities to determine the range
        all_velocities = np.concatenate([impact_velocities, deposition_velocities])

        # Compute bin edges using quantiles
        vel_bins = np.quantile(all_velocities, np.linspace(0, 1, num_bins + 1))

        # Assign each velocity to a bin
        impact_bins = np.digitize(impact_velocities, vel_bins, right=False)
        deposition_bins = np.digitize(deposition_velocities, vel_bins, right=False)

        # Ensure bin indices stay within valid range
        impact_bins[impact_bins == num_bins + 1] = num_bins
        deposition_bins[deposition_bins == num_bins + 1] = num_bins

        # Count the occurrences in each bin
        impact_counts = np.array([np.sum(impact_bins == i) for i in range(1, num_bins + 1)])
        deposition_counts = np.array([np.sum(deposition_bins == i) for i in range(1, num_bins + 1)])

        # Compute the probability of rebound
        Pr = np.divide(
            impact_counts, 
            deposition_counts + impact_counts, 
            out=np.full_like(impact_counts, np.nan, dtype=np.float64), 
            where=(deposition_counts + impact_counts) != 0)
    
        Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2
    
    return Pr, Uplot
    
    
    
    
    
    # # Combine all velocities to determine the range
    # all_velocities = np.concatenate([impact_velocities, deposition_velocities])

    # # Bin the velocities
    # impact_bins = np.digitize(impact_velocities, vel_bins)
    # deposition_bins = np.digitize(deposition_velocities, vel_bins)

    # # Count the occurrences in each bin
    # impact_counts = np.array([np.sum(impact_bins == i) for i in range(1, len(vel_bins))])
    # deposition_counts = np.array([np.sum(deposition_bins == i) for i in range(1, len(vel_bins))])
    # Pr = np.divide(impact_counts, deposition_counts+impact_counts, out=np.full_like(impact_counts, np.nan, dtype=np.float64), where=(deposition_counts+impact_counts) != 0)
    
    # Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2 
    # return Pr,Uplot