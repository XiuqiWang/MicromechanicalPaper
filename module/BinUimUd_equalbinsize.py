# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:51:45 2025

@author: WangX3
"""

import numpy as np

def BinUimUd_equalbinsize(impact_velocities, deposition_velocities, vel_bins):
    # Combine all velocities to determine the range
    all_velocities = np.concatenate([impact_velocities, deposition_velocities])

    # Bin the velocities
    impact_bins = np.digitize(impact_velocities, vel_bins)
    deposition_bins = np.digitize(deposition_velocities, vel_bins)

    n_bins = len(vel_bins) - 1
    impact_counts = np.zeros(n_bins, dtype=int)
    deposition_counts = np.zeros(n_bins, dtype=int)
    Uim_mean = np.full(n_bins, np.nan)
    Udep_mean = np.full(n_bins, np.nan)

    for i in range(1, len(vel_bins)):
        # Logical masks for current bin
        mask_im = impact_bins == i
        mask_dep = deposition_bins == i

        # Counts
        impact_counts[i - 1] = np.sum(mask_im)
        deposition_counts[i - 1] = np.sum(mask_dep)

        # Means (avoid empty slices)
        if np.any(mask_im):
            Uim_mean[i - 1] = np.mean(np.array(impact_velocities)[mask_im])
        if np.any(mask_dep):
            Udep_mean[i - 1] = np.mean(np.array(deposition_velocities)[mask_dep])

    # Compute probability of rebound
    total_counts = impact_counts + deposition_counts
    Pr = np.divide(
        impact_counts,
        total_counts,
        out=np.full_like(impact_counts, np.nan, dtype=np.float64),
        where=total_counts != 0
    )

    Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2
    print('number of incidence:', total_counts)
    print('number of rebounds:', impact_counts)

    return Pr, Uplot, impact_counts, Uim_mean, Udep_mean

# def BinUimUd_equalbinsize(impact_velocities,deposition_velocities,vel_bins):
#     # Combine all velocities to determine the range
#     all_velocities = np.concatenate([impact_velocities, deposition_velocities])

#     # Bin the velocities
#     impact_bins = np.digitize(impact_velocities, vel_bins)
#     deposition_bins = np.digitize(deposition_velocities, vel_bins)

#     # Count the occurrences in each bin
#     impact_counts = np.array([np.sum(impact_bins == i) for i in range(1, len(vel_bins))])
#     deposition_counts = np.array([np.sum(deposition_bins == i) for i in range(1, len(vel_bins))])
#     Pr = np.divide(impact_counts, deposition_counts+impact_counts, out=np.full_like(impact_counts, np.nan, dtype=np.float64), where=(deposition_counts+impact_counts) != 0)
    
#     Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2 
#     print('number of incidence', deposition_counts+impact_counts)
#     print('number of rebounds', impact_counts)
    
#     return Pr,Uplot,impact_counts