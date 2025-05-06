# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 13:51:45 2025

@author: WangX3
"""

import numpy as np

def BinUimUd_equalbinsize(impact_velocities,deposition_velocities,vel_bins):
    # Combine all velocities to determine the range
    all_velocities = np.concatenate([impact_velocities, deposition_velocities])

    # Bin the velocities
    impact_bins = np.digitize(impact_velocities, vel_bins)
    deposition_bins = np.digitize(deposition_velocities, vel_bins)

    # Count the occurrences in each bin
    impact_counts = np.array([np.sum(impact_bins == i) for i in range(1, len(vel_bins))])
    deposition_counts = np.array([np.sum(deposition_bins == i) for i in range(1, len(vel_bins))])
    Pr = np.divide(impact_counts, deposition_counts+impact_counts, out=np.full_like(impact_counts, np.nan, dtype=np.float64), where=(deposition_counts+impact_counts) != 0)
    
    Uplot = (vel_bins[:-1] + vel_bins[1:]) / 2 
    print('number of incidence', deposition_counts+impact_counts)
    print('number of rebounds', impact_counts)
    
    return Pr,Uplot,impact_counts