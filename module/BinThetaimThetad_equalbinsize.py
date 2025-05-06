# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:31:56 2025

@author: WangX3
"""

import numpy as np

def BinThetaimThetad_equalbinsize(impact_angles,deposition_angles,theta_bins):
    # Combine all velocities to determine the range
    all_velocities = np.concatenate([impact_angles, deposition_angles])

    # Bin the velocities
    impact_bins = np.digitize(impact_angles, theta_bins)
    deposition_bins = np.digitize(deposition_angles, theta_bins)

    # Count the occurrences in each bin
    impact_counts = np.array([np.sum(impact_bins == i) for i in range(1, len(theta_bins))])
    deposition_counts = np.array([np.sum(deposition_bins == i) for i in range(1, len(theta_bins))])
    Pr = np.divide(impact_counts, deposition_counts+impact_counts, out=np.full_like(impact_counts, np.nan, dtype=np.float64), where=(deposition_counts+impact_counts) != 0)
    
    Thetaplot = (theta_bins[:-1] + theta_bins[1:]) / 2 
    
    return Pr,Thetaplot