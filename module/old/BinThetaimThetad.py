# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:13:49 2025

@author: WangX3
"""

import numpy as np

def BinThetaimThetad(impact_angles,deposition_angles,num_bins):
    if not impact_angles and not deposition_angles:
        Pr, Thetaplot = [],[]
    else:
        # Combine all velocities to determine the range
        all_angles = np.concatenate([impact_angles, deposition_angles])

        # Compute bin edges using quantiles
        angle_bins = np.quantile(all_angles, np.linspace(0, 1, num_bins + 1))

        # Assign each velocity to a bin
        impact_bins = np.digitize(impact_angles, angle_bins, right=False)
        deposition_bins = np.digitize(deposition_angles, angle_bins, right=False)

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
    
        Thetaplot = (angle_bins[:-1] + angle_bins[1:]) / 2
    
    return Pr, Thetaplot