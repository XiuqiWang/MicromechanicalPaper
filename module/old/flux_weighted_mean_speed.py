# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 14:18:46 2025

@author: WangX3
"""
import numpy as np

def flux_weighted_mean_speed(ux, uy, w):
    ux = np.asarray(ux, dtype=float)
    uy = np.asarray(uy, dtype=float)
    w  = np.asarray(w,  dtype=float)

    if not (ux.shape == uy.shape == w.shape):
        raise ValueError(f"Shape mismatch: ux {ux.shape}, uy {uy.shape}, w {w.shape}")

    s = np.sqrt(ux**2 + uy**2)

    mask = np.isfinite(s) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan, 0.0

    s, w = s[mask], w[mask]

    W  = w.sum()
    W2 = np.sum(w**2)

    mean_speed = np.sum(w * s) / W

    # "Measure" variance (population under weights w = m/T)
    var = np.sum(w * (s - mean_speed)**2) / W
    std_speed = np.sqrt(var)

    # Kish effective sample size and SE of the weighted mean
    n_eff = (W**2) / W2
    se_mean = std_speed / np.sqrt(n_eff) if n_eff > 0 else np.nan
    
    # Convert exact/near-zero SE to NaN if requested
    if np.isfinite(se_mean) and (se_mean == 0.0 or np.isclose(se_mean, 0.0)):
        se_mean = np.nan

    return mean_speed, se_mean, std_speed, n_eff