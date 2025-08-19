# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 14:57:14 2025

@author: WangX3
"""
import numpy as np

def hop_average_Usal(times, u_stream, id_im, id_depa):
    times = np.asarray(times, dtype=float)
    u_stream = np.asarray(u_stream, dtype=float)
    t_depa_arr = np.asarray(times[id_depa], dtype=float)
    t_imp_arr = np.asarray(times[id_im], dtype=float)

    n_hops = len(t_depa_arr)
    U_sal_all = np.empty(n_hops)
    tau_h_all = np.empty(n_hops)
    n_samples_all = np.empty(n_hops, dtype=int)
    
    for k in range(n_hops):
        t_depa = t_depa_arr[k]
        t_imp = t_imp_arr[k]

        # Find slice indices
        i0 = int(np.searchsorted(times, t_depa, side='left'))
        iN = int(np.searchsorted(times, t_imp, side='right')) - 1
        # Extract segment
        t_seg = times[i0:iN+1]
        u_seg = u_stream[i0:iN+1]
        # Interpolate to exact hop boundaries if needed
        if t_seg[0] > t_depa:
            u0 = np.interp(t_depa, times, u_stream)
            t_seg = np.concatenate(([t_depa], t_seg))
            u_seg = np.concatenate(([u0], u_seg))
        if t_seg[-1] < t_imp:
            uN = np.interp(t_imp, times, u_stream)
            t_seg = np.concatenate((t_seg, [t_imp]))
            u_seg = np.concatenate((u_seg, [uN]))

        # Integration
        dt = np.diff(t_seg)
        if dt.size == 0:
            U_sal_all[k] = np.nan
            tau_h_all[k] = 0.0
            n_samples_all[k] = 0
            continue

        integral = np.sum(0.5 * (u_seg[:-1] + u_seg[1:]) * dt)
        tau_h = t_imp - t_depa

        U_sal_all[k] = integral / tau_h
        tau_h_all[k] = tau_h
        n_samples_all[k] = len(t_seg)

    return U_sal_all
