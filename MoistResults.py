# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:00:23 2025

@author: WangX3
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from collections import defaultdict
import module
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from itertools import chain
from sklearn.metrics import r2_score

# Load the data from the saved file
data_dict = {}  # Dictionary to store data
for i in range(2,7): #Theta = 0.02-0.06
    for j in range(5):
        filename = f"input_pkl/data{i}_{j}.pkl"  # Generate filename
        with open(filename, 'rb') as f:
            data_dict[f"data{5*(i-2)+(j+1)}"] = pickle.load(f)  # Store in dictionary  
   
#basic parameters
dt = 0.01
D = 0.00025
coe_h = 13.5 #critial height for a mobile particle to reach
coe_sal_h = 17
N_inter = 100 #number of output timesteps for erosion and deposition properties
t_inter = np.linspace(0,5,N_inter+1)
Omega = [0, 1, 5, 10, 20]
colors = plt.cm.viridis(np.linspace(1, 0, 5))  # 5 colors
colors_n = plt.cm.plasma(np.linspace(0, 1, 3))
t_ver = np.linspace(dt, 5, 500)

#initialize
EDindices = defaultdict(list)
ME,MD,MoE,MoD = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
VExz_mean_t, VDxz_mean_t= defaultdict(list), defaultdict(list)
E_vector_t,D_vector_t= defaultdict(list),defaultdict(list)

exz_mean_t,ez_mean_t = defaultdict(list),defaultdict(list)
# ez_t = defaultdict(list)
# exz_t = defaultdict(list)
exz_vector_t,Vim_vector_t = defaultdict(list),defaultdict(list)
IM_vector_t= defaultdict(list)
VIM_mean_t,ThetaIM_mean_t = defaultdict(list), defaultdict(list)
# VIM_t = defaultdict(list)
# Thetaim_t = defaultdict(list)
RIM = defaultdict(list)
Par = defaultdict(list)
VZ = defaultdict(list)
X,Z = defaultdict(list),defaultdict(list)
for i in range(25):
    filename = f"data{i+1}"
    data = data_dict[filename]
    if i in [11,17,18]:#0.04 1%, 0.05 5,10%
        num_p = 2714
    else:
        num_p = 2725
    ParticleID=np.linspace(num_p-300-10,num_p-10-1,300)
    ParticleID_int = ParticleID.astype(int)
    #cal erosion and deposition properties for each Omega
    #EDindices, E, VX, VExVector, VEzVector, VEx, VEz, ME, MD
    EDindices[i], ME[i], MD[i], VExz_mean_t[i], VDxz_mean_t[i], D_vector_t[i], E_vector_t[i]=module.store_particle_id_data(data,ParticleID_int,coe_h,dt,N_inter,D)
    #cal rebound properties for each Omega
    ParticleID_sal=np.linspace(num_p-310,num_p-1,310)
    ParticleID_salint = ParticleID_sal.astype(int)
    X[i] = np.array([[time_step['Position'][i][0] for i in ParticleID_salint] for time_step in data])
    Z[i] = np.array([[time_step['Position'][i][2] for i in ParticleID_salint] for time_step in data])
    Par[i], VZ[i], exz_mean_t[i], ez_mean_t[i], VIM_mean_t[i], ThetaIM_mean_t[i], RIM[i], exz_vector_t[i], IM_vector_t[i]=module.store_sal_id_data(data,ParticleID_salint, coe_sal_h, dt, N_inter, D)
 
    
exz_all,Vim_all,VD_all,ThetaD_all,Theta_all,Thetare_all,impact_list,impact_deposition_list,ejection_list = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
Vre_all, Vsal_all = defaultdict(list), defaultdict(list)
Vrep_all = defaultdict(list)
ThetaE_all, UE_all = defaultdict(list), defaultdict(list)
N_range = np.full(25, 0).astype(int)
for i in range (25):
    exz_all[i] = [value for sublist in exz_vector_t[i][N_range[i]:] for value in sublist]
    Vim_all[i] = [value[0] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Vre_all[i] = [value[10] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Vsal_all[i] = [value[11] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    VD_all[i] = [value[0] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    Vrep_all[i] = [value[-1] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    ThetaD_all[i] = [value[1] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    Theta_all[i] = [value[7] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Thetare_all[i] = [value[8] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    IDim = [value[1] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    IDre = [value[2] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    xim = [value[3] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    xre = [value[4] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    xcol = [value[5] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Pim = [value[6] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    impact_list[i] = [IDim, IDre, xim, xre, Vim_all[i], xcol, Pim, Theta_all[i]]
    # IDD =  [value[2] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    # xD = [value[3] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    # PD = [value[4] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    # ThetaD = [value[1] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    # impact_deposition_list[i] = [IDim + IDD, xcol + xD, Vim_all[i] + VD_all[i], Pim + PD, Theta_all[i] + ThetaD_all[i]]
    IDE = [value[1] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    xE = [value[2] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    PE = [value[3] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    EE = [value[5] for sublist in E_vector_t[i][N_range[i]:] for value in sublist] #kinetic energy
    ThetaE_all[i] = [value[6] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    UE_all[i] = [value[0] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    ejection_list[i] = [IDE, xE, UE_all[i], PE, EE, ThetaE_all[i]]
    #print('Ne/Nim',len(IDE)/len(IDim))

Vim_all_values = [value for sublist in Vim_all.values() for value in sublist]
Vim_bin = np.linspace(min(Vim_all_values), max(Vim_all_values), 10)
VD_all_values = [value for sublist in VD_all.values() for value in sublist]
Vde_bin = np.linspace(0, max(VD_all_values), 10)
Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
Vimde_bin = np.linspace(min(Vimde_all_values), max(Vimde_all_values), 10)
Vsal_all_values = [value for sublist in Vsal_all.values() for value in sublist] + [value for sublist in Vrep_all.values() for value in sublist]
Vsal_bin = np.linspace(min(Vsal_all_values), max(Vsal_all_values), 10)
Thetaimde_all_values = [value for sublist in Theta_all.values() for value in sublist] + [value for sublist in ThetaD_all.values() for value in sublist]
Thetaimde_bin = np.linspace(min(Thetaimde_all_values), max(Thetaimde_all_values), 10)
Thetaim_all_values = [value for sublist in Theta_all.values() for value in sublist]
Thetaim_bin = np.linspace(min(Thetaim_all_values), max(Thetaim_all_values), 10)
matched_Vim, matched_thetaim, matched_NE, matched_UE, matched_EE, matched_thetaE = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
for i in range (25):
    impact_ejection_list=module.match_ejection_to_impact(impact_list[i], ejection_list[i], dt)
    # impact_ejection_list=module.match_ejection_to_impactanddeposition(impact_deposition_list[i], ejection_list[i])
    matched_Vim[i] = [element for element in impact_ejection_list[0]]
    matched_thetaim[i] = [element for element in impact_ejection_list[1]]
    matched_NE[i] = [element for element in impact_ejection_list[2]]
    matched_UE[i] = [element for element in impact_ejection_list[3]]
    matched_EE[i] = [element for element in impact_ejection_list[4]]
    matched_thetaE[i] = [element for element in impact_ejection_list[5]]
    
constant = np.sqrt(9.81*D)  
#combine the values from all Shields numbers
Vim_all_Omega, Vre_all_Omega, exz_all_Omega, Thetaim_all_Omega, Thetare_all_Omega, VD_all_Omega, ThetaD_all_Omega, matched_Vim_Omega, matched_Thetaim_Omega, matched_NE_Omega, matched_UE_Omega, matched_EE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),defaultdict(list),defaultdict(list)
Vsal_all_Omega, Vrep_all_Omega = defaultdict(list), defaultdict(list)
ThetaE_all_Omega, UE_all_Omega = defaultdict(list), defaultdict(list)
matched_thetaE_Omega = defaultdict(list)
for i in range (5): #loop over Omega 0-20 %
    selected_indices = list(range(i, 25, 5))  # Get indices like [0,5,10,15,20], [1,6,11,16,21], etc.
    Vim_all_Omega[i] = np.concatenate([Vim_all[j] for j in selected_indices]).tolist()
    Vre_all_Omega[i] = np.concatenate([Vre_all[j] for j in selected_indices]).tolist()
    Vsal_all_Omega[i] = np.concatenate([Vsal_all[j] for j in selected_indices]).tolist()
    Vrep_all_Omega[i] = np.concatenate([Vrep_all[j] for j in selected_indices]).tolist()
    exz_all_Omega[i] = np.concatenate([exz_all[j] for j in selected_indices]).tolist()
    Thetaim_all_Omega[i] = np.concatenate([Theta_all[j] for j in selected_indices]).tolist()
    Thetare_all_Omega[i] = np.concatenate([Thetare_all[j] for j in selected_indices]).tolist()
    VD_all_Omega[i] = np.concatenate([VD_all[j] for j in selected_indices]).tolist()
    ThetaD_all_Omega[i] = np.concatenate([ThetaD_all[j] for j in selected_indices]).tolist()
    ThetaE_all_Omega[i] = np.concatenate([ThetaE_all[j] for j in selected_indices]).tolist()
    UE_all_Omega[i] = np.concatenate([UE_all[j] for j in selected_indices]).tolist()
    matched_Vim_Omega[i] = np.concatenate([matched_Vim[j] for j in selected_indices]).tolist()
    matched_Thetaim_Omega[i] = np.concatenate([matched_thetaim[j] for j in selected_indices]).tolist()
    matched_NE_Omega[i] = np.concatenate([matched_NE[j] for j in selected_indices]).tolist()
    matched_UE_Omega[i] = np.concatenate([matched_UE[j] for j in selected_indices]).tolist()
    matched_EE_Omega[i] = np.concatenate([matched_EE[j] for j in selected_indices]).tolist()
    matched_thetaE_Omega[i] = np.concatenate([matched_thetaE[j] for j in selected_indices]).tolist()

CORmean_Omega,N_COR_Omega,Uimplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list)
Uremean_Omega, Urestderr_Omega, Ure_Uim_Omega, Usal_Omega, Uimxplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
COR_theta_mean_Omega, COR_theta_std_Omega, Thetaremean_Omega, Thetarestderr_Omega, Thetaimplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
Uim_mean_Omega, Uim_stderr_Omega, UD_mean_Omega, UD_stderr_Omega, Usal_meanplot = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
Pr_Omega,Uplot_Omega, N_PrUre, Uim_meaninbin, UD_meaninbin = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
Pr_theta_Omega,Theta_pr_Omega = defaultdict(list), defaultdict(list)
NE_mean_Omega, UE_mean_Omega, UE_std_Omega, UE_stderr_Omega, Uplot_NE_Omega, N_Einbin = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),
ThetaE_mean_Omega, ThetaE_stderr_Omega = defaultdict(list), defaultdict(list)
NE_theta_mean_Omega, UE_theta_mean_Omega, UE_theta_std_Omega, Thetaplot_NE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
EE_mean_Omega, EE_stderr_Omega = defaultdict(list), defaultdict(list)
for i in range (5): #loop over Omega 0-20%
    CORmean_Omega[i], N_COR_Omega[i], Thetaremean_Omega[i], Thetarestderr_Omega[i],Uimplot_Omega[i] = module.BinUimCOR_equalbinsize(Vim_all_Omega[i],Vre_all_Omega[i],Thetare_all_Omega[i],Vim_bin)
    Uim_mean_Omega[i], Uim_stderr_Omega[i], UD_mean_Omega[i], UD_stderr_Omega[i], Usal_meanplot[i] = module.BinUincUsal(Vim_all_Omega[i], VD_all_Omega[i], Vsal_all_Omega[i], Vrep_all_Omega[i], Thetaim_all_Omega[i], ThetaD_all_Omega[i], Vsal_bin)
    Pr_Omega[i],Uplot_Omega[i],N_PrUre[i], Uim_meaninbin[i], UD_meaninbin[i] = module.BinUimUd_equalbinsize(Vim_all_Omega[i],VD_all_Omega[i],Vimde_bin)   
    NE_mean_Omega[i], UE_mean_Omega[i], UE_std_Omega[i], UE_stderr_Omega[i], ThetaE_mean_Omega[i], ThetaE_stderr_Omega[i], Uplot_NE_Omega[i], N_Einbin[i]=module.get_ejection_ratios_equalbinsize(matched_Vim_Omega[i], VD_all_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], matched_thetaE_Omega[i], Vimde_bin)#matched_EE_Omega[i]
    # COR_theta_mean_Omega[i], COR_theta_std_Omega[i], Thetaimplot_Omega[i] = module.BinThetaimCOR_equalbinsize(Theta_all_Omega[i], exz_all_Omega[i], Thetaim_bin)
    # Pr_theta_Omega[i],Theta_pr_Omega[i] = module.BinThetaimThetad_equalbinsize(Theta_all_Omega[i],ThetaD_all_Omega[i],Thetaimde_bin)   
    # NE_theta_mean_Omega[i], UE_theta_mean_Omega[i], UE_theta_std_Omega[i], Thetaplot_NE_Omega[i]=module.get_ejection_theta_equalbinsize(matched_Thetaim_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], Thetaim_bin)

# #global mean NE and UE
# def average(matched_NE_Omega):
#     matched_NE_Omega = np.array(matched_NE_Omega)  # Convert to NumPy array
#     return np.mean(matched_NE_Omega)

# def mean_std_nonempty_sublists(matched_UE_Omega):
#     # Flatten the non-empty sublists into a single list
#     all_values = [val for sublist in matched_UE_Omega if sublist for val in sublist]
    
#     if all_values:  # Check if the list is not empty
#         return np.mean(all_values), np.std(all_values)
#     else:
#         return None, None  # Return None if no valid values exist
    
# def mean_std_of_nonempty_sums(nested_list):
#     # Calculate sums of non-empty sublists
#     sublist_sums = [sum(sublist) for sublist in nested_list if sublist]
#     if not sublist_sums:
#         return None, None  # Return None if there are no non-empty lists
#     mean_val = np.mean(sublist_sums)
#     std_val = np.std(sublist_sums)
#     return mean_val, std_val
    
# #calculate the global mean NE, UE and EE
# NE_mean_glo,UE_mean_glo,UE_std_glo, EE_mean_glo, EE_std_glo = [0 for _ in range(5)],[0 for _ in range(5)],[0 for _ in range(5)], [0 for _ in range(5)], [0 for _ in range(5)]
# for i in range(5):
#     NE_mean_glo[i] = average(matched_NE_Omega[i])
#     mean_value, std_value = mean_std_nonempty_sublists(matched_UE_Omega[i])
#     UE_mean_glo[i] = mean_value
#     UE_std_glo[i] = std_value
#     EE_mean_glo[i], EE_std_glo[i] = mean_std_of_nonempty_sums(matched_EE_Omega[i])

def power_law(Omega, A, B, n):
    return A - B * Omega **n
# # Fit the power law model to the data
Omega_tbfit = np.array(Omega, dtype=float)*0.01
# params, covariance = curve_fit(power_law, Omega_tbfit, NE_mean_glo, p0=[1, 1, 0.5])
# a, b, n = params
# print(f'NE={a:.2f} - {b:.2f}*Omega**{n:.2f}')
# # Generate fitted values for plotting
# Omega_fit = np.linspace(min(Omega_tbfit), max(Omega_tbfit), 100)
# NE_fit = power_law(Omega_fit, *params)
# #calculate R^2
# # Create interpolator from 100-point fit
# interpolator = interp1d(Omega_fit, NE_fit, kind='linear', fill_value='extrapolate')
# # Evaluate fit at the same x-values as Uthetaplot
# NE_fit_resampled = interpolator(Omega_tbfit)
# r2_NE = r2_score(NE_mean_glo, NE_fit_resampled)
# print('r2_NE:',r2_NE)

def saturating_exp(x, A, B, C):
    return A * (1 - np.exp(-B * x)) + C
# # Fit the curve
# params, _ = curve_fit(saturating_exp, Omega_tbfit, UE_mean_glo/constant, bounds=(0, np.inf))
# A, B, C = params
# print(f"UE = {A:.2f}*(1- exp(-{B:.2f}*Omega)) + {C:.2f}")
# # Generate smooth fit line
# UE_fit = saturating_exp(Omega_fit, *params)

# #calculate R^2
# # Create interpolator from 100-point fit
# interpolator = interp1d(Omega_fit, UE_fit, kind='linear', fill_value='extrapolate')
# # Evaluate fit at the same x-values as Uthetaplot
# UE_fit_resampled = interpolator(Omega_tbfit)
# r2_UE = r2_score(UE_mean_glo/constant, UE_fit_resampled)
# print('r2_UE:',r2_UE)

# # plt.figure(figsize=(12,5))
# # plt.subplot(1,2,1)
# # plt.plot(Omega, NE_mean_glo, 'o', color='#3776ab', label='Simulated data')
# # plt.plot(Omega_fit*100, NE_fit, '--', color='#3776ab', label='Fit')
# # plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# # plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
# # plt.legend(fontsize=12)
# # plt.subplot(1,2,2)
# # plt.errorbar(Omega, UE_mean_glo/constant, yerr=UE_std_glo/constant, fmt='o', capsize=5, color='#3776ab')
# # plt.plot(Omega_fit*100, UE_fit, '--', color='#3776ab')
# # plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# # plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
# # plt.tight_layout()

# # plt.figure()
# # plt.errorbar(Omega, EE_mean_glo, yerr=EE_std_glo, fmt='o', capsize=5, color='#3776ab')
# # plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# # plt.ylabel(r'$\bar{E}_\mathrm{E}$ [$kgm^2/s^2$]', fontsize=14)
# # plt.ylim(0,0.012)
 
# #global mean and std show weak sensitivity of e with Omega
# # e_mean_glo = np.array([np.mean(v) for v in exz_all_Omega.values()])
# # e_std_glo = np.array([np.std(v) for v in exz_all_Omega.values()])
# # plt.figure()
# # plt.errorbar(Omega, e_mean_glo, yerr=e_std_glo, fmt='o', capsize=5, color='#3776ab')
# # plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# # plt.ylabel(r'$e$ [-]', fontsize=14)
# # plt.tight_layout()

# #compare with static-bed experiment of Ge (2024)
# #try filtering the CORs with Vim=4 m/s (80)
# Theta_Ge = [7, 9, 12, 19]
# COR_Ge = [0.62, 0.57, 0.49, 0.47]
# Theta_GeWet1 = [11, 14, 18]#1.45%
# COR_GeWet1 = [0.65, 0.625, 0.525]
# Theta_GeWet2 = [7, 13, 21]#22.79%
# COR_GeWet2 = [0.69, 0.67, 0.55]
# # Select the Omegas you want to work with
# selected_indices = [0, 4]
# # Collect valid Vim values from both datasets
# Thetaim_validation = []
# for i in selected_indices:
#     Thetaim_validation += [v for v, t in zip(Thetaim_all_Omega[i],Vim_all_Omega[i]) if 3.5 <= t <= 4.5]
# # Define the validation range
# Thetaim_validation_range = np.linspace(min(Thetaim_validation), max(Thetaim_validation), 10)
# COR_test, N_COR_test, Theta_test = defaultdict(list), defaultdict(list), defaultdict(list)
# for i in selected_indices:
#     valid = [(theta, Uim, Ure) for theta, Uim, Ure, t in zip(Thetaim_all_Omega[i], Vim_all_Omega[i], Vre_all_Omega[i], Vim_all_Omega[i]) if 3.5 <= t <= 4.5]
#     Thetaim_valid, Uim_valid, Ure_valid = zip(*valid)
#     COR_test[i], N_COR_test[i], Theta_test[i] = module.BinThetaimCOR_equalbinsize(Thetaim_valid, Uim_valid, Ure_valid, Thetaim_validation_range)

# #try filtering the CORs with thetaim=11.5 degree
# U_Ge = [125, 225, 310]
# UE_Ge = [0.49, 0.52, 0.75]
# U_GeWet = [60, 145]#22.79%
# UE_GeWet = [0.7, 2.7]
# NE_test, UE_test, UE_test_std, UE_test_stderr, U_testNE, N_EinbinTest = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)

# # Collect valid Vim values from both datasets
# Vim_validation,VD_validation = [],[]
# for i in selected_indices:
#     Vim_validation += [v for v, t in zip(matched_Vim_Omega[i], matched_Thetaim_Omega[i]) if 10 <= t <= 13]
#     VD_validation += [v for v, t in zip(VD_all_Omega[i], ThetaD_all_Omega[i]) if 10 <= t <= 13]
# # Define the validation range
# Vim_validation_range = np.linspace(min(Vim_validation + VD_validation), max(Vim_validation + VD_validation), 10)
# # Compute NE_test, UE_test, UE_test_std, U_testNE for selected indices
# for i in selected_indices:
#     valid = [(vim, vD, ne, ue, thetae) for vim, vD, ne, ue, thetae, t in zip(matched_Vim_Omega[i], VD_all_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], matched_thetaE_Omega[i], matched_Thetaim_Omega[i]) if 10 <= t <= 13]
#     Vim_valid, VD_valid, NE_valid, UE_valid, ThetaE_valid = zip(*valid)
#     NE_test[i], UE_test[i], UE_test_std[i], UE_test_stderr[i], thetaEtest, thetaEtest_stderr,U_testNE[i], N_EinbinTest[i] = module.get_ejection_ratios_equalbinsize(Vim_valid, VD_valid, NE_valid, UE_valid, ThetaE_valid, Vim_validation_range)


# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# for i in selected_indices:
#     plt.scatter(Theta_test[i], COR_test[i], s=np.sqrt(N_COR_Omega[i])*5, label=f'$\Omega$={Omega[i]}% (this study)',color=colors[i])
# plt.scatter(Theta_Ge, COR_Ge, marker='d', facecolors='none', edgecolors='k', s=100, label=r'$\Omega$=0$\%$ (Ge et al., 2024)')
# # plt.plot(Theta_GeWet1, COR_GeWet1, '*k', label=r'$\Omega$=1.45$\%$ (Ge et al., 2024)')
# plt.scatter(Theta_GeWet2, COR_GeWet2, marker='s', facecolors='none', edgecolors='k', s=100, label=r'$\Omega$=22.79$\%$ (Ge et al., 2024)')
# plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
# plt.ylabel(r'$\bar{e}$ [-]', fontsize=14)
# plt.ylim(0,1.2)
# plt.xlim(0, 27)
# plt.legend(fontsize=12)
# plt.text(0.03, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(1,2,2)
# for i in selected_indices:
#     plt.errorbar(U_testNE[i]/constant, UE_test[i]/constant, yerr=UE_test_stderr[i]/constant, fmt='o',capsize=5, label=f'$\Omega$={Omega[i]}% (this study)',color=colors[i])
# plt.scatter(U_Ge, UE_Ge/np.sqrt(9.81*0.0003), marker='d', facecolors='none', edgecolors='k', s=100, label=r'$\Omega$=0$\%$ (Ge et al., 2024)')
# plt.scatter(U_GeWet, UE_GeWet/np.sqrt(9.81*0.0003), marker='s', facecolors='none', edgecolors='k', s=100, label=r'$\Omega$=22.79$\%$ (Ge et al., 2024)')
# # plt.plot(np.linspace(0,375,100), np.exp(-1.48+0.082*np.linspace(0,375,100)*constant-0.003*np.radians(11.5))/constant, 'k-', label=r'$\Omega$=0$\%$ (Chen et al., 2019)')
# plt.xlabel(r'$U_{inc}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylim(0,52)
# plt.xlim(0,375)
# plt.legend(fontsize=12)
# plt.text(0.03, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.tight_layout()

#fitting functions
def power_fit(U, a, b):
    return a*U**b
# # def fit_sin_theta(Uim_over_sqrtgd, A, B):
# #     return A / (Uim_over_sqrtgd - B)
def fit_arcsin(Uim_over_sqrtgd, A, B):
    return np.arcsin(A / (Uim_over_sqrtgd + B))

# def log(Omega, a_A0, b_A0):
#     return a_A0 * np.log(1 + b_A0 * Omega)
def log(U, a, b):
    return a*np.log(b*U + 1)

def exp(U, a, b):
    return a*(1 - np.exp(-b*U))

#weighted_R2 functions
def weighted_r2(y_true, y_pred, weights):
    y_avg = np.average(y_true, weights=weights)
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - y_avg)**2)
    return 1 - ss_res / ss_tot

def weighted_r2_N(y_true, y_pred, N):
    # 计算加权平均值
    w = 1 / np.sqrt(N)
    weighted_mean = np.sum(w * y_true) / np.sum(w)
    # 计算加权残差平方和
    weighted_residual_sum_of_squares = np.sum(w * (y_true - y_pred)**2)
    # 计算加权总变差
    weighted_total_sum_of_squares = np.sum(w * (y_true - weighted_mean)**2)
    # 计算加权R^2
    r2 = 1 - (weighted_residual_sum_of_squares / weighted_total_sum_of_squares)
    return r2

# # theta = f(Uim)
# mean_thetaim, stderr_thetaim, Uthetaplot, N_Uim = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
# mean_thetaD, stderr_thetaD, UthetaDplot, N_UD = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
# for i in range(5):
#     mean_thetaim[i], stderr_thetaim[i], Uthetaplot[i], N_Uim[i] = module.match_Uim_thetaim(matched_Vim_Omega[i], matched_Thetaim_Omega[i], Vim_bin)
#     # Vimde_Omega = np.array(Vim_all_Omega[i] + VD_all_Omega[i])
#     # Thetaimde_Omega = np.array(Thetaim_all_Omega[i] + ThetaD_all_Omega[i])
#     mean_thetaD[i], stderr_thetaD[i], UthetaDplot[i], N_UD[i] = module.match_Uim_thetaim(VD_all_Omega[i], ThetaD_all_Omega[i], Vde_bin)

# theta_rad = {key: np.radians(value) for key, value in mean_thetaim.items()}
# theta_rad_stderr = {key: np.radians(value) for key, value in stderr_thetaim.items()}
# U_fit, theta_fit, a_fit, b_fit, c_fit = defaultdict(),defaultdict(),np.zeros(5),np.zeros(5),np.zeros(5) #U_fit, theta_fit=defaultdict(),defaultdict()
# # --- Fit ---
# for i in range(5):
#     # --- Remove NaN values before fitting ---
#     valid_indices = ~np.isnan(mean_thetaim[i])  # Get boolean mask where theta_all is NOT NaN
#     U_clean = Uthetaplot[i][valid_indices]/constant       # Keep only valid U values
#     theta_clean = theta_rad[i][valid_indices]# Keep only valid theta values
#     N_Uim_clean = N_Uim[i][valid_indices]
#     stderr = theta_rad_stderr[i][valid_indices]
#     popt, _ = curve_fit(fit_arcsin, U_clean, theta_clean, p0=[1,0], sigma=stderr, absolute_sigma=True)
#     a_fit[i], b_fit[i] = popt 
#     # Generate fitted curve
#     U_fit[i] = np.linspace(min(U_clean), max(U_clean), 200)
#     theta_fit[i] = fit_arcsin(U_fit[i], *popt)
#     theta_fit_dis = fit_arcsin(U_clean, *popt)
#     R2_weighted = weighted_r2(theta_clean, theta_fit_dis, weights=1 / (stderr ** 2))
#     print('R2_weighted:', R2_weighted)
    

# plt.figure(figsize=(6,5))
# for i in range(5):
#     plt.errorbar(Uthetaplot[i]/constant, mean_thetaim[i], yerr=stderr_thetaim[i], 
#                   fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
# # Plot fitted curve
# for i in range(5):
#     plt.plot(U_fit[i], np.degrees(theta_fit[i]), '--', color=colors[i])
# plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\theta_{im}$ [-]', fontsize=14)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()     

# a_fit_new = np.delete(a_fit, 1)
# b_fit_new = np.delete(b_fit, 1)
# #fit a_fit=f(omega)
Omega_tbfit = np.array(Omega, dtype=float)*0.01
Omega_tbfit_new = np.delete(Omega_tbfit, 1)

# params1, _ = curve_fit(power_law, Omega_tbfit_new, a_fit_new, p0=[65, 20, 0.5])
# # Generate smooth curve for plotting
Omega_smooth = np.linspace(min(Omega_tbfit_new), max(Omega_tbfit_new), 100)
# a_fit_smooth = power_law(Omega_smooth, *params1)
# A1_opt, B1_opt, n1_opt = params1

# plt.figure()
# plt.plot(Omega_tbfit, a_fit, 'o', label= 'Original data')
# plt.plot(Omega_smooth, a_fit_smooth, '--', label='Power-law fit')
# plt.xlabel('omega')
# plt.ylabel('a_fit')

# alpha = A1_opt - B1_opt*Omega_tbfit**n1_opt
# print(rf"$\alpha$ = {A1_opt:.2f} - {B1_opt:.2f}*$\Omega$**{n1_opt:.2f}")

# params2, _ = curve_fit(saturating_exp, Omega_tbfit_new, b_fit_new, p0=[65, 20, 0.5])
# # Generate smooth curve for plotting
# b_fit_smooth = saturating_exp(Omega_smooth, *params2)
# A2_opt, B2_opt, n2_opt = params2

# plt.figure()
# plt.plot(Omega_tbfit, b_fit, 'o', label= 'Original data')
# plt.plot(Omega_smooth, b_fit_smooth, '--', label='Power-law fit')
# plt.xlabel('omega')
# plt.ylabel('b_fit')

# beta = A2_opt * (1 - np.exp(-B2_opt*Omega_tbfit)) + n2_opt
# print(rf"$\beta$ = {A2_opt:.2f} * (1 - exp(-{B2_opt:.2f}*$\Omega$)) + {n2_opt:.2f}")

# # Combine all data from the 5 moisture levels
# U_fit_new = np.linspace(min(Uthetaplot[0]/constant), max(Uthetaplot[0]/constant), 100)
# Theta_fit_new = defaultdict()
# # Loop over each element in alpha and multiply with U_fit_new
# for i in range(5):
#     Theta_fit_new[i] = fit_arcsin(U_fit_new, alpha[i], beta[i])

# #plot the fit using the global function
# # Plot original data
# plt.figure(figsize=(6,5))
# for i in range(5):
#     plt.errorbar(Uthetaplot[i]/constant, mean_thetaim[i], yerr=stderr_thetaim[i], 
#                   fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
# # Plot fitted hyperbolic curve
# for i in range(5):
#     plt.plot(U_fit_new, np.degrees(Theta_fit_new[i]), '--', color=colors[i])
# plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\theta_\mathrm{im}$ [$^\circ$]', fontsize=14)
# plt.xlim(0,225)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()     

# #calculate R^2
# all_theta_ori, all_theta_fit_resampled, weight_theta_all = [],[],[]
# for i in range(5):
#     # Create interpolator from 100-point fit
#     interpolator = interp1d(U_fit_new, Theta_fit_new[i], kind='linear', fill_value='extrapolate')
#     # Evaluate fit at the same x-values as Uthetaplot
#     # --- Remove NaN values before fitting ---
#     valid_indices = ~np.isnan(theta_rad[i])  # Get boolean mask where theta_all is NOT NaN
#     U_clean = Uthetaplot[i][valid_indices]/constant       # Keep only valid U values
#     theta_clean = theta_rad[i][valid_indices]# Keep only valid theta values
#     stderr = theta_rad_stderr[i][valid_indices]
#     theta_fit_resampled = interpolator(U_clean)
#     all_theta_ori.append(theta_clean)
#     all_theta_fit_resampled.append(theta_fit_resampled)
#     weight_theta_all.append(1/(stderr**2))
    
# y_theta_all = np.concatenate(all_theta_ori)
# y_predtheta_all = np.concatenate(all_theta_fit_resampled)
# weight_theta_glo = np.concatenate(weight_theta_all)
# # Now compute R²
# R2_theta = weighted_r2(y_theta_all, y_predtheta_all, weights=weight_theta_glo)
# print('R2_theta:',R2_theta)

# UD - thetaD
# plt.figure(figsize=(6,5))
# for i in range(5):
#     plt.errorbar(UthetaDplot[i]/constant, mean_thetaD[i], yerr=stderr_thetaD[i], 
#                   fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
# plt.xlabel(r'$U_{D}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\theta_{D}$ [$^\circ$]', fontsize=14)
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()     

# def fit_arcsin_thetainf(UD, thetainf, A, B):
#     val = A / (UD + B)
#     val = np.clip(val, -0.9999, 0.9999)
#     return thetainf * np.arcsin(val)

# thetaD_tbfit = list(chain.from_iterable(mean_thetaD.values()))
# UD_tbfit = list(chain.from_iterable(UthetaDplot.values()))
# valid_indices = ~np.isnan(np.array(thetaD_tbfit))
# UD_tbfit_new = np.array(UD_tbfit)[valid_indices]/constant
# thetaD_tbfit_new = np.deg2rad(np.array(thetaD_tbfit))[valid_indices]
# thetaDstderr_tbfit = list(chain.from_iterable(stderr_thetaD.values()))
# thetaDstderr_tbfit_new = np.deg2rad(np.array(thetaDstderr_tbfit))[valid_indices]
# valid_mask = thetaDstderr_tbfit_new != 0
# UD_tbfit_new = UD_tbfit_new[valid_mask]
# thetaD_tbfit_new = thetaD_tbfit_new[valid_mask]
# thetaDstderr_tbfit_new = thetaDstderr_tbfit_new[valid_mask]
# params_thetaD, _ = curve_fit(fit_arcsin_thetainf, UD_tbfit_new, thetaD_tbfit_new, p0=[0.4, 20, 5], 
#                              sigma=thetaDstderr_tbfit_new, absolute_sigma=True, maxfev=10000)
# thetainf, A_thetaD, B_thetaD = params_thetaD
# print(f'thetainf={thetainf:2f}, A_thetaD={A_thetaD:.4f} B_thetaD={B_thetaD:.4f}')

# UD_fit = np.linspace(min(UD_tbfit_new), max(UD_tbfit_new), 100) #for the fit
# thetaD_fit = fit_arcsin_thetainf(UD_fit, *params_thetaD)

# #calculate R^2
# interpolator = interp1d(UD_fit, thetaD_fit, kind='linear', fill_value='extrapolate')
# thetaD_fit_resampled = interpolator(UD_tbfit_new)
# # Now compute R²
# r2_thetaD = weighted_r2(thetaD_tbfit_new, thetaD_fit_resampled, weights=1/thetaDstderr_tbfit_new**2)
# print('r2_thetaD:',r2_thetaD)

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# for i in range(5):
#     plt.errorbar(Uthetaplot[i]/constant, mean_thetaim[i], yerr=stderr_thetaim[i], 
#                   fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
# # Plot fitted curve
# for i in range(5):
#     plt.plot(U_fit[i], np.degrees(theta_fit[i]), '--', color=colors[i])
# plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\theta_{im}$ [$^\circ$]', fontsize=14)
# plt.ylim(0,35)
# plt.legend(fontsize=12)
# plt.text(0.03, 0.94, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(1,2,2)
# for i in range(5):
#     plt.errorbar(UthetaDplot[i]/constant, mean_thetaD[i], yerr=stderr_thetaD[i], 
#                   fmt='o', capsize=5, color=colors[i])
# plt.plot(UD_fit, np.degrees(thetaD_fit), 'k--')
# plt.xlim(left=0)
# plt.ylim(0,35)
# plt.xlabel(r'$U_{D}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\theta_{D}$ [$^\circ$]', fontsize=14)
# plt.text(0.03, 0.94, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()

# Uim, UD - Usal
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# for i in range(5):
#     plt.errorbar(Usal_meanplot[i]/constant, Uim_mean_Omega[i]/constant, yerr=Uim_stderr_Omega[i]/constant, 
#                   fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
# plt.xlabel(r'$U_\mathrm{sal}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.legend(fontsize=12)
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.subplot(1,2,2)
# for i in range(5):
#     plt.errorbar(Usal_meanplot[i]/constant, UD_mean_Omega[i]/constant, yerr=UD_stderr_Omega[i]/constant, 
#                   fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
# plt.xlabel(r'$U_\mathrm{sal}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{D}/\sqrt{gd}$ [-]', fontsize=14)
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.tight_layout()
# plt.show()     

# Uim_fit = defaultdict(list)
# a_Uim, b_Uim = np.zeros(5), np.zeros(5)
# for i in range(5):
#     # --- Remove NaN values before fitting ---
#     valid_indices = ~np.isnan(Uim_mean_Omega[i])  # Get boolean mask where theta_all is NOT NaN
#     Usal_clean = Usal_meanplot[i][valid_indices]/constant       # Keep only valid U values
#     Uim_clean = Uim_mean_Omega[i][valid_indices]/constant # Keep only valid COR values
#     Uimstderr_clean = Uim_stderr_Omega[i][valid_indices]/constant
#     popt, _ = curve_fit(power_fit, Usal_clean, Uim_clean, p0=[1,1], sigma=Uimstderr_clean, absolute_sigma=True)
#     a_Uim[i], b_Uim[i] = popt 
#     # Generate fitted curve
#     Usalim_fit = np.linspace(0, max(Usal_clean), 200)
#     Uim_fit[i] = power_fit(Usalim_fit, *popt)
#     Uim_fit_dis = power_fit(Usal_clean, *popt)
#     R2 = weighted_r2_N(Uim_clean, Uim_fit_dis, 1/Uimstderr_clean**2)
#     print('R2_Uim:', R2)
    
# UD_fit = defaultdict(list)
# a_UD, b_UD = np.zeros(5), np.zeros(5)
# for i in range(5):
#     # --- Remove NaN values before fitting ---
#     valid_indices = ~np.isnan(UD_mean_Omega[i])  # Get boolean mask where theta_all is NOT NaN
#     Usal_clean = Usal_meanplot[i][valid_indices]/constant       # Keep only valid U values
#     UD_clean = UD_mean_Omega[i][valid_indices]/constant # Keep only valid COR values
#     UDstderr_clean = UD_stderr_Omega[i][valid_indices]/constant
#     valid_mask = UDstderr_clean != 0
#     Usal_clean = Usal_clean[valid_mask]
#     UD_clean = UD_clean[valid_mask]
#     stderr = UDstderr_clean[valid_mask]
#     popt, _ = curve_fit(power_fit, Usal_clean, UD_clean, p0=[1,1], sigma=stderr, absolute_sigma=True)
#     a_UD[i], b_UD[i] = popt 
#     # Generate fitted curve
#     UsalD_fit = np.linspace(0, max(Usal_meanplot[i][valid_indices]/constant), 200)
#     UD_fit[i] = power_fit(UsalD_fit, *popt)
#     UD_fit_dis = power_fit(Usal_clean, *popt)
#     R2 = weighted_r2_N(UD_clean, UD_fit_dis, 1/stderr**2)
#     print('R2_UD:', R2)
    
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# for i in range(5):
#     plt.errorbar(Usal_meanplot[i]/constant, Uim_mean_Omega[i]/constant, yerr=Uim_stderr_Omega[i]/constant, 
#                   fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
#     plt.plot(Usalim_fit, Uim_fit[i], '--', color=colors[i])
# plt.xlabel(r'$\bar{U}_\mathrm{sal}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\bar{U}_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.legend(fontsize=12)
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.subplot(1,2,2)
# for i in range(5):
#     plt.errorbar(Usal_meanplot[i]/constant, UD_mean_Omega[i]/constant, yerr=UD_stderr_Omega[i]/constant, 
#                   fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
#     plt.plot(UsalD_fit, UD_fit[i], '--', color=colors[i])
# plt.xlabel(r'$\bar{U}_\mathrm{sal}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\bar{U}_\mathrm{D}/\sqrt{gd}$ [-]', fontsize=14)
# plt.legend(fontsize=12)
# plt.xlim(left=0)
# plt.ylim(bottom=0)   
# plt.tight_layout()
# plt.show()      

# plt.figure(figsize=(6,5))
# for i in range(5):
#     plt.errorbar(Uincplot[i]/constant, Usal_mean_Omega[i]/constant, yerr=Usal_stderr_Omega[i]/constant, 
#                   fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
# plt.plot(Uinc_fit, Usal_fit, 'k--', label='Power-law fit')
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{sal}/\sqrt{gd}$ [-]', fontsize=14)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()

# binned Uinc - Uim & UD
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# for i in range(5):
#     plt.scatter(Uplot_Omega[i]/constant, Uim_meaninbin[i]/constant, s=np.sqrt(N_PrUre[i])*5,color=colors[i])
#     plt.scatter(Uplot_Omega[i]/constant, UD_meaninbin[i]/constant, s=np.sqrt(N_PrUre[i])*5, marker='x',color=colors[i])
# # Add one 'o' and one 'x' marker to legend manually
# plt.scatter([], [], marker='o', color='black', label=r"$U_\mathrm{im}/\sqrt{gd}$")
# plt.scatter([], [], marker='x', color='black', label=r"$U_\mathrm{D}/\sqrt{gd}$")
# plt.xlim(0,210)
# plt.ylim(0,220)
# plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{im}/\sqrt{gd}$, $U_\mathrm{D}/\sqrt{gd}$ [-]', fontsize=14)
# plt.legend(fontsize=12)
# plt.subplot(1,2,2)
# for i in range(5):
#     plt.scatter(Uplot_Omega[i]/constant, UD_meaninbin[i]/Uim_meaninbin[i], s=np.sqrt(N_PrUre[i])*5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
# plt.xlim(0,210)
# plt.ylim(bottom=0)
# plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{D}/U_\mathrm{im}$ [-]', fontsize=14)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.show()

# distribution
# deposition velocity
# def compute_percentiles_by_omega(data_dict, percentiles=[10, 25, 50, 75, 90]):
#     omega_values = sorted(data_dict.keys())
#     percentile_data = []
#     for omega in omega_values:
#         data = np.array(data_dict[omega])
#         percentiles_for_omega = [np.percentile(data, p) for p in percentiles]
#         percentile_data.append(percentiles_for_omega)
#     return percentile_data
    
# # VD_mean_glo = np.array([np.mean(v) for v in VD_all_Omega.values()])
# # VD_std_glo = np.array([np.std(v) for v in VD_all_Omega.values()])

# percentiles=[10, 25, 50, 75, 90]
# VD_per = compute_percentiles_by_omega(VD_all_Omega, percentiles)
# percentile_VD = np.array(VD_per)
# p10_VD, p25_VD, p50_VD, p75_VD, p90_VD = percentile_VD.T
# ThetaD_per = compute_percentiles_by_omega(ThetaD_all_Omega, percentiles)
# percentile_ThetaD = np.array(ThetaD_per)
# p10_ThetaD, p25_ThetaD, p50_ThetaD, p75_ThetaD, p90_ThetaD = percentile_ThetaD.T

# plt.figure(figsize=(12, 9))
# # Plot PDF of U_im/sqrt(gd)
# plt.subplot(2, 2, 1)
# for i in range(5):
#     # Calculate histogram (density=True for probability density)
#     counts, bin_edges = np.histogram(VD_all_Omega[i], bins=50, density=True)
#     # Create the step plot
#     plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# plt.xlabel(r'$U_\mathrm{D}$ [m/s]', fontsize=14)
# plt.ylabel('Probability Density [-]', fontsize=14)
# plt.text(0.03, 0.94, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# # plt.ylim(-0.002, 0.03)
# plt.legend(fontsize=12)
# plt.subplot(2, 2, 2)
# for i in range(5):
#     # Calculate histogram (density=True for probability density)
#     counts, bin_edges = np.histogram(ThetaD_all_Omega[i], bins=50, density=True)
#     # Create the step plot
#     plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# plt.xlabel(r'$\Theta_\mathrm{D}$ [$^\circ$]', fontsize=14)
# plt.ylabel('Probability Density [-]', fontsize=14)
# plt.text(0.03, 0.94, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(2, 2, 3)
# plt.errorbar(Omega,p50_VD,yerr=[p50_VD - p25_VD, p75_VD - p50_VD],fmt='o',capsize=5, color='#3776ab')
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{D}$ [m/s]', fontsize=14)
# plt.text(0.03, 0.94, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(2, 2, 4)
# plt.errorbar(Omega,p50_ThetaD,yerr=[p50_ThetaD - p25_ThetaD, p75_ThetaD - p50_ThetaD],fmt='o',capsize=5, color='#3776ab')
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$\Theta_\mathrm{D}$ [$^\circ$]', fontsize=14)
# plt.text(0.03, 0.94, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.tight_layout()

# distribution of theta and Uim
# percentiles=[10, 25, 50, 75, 90]
# Vim_per = compute_percentiles_by_omega(Vim_all_Omega, percentiles)
# percentile_Vim = np.array(Vim_per)
# p10_Vim, p25_Vim, p50_Vim, p75_Vim, p90_Vim = percentile_Vim.T
# Thetaim_per = compute_percentiles_by_omega(Thetaim_all_Omega, percentiles)
# percentile_thetaim = np.array(Thetaim_per)
# p10_thetaim, p25_thetaim, p50_thetaim, p75_thetaim, p90_thetaim = percentile_thetaim.T

# plt.figure(figsize=(12, 9))
# # Plot PDF of U_im/sqrt(gd)
# plt.subplot(2, 2, 1)
# for i in range(5):
#     # Calculate histogram (density=True for probability density)
#     counts, bin_edges = np.histogram(Vim_all_Omega[i], bins=50, density=True)
#     plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# # UD
# for i in range(5):
#     counts, bin_edges = np.histogram(VD_all_Omega[i], bins=50, density=True)
#     plt.step(bin_edges[:-1], counts, where='mid', linestyle='--', color=colors[i])
# plt.plot([], [], color='black', label=r"$U_\mathrm{im}$")
# plt.plot([], [], '--', color='black', label=r"$U_\mathrm{D}$")
# plt.xlabel(r'$U_\mathrm{inc}$ [m/s]', fontsize=14)
# plt.ylabel('Probability Density [-]', fontsize=14)
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend(fontsize=12)
# plt.text(0.03, 0.94, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# # Plot PDF of theta_im
# plt.subplot(2, 2, 2)
# for i in range(5):
#     # Calculate histogram (density=True for probability density)
#     counts, bin_edges = np.histogram(Thetaim_all_Omega[i], bins=50, density=True)
#     # Create the step plot
#     plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# for i in range(5):    
#     counts, bin_edges = np.histogram(ThetaD_all_Omega[i], bins=50, density=True)
#     plt.step(bin_edges[:-1], counts, where='mid', linestyle='--', color=colors[i])
# plt.plot([], [], color='black', label=r"$\theta_\mathrm{im}$")
# plt.plot([], [], '--', color='black', label=r"$\theta_\mathrm{D}$")
# plt.legend(fontsize=12)
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.xlabel(r'$\theta_\mathrm{inc}$ [$^\circ$]', fontsize=14)
# plt.ylabel('Probability Density [-]', fontsize=14)
# plt.text(0.03, 0.94, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(2, 2, 3)
# # plt.errorbar(Omega, Vim_mean_glo, yerr=Vim_std_glo, fmt='o', capsize=5, color='#3776ab')
# plt.errorbar(Omega,p50_Vim,yerr=[p50_Vim - p25_Vim, p75_Vim - p50_Vim],fmt='ko',capsize=5, label=r'$U_\mathrm{im}$')
# plt.errorbar(Omega,p50_VD,yerr=[p50_VD - p25_VD, p75_VD - p50_VD],fmt='o',capsize=5, label=r'$U_\mathrm{D}$')
# plt.legend(loc='upper right', fontsize=12)
# plt.ylim(0,5.5)
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{inc}$ [m/s]', fontsize=14)
# plt.text(0.03, 0.94, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(2, 2, 4)
# # plt.errorbar(Omega, Thetaim_mean_glo, yerr=Thetaim_std_glo, fmt='o', capsize=5, color='#3776ab')
# plt.errorbar(Omega,p50_thetaim,yerr=[p50_thetaim - p25_thetaim, p75_thetaim - p50_thetaim],fmt='ko',capsize=5, label=r'$\theta_\mathrm{im}$')
# plt.errorbar(Omega,p50_ThetaD,yerr=[p50_ThetaD - p25_ThetaD, p75_ThetaD - p50_ThetaD],fmt='o',capsize=5, label=r'$\theta_\mathrm{D}$')
# plt.ylim(8,48)
# plt.legend(loc='upper right', fontsize=12)
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$\theta_\mathrm{inc}$ [$\circ$]', fontsize=14)
# plt.text(0.03, 0.94, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()

# print(f'UD is {np.mean(p50_VD/p50_Vim):.2f} of Uim')

#distribution of theta_re and theta_E
# Thetare_per = compute_percentiles_by_omega(Thetare_all_Omega, percentiles)
# percentile_thetare = np.array(Thetare_per)
# p10_thetare, p25_thetare, p50_thetare, p75_thetare, p90_thetare = percentile_thetare.T

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# for i in range(5):
#     # Calculate histogram (density=True for probability density)
#     counts, bin_edges = np.histogram(Thetare_all_Omega[i], bins=50, density=True)
#     # Create the step plot
#     plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# # plt.ylim(-0.01,0.13)
# plt.xlabel(r'$\theta_\mathrm{re}$ [$^\circ$]', fontsize=14)
# plt.ylabel('Probability Density [-]', fontsize=14)
# plt.text(0.03, 0.94, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(1, 2, 2)
# plt.errorbar(Omega,p50_thetare,yerr=[p50_thetare - p25_thetare, p75_thetare - p50_thetare],fmt='o',capsize=5, color='#3776ab')
# plt.ylim(13, 55)
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$\theta_\mathrm{re}$ [$\circ$]', fontsize=14)
# plt.text(0.03, 0.94, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()

# ThetaE_per = compute_percentiles_by_omega(ThetaE_all_Omega, percentiles)
# percentile_thetaE = np.array(ThetaE_per)
# p10_thetaE, p25_thetaE, p50_thetaE, p75_thetaE, p90_thetaE = percentile_thetaE.T

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# for i in range(5):
#     # Calculate histogram (density=True for probability density)
#     counts, bin_edges = np.histogram(ThetaE_all_Omega[i], bins=50, density=True)
#     # Create the step plot
#     plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# # plt.ylim(-0.01,0.13)
# plt.xlabel(r'$\theta_\mathrm{E}$ [$^\circ$]', fontsize=14)
# plt.ylabel('Probability Density [-]', fontsize=14)
# plt.text(0.03, 0.94, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(1, 2, 2)
# plt.errorbar(Omega,p50_thetaE,yerr=[p50_thetaE - p25_thetaE, p75_thetaE - p50_thetaE],fmt='o',capsize=5, color='#3776ab')
# plt.ylim(22, 75)
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$\theta_\mathrm{E}$ [$\circ$]', fontsize=14)
# plt.text(0.03, 0.94, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()

# ThetaD_per = compute_percentiles_by_omega(ThetaD_all_Omega, percentiles)
# percentile_thetaD = np.array(ThetaD_per)
# p10_thetaD, p25_thetaD, p50_thetaD, p75_thetaD, p90_thetaD = percentile_thetaD.T

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# for i in range(5):
#     # Calculate histogram (density=True for probability density)
#     counts, bin_edges = np.histogram(ThetaD_all_Omega[i], bins=50, density=True)
#     # Create the step plot
#     plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# # plt.ylim(-0.01,0.13)
# plt.xlabel(r'$\theta_\mathrm{D}$ [$^\circ$]', fontsize=14)
# plt.ylabel('Probability Density [-]', fontsize=14)
# plt.text(0.03, 0.94, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(1, 2, 2)
# plt.errorbar(Omega,p50_thetaD,yerr=[p50_thetaD - p25_thetaD, p75_thetaD - p50_thetaD],fmt='o',capsize=5, color='#3776ab')
# # plt.ylim(22, 75)
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$\theta_\mathrm{D}$ [$\circ$]', fontsize=14)
# plt.text(0.03, 0.94, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()


# #4 terms with Uim
# Ure - Uim
plt.figure(figsize=(6,5.5))
for i in range(5):
    plt.scatter(Uimplot_Omega[i]/constant,CORmean_Omega[i], s=np.sqrt(N_COR_Omega[i])*5, color=colors[i], label=f"$\\Omega$={Omega[i]}%")
plt.xlim(left=0)
plt.ylim(0,1)
plt.legend(fontsize=12)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{e}$ [-]', fontsize=14) 
plt.tight_layout()

def GaussianBell(U, a, b, A0, mu, sigma):
    e = a*U**(-b) + A0 * np.exp(-(U - mu)**2 / (2 * sigma**2)) #
    return e

UCOR_fit, COR_fit = defaultdict(list),defaultdict(list)
a_COR, b_COR, A0_COR, mu_COR, sigma_COR = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
for i in range(5):
    # --- Remove NaN values before fitting ---
    valid_indices = ~np.isnan(CORmean_Omega[i])  # Get boolean mask where theta_all is NOT NaN
    U_clean = Uimplot_Omega[i][valid_indices]/constant       # Keep only valid U values
    COR_clean = CORmean_Omega[i][valid_indices] # Keep only valid COR values
    weights = 1/np.sqrt(N_COR_Omega[i][valid_indices])
    popt, _ = curve_fit(GaussianBell, U_clean, COR_clean, p0=[3,0.6,0.2,150,30], sigma=weights, absolute_sigma=True)
    a_COR[i], b_COR[i], A0_COR[i], mu_COR[i], sigma_COR[i] = popt 
    # Generate fitted curve
    UCOR_fit[i] = np.linspace(min(U_clean), max(U_clean), 200)
    COR_fit[i] = GaussianBell(UCOR_fit[i], *popt)
    COR_fit_dis = GaussianBell(U_clean, *popt)
    R2 = weighted_r2_N(COR_clean, COR_fit_dis, N_COR_Omega[i][valid_indices])
    print('R2_NE:', R2)

plt.figure(figsize=(6,5.5))
for i in range(5):
    plt.scatter(Uimplot_Omega[i]/constant,CORmean_Omega[i], s=np.sqrt(N_COR_Omega[i])*5, color=colors[i], label=f"$\\Omega$={Omega[i]}%")
    plt.plot(UCOR_fit[i], COR_fit[i], '--', color=colors[i])
plt.xlim(left=0)
plt.ylim(0,1)
plt.legend(fontsize=12)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{e}$ [-]', fontsize=14) 
plt.tight_layout()

plt.figure()
plt.plot(Omega, a_COR, 'o')
plt.xlabel('omega')
plt.ylabel('a_COR')
print('aCOR=', a_COR[0])

plt.figure()
plt.plot(Omega, b_COR, 'o')
print('bCOR=', b_COR[0])

Omega_tbfit_COR = np.delete(Omega_tbfit,2)
paramsA0, _ = curve_fit(log, Omega_tbfit_COR, np.delete(A0_COR,2), p0=[1, 1])
# Generate smooth curve for plotting
A0_fit_smooth = log(Omega_smooth, *paramsA0)
a_A0_opt, b_A0_opt = paramsA0

plt.figure()
plt.plot(Omega_tbfit, A0_COR, 'o', label= 'Original data')
plt.plot(Omega_smooth, A0_fit_smooth, '--', label='Power-law fit')
plt.xlabel('omega')
plt.ylabel('A0_COR')

A0 = a_A0_opt * np.log(1 + b_A0_opt*Omega_tbfit)
print(rf"$A0$ = {a_A0_opt:.2f} * log(1 + {b_A0_opt:.2f}*Omega)")
    
plt.figure()
plt.plot(Omega, mu_COR, 'o')
mu = np.mean(mu_COR)
print('mu=',mu)

plt.figure()
plt.plot(Omega, sigma_COR, 'o')
sigma = np.mean(sigma_COR)
print('sigma=',sigma)

# 展示拟合结果
# Combine all data from the 5 moisture levels
Ue_fit_new = np.linspace(min(Uimplot_Omega[0]/constant), max(Uimplot_Omega[0]/constant), 100)
e_fit_new = defaultdict()
# Loop over each element in alpha and multiply with U_fit_new
for i in range(5):
    e_fit_new[i] = GaussianBell(Ue_fit_new, a_COR[0], b_COR[0], A0[i], mu, sigma)

#calculate R^2
e_all, e_fit_resampled_all, weight_e_all = [],[],[]
for i in range(5):
    # Create interpolator from 100-point fit
    interpolator = interp1d(Ue_fit_new, e_fit_new[i], kind='linear', fill_value='extrapolate')
    # Evaluate fit at the same x-values as Uthetaplot
    # --- Remove NaN values before fitting ---
    valid_indices = ~np.isnan(CORmean_Omega[i])  # Get boolean mask where theta_all is NOT NaN
    U_clean = Uimplot_Omega[i][valid_indices]/constant       # Keep only valid U values
    e_clean = np.array(CORmean_Omega[i])[valid_indices]# Keep only valid theta values
    e_fit_resampled = interpolator(U_clean)
    e_all.append(e_clean)
    e_fit_resampled_all.append(e_fit_resampled)
    weight_e_all.append(N_COR_Omega[i][valid_indices])
    
y_e_all = np.concatenate(e_all)
y_prede_all = np.concatenate(e_fit_resampled_all)
weight_e_glo = np.concatenate(weight_e_all)
# Now compute R²
R2_e = weighted_r2_N(y_e_all, y_prede_all, weight_e_glo)
print('R2_e:',R2_e)

plt.figure(figsize=(6,5.5))
for i in range(5):
    plt.scatter(Uimplot_Omega[i]/constant,CORmean_Omega[i], s=np.sqrt(N_COR_Omega[i])*5, color=colors[i], label=f"$\\Omega$={Omega[i]}%")
    plt.plot(Ue_fit_new, e_fit_new[i], '--', color=colors[i])
plt.xlim(left=0)
plt.ylim(0,1)
plt.legend(fontsize=12)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{e}$ [-]', fontsize=14) 
plt.tight_layout()

#fit theta_re
def arcsinlinear(U, a, b):
    return np.arcsin(a * U + b)

Uthetare_fit, thetare_fit = defaultdict(list),defaultdict(list)
a_thetare, b_thetare = np.zeros(5), np.zeros(5)
for i in range(5):
    # --- Remove NaN values before fitting ---
    valid_indices = ~np.isnan(Thetaremean_Omega[i])  # Get boolean mask where theta_all is NOT NaN
    U_clean = Uimplot_Omega[i][valid_indices]/constant       # Keep only valid U values
    thetare_clean = np.radians(Thetaremean_Omega[i])[valid_indices] # Keep only valid theta values
    stderr = np.radians(Thetarestderr_Omega[i])[valid_indices]
    popt, _ = curve_fit(arcsinlinear, U_clean, thetare_clean, p0=[-0.001,1], sigma=stderr, absolute_sigma=True)
    a_thetare[i], b_thetare[i] = popt 
    # Generate fitted curve
    Uthetare_fit[i] = np.linspace(min(U_clean), max(U_clean), 200)
    thetare_fit[i] = arcsinlinear(Uthetare_fit[i], *popt)
    thetare_fit_dis = arcsinlinear(U_clean, *popt)
    R2 = weighted_r2(thetare_clean, thetare_fit_dis, weights=1/(stderr**2))
    print('R2_thetare:', R2)

plt.figure()
for i in range(5):
    plt.errorbar(Uimplot_Omega[i]/constant, Thetaremean_Omega[i], yerr=Thetarestderr_Omega[i], fmt='o', capsize=5, color=colors[i])
    plt.plot(Uthetare_fit[i], np.degrees(thetare_fit[i]), '--', color=colors[i])
plt.xlabel(r'$U_{im}/\sqrt{gd}$ []')
plt.ylabel(r'$\theta_{re}$ [$^\circ$]')

paramsthetarea, _ = curve_fit(power_law, Omega_tbfit, a_thetare)
a_thetare_fit_smooth = power_law(Omega_smooth, *paramsthetarea)
A_thetare, B_thetare, n_thetare = paramsthetarea

plt.figure()
plt.plot(Omega_tbfit, a_thetare, 'o')
plt.plot(Omega_smooth, a_thetare_fit_smooth, '--', label='Power-law fit')
plt.xlabel('Omega')
plt.ylabel('a_thetare')

athetare_fit = A_thetare - B_thetare*Omega_tbfit**n_thetare
print(f"athetare_fit = {A_thetare:.4f} - {B_thetare:.4f}*Omega_tbfit**{n_thetare:.2f}")
bthetare_fit = np.mean(b_thetare)
print(f"bthetare_fit = {bthetare_fit:.2f}")

Uthetare_fit_new = np.linspace(0, max(Uimplot_Omega[0]/constant), 100)
thetare_fit_new = defaultdict()
# Loop over each element in alpha and multiply with U_fit_new
for i in range(5):
    thetare_fit_new[i] = arcsinlinear(Uthetare_fit_new, athetare_fit[i], bthetare_fit)
    
#calculate R^2
all_thetare_clean = []
all_thetare_fit_resampled = []
all_weights_thetare = []

for i in range(5):
    # Create interpolator from 100-point fit
    interpolator = interp1d(Uthetare_fit_new, thetare_fit_new[i], kind='linear', fill_value='extrapolate')
    # Remove NaNs
    valid_indices = ~np.isnan(Thetaremean_Omega[i])
    U_clean = Uimplot_Omega[i][valid_indices]/constant       # Keep only valid U values
    thetare_clean = np.radians(Thetaremean_Omega[i])[valid_indices]  # Keep only valid theta values
    stderr = np.radians(Thetarestderr_Omega[i])[valid_indices]
    thetare_fit_resampled = interpolator(U_clean)
    # Store for global R²
    all_thetare_clean.append(thetare_clean)
    all_thetare_fit_resampled.append(thetare_fit_resampled)
    all_weights_thetare.append(1/(stderr**2))
    
y_thetare_all = np.concatenate(all_thetare_clean)
y_predthetare_all = np.concatenate(all_thetare_fit_resampled)
weight_thetare_glo = np.concatenate(all_weights_thetare)
# Now compute R²
R2_thetare = weighted_r2(y_thetare_all, y_predthetare_all, weights=weight_thetare_glo)
print('R2_thetare:',R2_thetare)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for i in range(5):
    plt.scatter(Uimplot_Omega[i]/constant,CORmean_Omega[i], s=np.sqrt(N_COR_Omega[i])*5, color=colors[i], label=f"$\\Omega$={Omega[i]}%")
    plt.plot(Ue_fit_new, e_fit_new[i], '--', color=colors[i])
# plt.plot(Ue_fit_new, 0.7469*np.exp(0.1374*1.5)*Ue_fit_new**(-0.0741*np.exp(0.2140*1.5)), 'k-', label='Jiang et al. (2024)')
plt.text(0.03, 0.94, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.xlim(0,225)
plt.ylim(0, 1.4)
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$e$ [-]', fontsize=14) 
plt.legend(fontsize=12)
plt.subplot(1,2,2)
for i in range(5):
    plt.errorbar(Uimplot_Omega[i]/constant, Thetaremean_Omega[i], yerr=Thetarestderr_Omega[i], fmt='o', capsize=5, color=colors[i])
    plt.plot(Uthetare_fit[i], np.degrees(thetare_fit[i]), '--', color=colors[i])
plt.text(0.03, 0.94, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{re}$ [$^\circ$]', fontsize=14)
plt.tight_layout()
plt.show()

#fit Pr with gompertz
def gompertz(x, L, a, b):
    return L * np.exp(-a*np.exp(-b*x))

Pr_tbfit = list(chain.from_iterable(Pr_Omega.values()))
Upr_tbfit = list(chain.from_iterable(Uplot_Omega.values()))
valid_indices = ~np.isnan(np.array(Pr_tbfit))
Upr_tbfit_new = np.array(Upr_tbfit)[valid_indices]/constant
Pr_tbfit_new = np.array(Pr_tbfit)[valid_indices]
N_PrUre_tbfit = list(chain.from_iterable(N_PrUre.values()))
N_PrUre_tbfit_new = np.array(N_PrUre_tbfit)[valid_indices]
weights_pr = 1/np.sqrt(N_PrUre_tbfit_new)
params, _ = curve_fit(gompertz, Upr_tbfit_new, Pr_tbfit_new, p0=[0.9, 10, 0.1], sigma=weights_pr, absolute_sigma=True)
L_pr,a_pr, b_pr = params
print(f'L_pr={L_pr:.4f} a_pr={a_pr:.4f} b_pr={b_pr:.4f}')
# 3. Plot the fit 
Uimpr_fit = np.linspace(0, max(Upr_tbfit_new), 100) #for the fit
Pr_fit = gompertz(Uimpr_fit, *params)

#calculate R^2
# Create interpolator from 100-point fit
interpolator = interp1d(Uimpr_fit, Pr_fit, kind='linear', fill_value='extrapolate')
# Evaluate fit at the same x-values as Uthetaplot
pr_fit_resampled = interpolator(Upr_tbfit_new)
# Now compute R²
r2_pr = weighted_r2_N(Pr_tbfit_new, pr_fit_resampled, N_PrUre_tbfit_new)
print('r2_pr:',r2_pr)

#get the first few dots with non-equal bin size to prove the Gompertz
Pr_drybin, Uprdrybin = module.BinUimUd(Vim_all_Omega[4],VD_all_Omega[4],8)
plt.figure(figsize=(6,5))
for i in range(5):
    plt.scatter(Uplot_Omega[i]/constant, Pr_Omega[i], s=np.sqrt(N_PrUre[i])*5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.plot(Uimpr_fit, Pr_fit, 'k--', label='Gompertz fit')
# plt.plot(Uimpr_fit, 0.9945*1.5**(-0.0166)*(1-np.exp(-0.1992*1.5**(-0.8686)*Uimpr_fit)), 'k-', label='Jiang et al. (2024)')
plt.plot(Uprdrybin/constant, Pr_drybin, 'o', color='gray', ms=5, label='Equal-count binning')
plt.xlim(0,210)
plt.ylim(0, 1.1)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$Pr$ [-]', fontsize=14)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


# Ejection
def linear(U, a):
    return a*U

U_NEfit, NE_fit = defaultdict(list),defaultdict(list)
a_NE = np.zeros(5)
for i in range(5):
    # --- Remove NaN values before fitting ---
    valid_indices = ~np.isnan(NE_mean_Omega[i])  # Get boolean mask where theta_all is NOT NaN
    U_clean = Uplot_NE_Omega[i][valid_indices]/constant       # Keep only valid U values
    NE_clean = np.array(NE_mean_Omega[i])[valid_indices]# Keep only valid theta values
    N_Einbin_clean = np.array(N_Einbin[i])[valid_indices]
    weights_NE = 1/np.sqrt(N_Einbin_clean)
    popt, _ = curve_fit(linear, U_clean, NE_clean, sigma=weights_NE, absolute_sigma=True)
    a_NE[i]= popt 
    # Generate fitted curve
    U_NEfit[i] = np.linspace(min(U_clean), max(U_clean), 200)
    NE_fit[i] = linear(U_NEfit[i], *popt)
    ne_fit_dis = linear(U_clean, *popt)
    R2 = weighted_r2(NE_clean, ne_fit_dis, weights=N_Einbin_clean)
    print('R2_NE:', R2)

plt.figure(figsize=(6,5.5))
for i in range(5):
    plt.scatter(Uplot_NE_Omega[i]/constant, NE_mean_Omega[i], s=np.sqrt(N_Einbin[i])*5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
    plt.plot(U_NEfit[i], NE_fit[i], '--', color=colors[i])
plt.xlim(0,225)
plt.ylim(0, 5.75)
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
plt.legend(loc='upper left', fontsize=12)

paramsNEa, _ = curve_fit(power_law, Omega_tbfit, a_NE)
aNE_fit_smooth = power_law(Omega_smooth, *paramsNEa)
A_NE1, B_NE1, n_NE1 = paramsNEa

plt.figure()
plt.plot(Omega_tbfit, a_NE, 'o')
plt.plot(Omega_smooth, aNE_fit_smooth, '--', label='Power-law fit')
plt.xlabel('Omega')
plt.ylabel('a_NE')

aNE_fit = A_NE1 - B_NE1*Omega_tbfit**n_NE1
print(f"aNE_fit = {A_NE1:.4f} - {B_NE1:.4f}*Omega_tbfit**{n_NE1:.4f}")
UNE_fit_new = np.linspace(0, max(Uplot_NE_Omega[0]/constant), 100)
NE_fit_new = defaultdict()
# Loop over each element in alpha and multiply with U_fit_new
for i in range(5):
    NE_fit_new[i] = linear(UNE_fit_new, aNE_fit[i])
    
#calculate R^2
# Lists for global R²
all_NE_clean = []
all_NE_fit_resampled = []
all_weights_NE = []

for i in range(5):
    # Create interpolator from 100-point fit
    interpolator = interp1d(UNE_fit_new, NE_fit_new[i], kind='linear', fill_value='extrapolate')
    
    # Remove NaNs
    valid_indices = ~np.isnan(NE_mean_Omega[i])
    U_clean = Uplot_NE_Omega[0][valid_indices] / constant
    NE_clean = np.array(NE_mean_Omega[i])[valid_indices]
    NE_fit_resampled = interpolator(U_clean)
    
    # Weights
    N_Einbin_clean = np.array(N_Einbin[i])[valid_indices] #same as 1/stderr**2
    
    # Store for global R²
    all_NE_clean.append(NE_clean)
    all_NE_fit_resampled.append(NE_fit_resampled)
    all_weights_NE.append(N_Einbin_clean)
    
y_true_all = np.concatenate(all_NE_clean)
y_pred_all = np.concatenate(all_NE_fit_resampled)
weights_all = np.concatenate(all_weights_NE)
# Weighted mean of y_true
y_mean_weighted = np.average(y_true_all, weights=weights_all)
# Weighted TSS and RSS
wtss = np.sum(weights_all * (y_true_all - y_mean_weighted) ** 2)
wrss = np.sum(weights_all * (y_true_all - y_pred_all) ** 2)
r2_global = 1 - (wrss / wtss)
print(f'Global R² (all groups combined): {r2_global:.3f}')


#UE - Uinc
U_UEfit, UE_fit = defaultdict(list), defaultdict(list)
a_UE, b_UE = np.zeros(5), np.zeros(5)
for i in range(5):
    # --- Remove NaN values before fitting ---
    valid_indices = ~np.isnan(UE_mean_Omega[i])  # Get boolean mask where theta_all is NOT NaN
    U_clean = Uplot_NE_Omega[i][valid_indices]/constant       # Keep only valid U values
    UE_clean = np.array(UE_mean_Omega[i])[valid_indices]/constant     # Keep only valid UE values
    stderr = np.array(UE_stderr_Omega[i])[valid_indices]/constant  
    valid_mask = stderr != 0
    # 用该掩码过滤所有列表
    U_clean = U_clean[valid_mask]
    UE_clean = UE_clean[valid_mask]
    stderr = stderr[valid_mask]
    popt, _ = curve_fit(power_fit, U_clean, UE_clean, sigma=stderr, absolute_sigma=True, maxfev=10000)
    a_UE[i], b_UE[i] = popt 
    # Generate fitted curve
    U_UEfit[i] = np.linspace(min(U_clean), max(U_clean), 200)
    UE_fit[i] = power_fit(U_NEfit[i], *popt)
    UE_fit_dis = power_fit(U_clean, *popt)
    R2_UE = weighted_r2(UE_clean, UE_fit_dis, weights=1/(stderr**2))
    print('R2_UE:', R2_UE)
    
plt.figure(figsize=(6,5.5))  
for i in range(5):
    plt.errorbar(Uplot_NE_Omega[i]/constant, UE_mean_Omega[i]/constant, yerr=UE_stderr_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
    plt.plot(U_UEfit[i], UE_fit[i], '--', color=colors[i])
plt.xlim(0,225)
plt.ylim(0, 14.5)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)

def linearincrease(Omega, a, b):
    return a*Omega + b

paramsUEa, _ = curve_fit(linearincrease, Omega_tbfit, a_UE)
aUE_fit_smooth = linearincrease(Omega_smooth, *paramsUEa)
a_UE1, b_UE1 = paramsUEa
plt.figure()
plt.plot(Omega_tbfit, a_UE, 'o')
plt.plot(Omega_smooth, aUE_fit_smooth, '--', label='Linear fit')
plt.xlabel('Omega')
plt.ylabel('a_UE')
aUE_fit = linearincrease(Omega_tbfit, *paramsUEa)
print(f"aUE_fit = {a_UE1:.4f} * Omega_tbfit + {b_UE1:.4f}")

# def quad_fit(omega, a, b, c):
#     return a * omega**2 + b * omega + c

paramsUEb, _ = curve_fit(linearincrease, Omega_tbfit[1:], b_UE[1:])
bUE_fit_smooth = linearincrease(Omega_smooth, *paramsUEb)
a_UE2, b_UE2 = paramsUEb
plt.figure()
plt.plot(Omega_tbfit, b_UE, 'o')
plt.plot(Omega_smooth, bUE_fit_smooth, '--', label='quadratic fit')
plt.xlabel('Omega')
plt.ylabel('b_UE')
bUE_fit = np.append(0, linearincrease(Omega_tbfit[1:], *paramsUEb))
print(f"{a_UE2:.2f}*omega + {b_UE2:.2f}")

UUE_fit_new = np.linspace(min(Uplot_NE_Omega[0]/constant), max(Uplot_NE_Omega[0]/constant), 100)
UE_fit_new = defaultdict()
# Loop over each element in alpha and multiply with U_fit_new
for i in range(5):
    UE_fit_new[i] = power_fit(UUE_fit_new, aUE_fit[i], bUE_fit[i])
    
#calculate R^2
all_UE_clean, all_UE_fit_resampled, all_weights_UE=[],[],[]
for i in range(5):
    # Create interpolator from 100-point fit
    interpolator = interp1d(UUE_fit_new, UE_fit_new[i], kind='linear', fill_value='extrapolate')
    # Evaluate fit at the same x-values as Uthetaplot
    # --- Remove NaN values before fitting ---
    valid_indices = ~np.isnan(UE_mean_Omega[i])  # Get boolean mask where theta_all is NOT NaN
    U_clean = Uplot_NE_Omega[0][valid_indices]/constant       # Keep only valid U values
    UE_clean = np.array(UE_mean_Omega[i])[valid_indices]/constant  # Keep only valid theta values
    stderr = np.array(UE_stderr_Omega[i])[valid_indices]/constant  
    valid_mask = stderr != 0
    # 用该掩码过滤所有列表
    U_clean = U_clean[valid_mask]
    UE_clean = UE_clean[valid_mask]
    stderr = stderr[valid_mask]
    UE_fit_resampled = interpolator(U_clean)
    # Store for global R²
    all_UE_clean.append(UE_clean)
    all_UE_fit_resampled.append(UE_fit_resampled)
    all_weights_UE.append(1/(stderr**2))
    
y_ue_all = np.concatenate(all_UE_clean)
y_predue_all = np.concatenate(all_UE_fit_resampled)
weightsue_all = np.concatenate(all_weights_UE)
# Weighted TSS and RSS
r2_ue = weighted_r2(y_ue_all, y_predue_all, weightsue_all)
print(f'Global UE R² (all groups combined): {r2_ue:.3f}')


#thetaE
plt.figure(figsize=(18,5.5))
plt.subplot(1,3,1)
for i in range(5):
    plt.scatter(Uplot_NE_Omega[i]/constant, NE_mean_Omega[i], s=np.sqrt(N_Einbin[i])*5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
    plt.plot(UNE_fit_new, NE_fit_new[i], '--', color=colors[i])
plt.xlim(0,225)
plt.ylim(0, 5.75)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
plt.text(0.03, 0.93, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.legend(loc='upper left', bbox_to_anchor=(0.12, 0.99), fontsize=12)
plt.subplot(1,3,2)
for i in range(5):
    plt.errorbar(Uplot_NE_Omega[i]/constant, UE_mean_Omega[i]/constant, yerr=UE_stderr_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
    plt.plot(UUE_fit_new, UE_fit_new[i], '--', color=colors[i])
plt.xlim(0,225)
plt.ylim(0, 14.5)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
plt.text(0.03, 0.93, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(1,3,3)
for i in range(5):
    plt.errorbar(Uplot_NE_Omega[i]/constant, ThetaE_mean_Omega[i], yerr=ThetaE_stderr_Omega[i], fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
    # plt.plot(UNE_fit_new, NE_fit_new[i], '--', color=colors[i])
plt.xlim(0,225)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{\theta}_\mathrm{E}$ [$^\circ$]', fontsize=14)
plt.text(0.03, 0.93, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.tight_layout()

thetaEmean = np.zeros(5)
for i in range(5):
    mask = ~np.isnan(np.array(ThetaE_mean_Omega[i]))
    thetaEmean[i] = np.average(np.array(ThetaE_mean_Omega[i])[mask], weights=np.array(ThetaE_stderr_Omega[i])[mask])
print('thetaE:', thetaEmean)
print('global mean:', np.mean(thetaEmean))

# final plot with 6 terms
plt.figure(figsize=(12,13.5))
plt.subplot(3,2,1)
for i in range(5):
    plt.scatter(Uimplot_Omega[i]/constant,CORmean_Omega[i], s=np.sqrt(N_COR_Omega[i])*5, color=colors[i], label=f"$\\Omega$={Omega[i]}%")
    plt.plot(Ue_fit_new, e_fit_new[i], '--', color=colors[i])
# plt.plot(Ue_fit_new, 0.7469*np.exp(0.1374*1.5)*Ue_fit_new**(-0.0741*np.exp(0.2140*1.5)), 'k-', label='Jiang et al. (2024)')
plt.text(0.02, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.xlim(0,225)
plt.ylim(0, 1.4)
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{e}$ [-]', fontsize=14) 
plt.legend(fontsize=12)
plt.subplot(3,2,2)
for i in range(5):
    plt.errorbar(Uimplot_Omega[i]/constant, Thetaremean_Omega[i], yerr=Thetarestderr_Omega[i], fmt='o', capsize=5, color=colors[i])
    plt.plot(Uthetare_fit[i], np.degrees(thetare_fit[i]), '--', color=colors[i])
plt.text(0.02, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{re}$ [$^\circ$]', fontsize=14)
plt.subplot(3,2,3)
for i in range(5):
    plt.scatter(Uplot_Omega[i]/constant, Pr_Omega[i], s=np.sqrt(N_PrUre[i])*5, label='_nolegend_', color=colors[i])
plt.plot(Uimpr_fit, Pr_fit, 'k--', label='_nolegend_')
# plt.plot(Uimpr_fit, 0.9945*1.5**(-0.0166)*(1-np.exp(-0.1992*1.5**(-0.8686)*Uimpr_fit)), 'k-', label='Jiang et al. (2024)')
plt.plot(Uprdrybin/constant, Pr_drybin, 'o', color='gray', ms=5, label='Equal-count binning')
plt.xlim(0,210)
plt.ylim(0, 1.1)
plt.text(0.02, 0.92, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$Pr$ [-]', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.subplot(3,2,4)
for i in range(5):
    plt.scatter(Uplot_NE_Omega[i]/constant, NE_mean_Omega[i], s=np.sqrt(N_Einbin[i])*5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
    plt.plot(UNE_fit_new, NE_fit_new[i], '--', color=colors[i])
plt.xlim(0,225)
plt.ylim(0, 5.75)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(3,2,5)
for i in range(5):
    plt.errorbar(Uplot_NE_Omega[i]/constant, UE_mean_Omega[i]/constant, yerr=UE_stderr_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
    plt.plot(UUE_fit_new, UE_fit_new[i], '--', color=colors[i])
plt.xlim(0,225)
plt.ylim(0, 14.5)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(e)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(3,2,6)
for i in range(5):
    plt.errorbar(Uplot_NE_Omega[i]/constant, ThetaE_mean_Omega[i], yerr=ThetaE_stderr_Omega[i], fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
    # plt.plot(UNE_fit_new, NE_fit_new[i], '--', color=colors[i])
plt.xlim(0,225)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{E}$ [$^\circ$]', fontsize=14)
plt.text(0.03, 0.93, '(f)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.tight_layout()


# plt.figure()
# for i in range(5):
#     plt.plot(Uplot_NE_Omega[i]/constant, EE_mean_Omega[i],'o', color=colors[i])
# plt.xlabel(r'$U_{inc}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$E_\mathrm{E}$ []', fontsize=14)

# plt.subplot(1,3,3)
# plt.errorbar(Omega, UE_mean_glo/constant, yerr=UE_std_glo/constant, fmt='o', capsize=5, color='#3776ab', label='Simulated data')
# plt.plot(Omega_fit*100, UE_fit, 'k--', label='Exponential fit')
# plt.ylim(0, 20)
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
# plt.text(0.03, 0.93, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.legend(fontsize=12)

# plt.subplot(2,2,3)
# plt.plot(Omega, NE_mean_glo, 'o', color='#3776ab', label='Simulated data')
# plt.plot(Omega_fit*100, NE_fit, 'k--', label='Power-law fit')
# plt.ylim(0, 1)
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
# plt.text(0.03, 0.93, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.legend(fontsize=12)
# plt.subplot(2,2,4)
# handles,labels =[],[]
# h1 = plt.errorbar(Omega, UE_mean_glo/constant, yerr=UE_std_glo/constant, fmt='o', capsize=5, color='#3776ab', label='Simulated data')
# handles.append(h1)
# labels.append('Simulated data')
# h2, = plt.plot(Omega_fit*100, UE_fit, 'k--', label='Exponential fit')
# handles.append(h2)
# labels.append('Exponential fit')
# plt.ylim(0, 20)
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
# plt.text(0.03, 0.93, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.legend(handles=handles, labels=labels, fontsize=12)
# plt.tight_layout()
# plt.show()


plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
for i in range(5):
    plt.errorbar(Thetaimplot_Omega[i], COR_theta_mean_Omega[i], yerr=COR_theta_std_Omega[i], fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
plt.ylabel(r'$e$ [-]', fontsize=14)
plt.legend(fontsize=12)
plt.subplot(2,2,2)
for i in range(5):
    plt.plot(Theta_pr_Omega[i], Pr_theta_Omega[i], 'o', label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
plt.ylabel(r'$P_\mathrm{r}$ [-]', fontsize=14)
plt.subplot(2,2,3)
for i in range(5):
    plt.plot(Thetaplot_NE_Omega[i], NE_theta_mean_Omega[i], 'o', label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
plt.subplot(2,2,4)
for i in range(5):
    plt.errorbar(Thetaplot_NE_Omega[i], UE_theta_mean_Omega[i]/constant, yerr=UE_theta_std_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
plt.tight_layout()
plt.show()

# # plt.figure(figsize=(12,5))
# # plt.subplot(1,2,1)
# # plt.errorbar(Omega, list(NEmeanUIM.values()), yerr=list(NEstdUIM.values()), fmt='o', capsize=5, color='#3776ab')
# # plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# # plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
# # plt.subplot(1,2,2)
# # plt.errorbar(Omega, list(UEmeanUIM.values())/constant, yerr=list(UEstdUIM.values())/constant, fmt='o', capsize=5, color='#3776ab')
# # plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# # plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
# # plt.tight_layout()
