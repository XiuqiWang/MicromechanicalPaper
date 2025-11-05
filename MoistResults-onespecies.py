# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:57:30 2025

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
coe_sal_h = 13.5 # treat all hopping rains by one species
N_inter = 100 #number of output timesteps for erosion and deposition properties
t_inter = np.linspace(0.01+(5-0.01)/N_inter/2,5,N_inter)
Omega = [0, 1, 5, 10, 20]
Omega_tbfit = np.array(Omega, dtype=float)*0.01
Omega_smooth = np.linspace(min(Omega_tbfit), max(Omega_tbfit), 100)
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
IM_vector_t = defaultdict(list)
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
    ParticleID=np.linspace(0,num_p-1,num_p)
    ParticleID_int = ParticleID.astype(int)
    #cal erosion and deposition properties for each Omega
    #EDindices, E, VX, VExVector, VEzVector, VEx, VEz, ME, MD
    EDindices[i], ME[i], MD[i], VExz_mean_t[i], VDxz_mean_t[i], D_vector_t[i], E_vector_t[i], _=module.store_particle_id_data(data,ParticleID_int,coe_h,dt,N_inter,D)
    #cal rebound properties for each Omega
    ParticleID_sal=np.linspace(0,num_p-1,num_p)
    ParticleID_salint = ParticleID_sal.astype(int)
    X[i] = np.array([[time_step['Position'][i][0] for i in ParticleID_salint] for time_step in data])
    Z[i] = np.array([[time_step['Position'][i][2] for i in ParticleID_salint] for time_step in data])
    Par[i], VZ[i], exz_mean_t[i], ez_mean_t[i], VIM_mean_t[i], ThetaIM_mean_t[i], RIM[i], exz_vector_t[i], IM_vector_t[i]=module.store_sal_id_data(data,ParticleID_salint, coe_sal_h, dt, N_inter, D)
 
# U_list = []
# for i in range(5):
#     file_c = f'../ContinuumModel/dcdt/Discrete_CU/S00{i+2}M20discrete.txt'
#     data_dpm = np.loadtxt(file_c)
#     U_dpm = data_dpm[:, 1]
#     U_list.append(U_dpm)
t_dpm = np.linspace(0.01, 5, 501)

# plt.close('all')
# for i in range(5):  # 5 groups
#     plt.figure(figsize=(12, 8))
#     for j in range(5):
#         plt.subplot(2, 3, j+1)
#         index = j*5+i # BY Omega
#         plt.plot(t_dpm, U_list[index], label=r'$U$')
#         index_byS = i*5+j # by Shields
#         UD = VDxz_mean_t[index_byS] 
#         plt.plot(t_inter, UD, label=r'$U_{D}$')
#         plt.title(fr'$\Omega$ = {Omega[j]} $\%$')
#         plt.ylabel(r'$U$ and $U_{D}$ [m/s]')
#         plt.xlabel(r't [s]')
#         plt.xlim(left=0)
#         plt.ylim(0,10)
#         plt.grid(True)
#     plt.legend()
#     plt.suptitle(f'Shields=0.0{i+2}')
#     plt.tight_layout()
#     plt.show()    
    
# plt.figure(figsize=(12, 8))
# for j in range(5):
#     plt.subplot(2, 3, j+1)
#     index_byS = 4*5+j # by Shields
#     Uim = VIM_mean_t[index_byS] 
#     plt.plot(t_inter, Uim, label=r'$U_{im}$')
#     index = j*5+4 # BY Omega
#     plt.plot(t_dpm, U_list[index]/np.cos(12/180*np.pi), label=r'$U/\cos(\theta_{im})$')
#     plt.title(fr'$\Omega$ = {Omega[j]} $\%$')
#     plt.ylabel(r'$U_{im}$ [m/s]')
#     plt.xlabel(r't [s]')
#     plt.xlim(left=0)
#     plt.ylim(0,10)
#     plt.grid(True)
# plt.legend()
# plt.suptitle(f'Shields=0.0{i+2}')
# plt.tight_layout()
# plt.show()    
    
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
    # Vrep_all[i] = [value[-1] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
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
Vim_bin = np.linspace(0, max(Vim_all_values)+1, 10)
VD_all_values = [value for sublist in VD_all.values() for value in sublist]
Vde_bin = np.linspace(0, max(VD_all_values)+1, 15)
Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
Vimde_bin = np.linspace(0, max(Vimde_all_values)+1, 10)
Vre_all_values = [value for sublist in Vre_all.values() for value in sublist]
Vre_bin = np.linspace(0, max(Vre_all_values)+1, 10)
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

CORmean_Omega,CORstderr_Omega,N_COR_Omega,Uimplot_Omega, Thetaremean_Omega, Thetarestderr_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
Pr_Omega,Uplot_Omega, N_PrUre, Uim_meaninbin, UD_meaninbin = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
NE_mean_Omega, UE_mean_Omega, UE_stderr_Omega, Uplot_NE_Omega, N_Einbin = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
ThetaE_mean_Omega, ThetaE_stderr_Omega = defaultdict(list), defaultdict(list)
for i in range (5): #loop over Omega 0-20%
    CORmean_Omega[i], CORstd_Omega, CORstderr_Omega[i], N_COR_Omega[i], Thetaremean_Omega[i], Thetastd_Omega, Thetarestderr_Omega[i], Uimplot_Omega[i] = module.BinUimCOR_equalbinsize(Vim_all_Omega[i],Vre_all_Omega[i],Thetare_all_Omega[i],Vim_bin)
    Pr_Omega[i],Uplot_Omega[i],N_PrUre[i], Uim_meaninbin[i], UD_meaninbin[i] = module.BinUimUd_equalbinsize(Vim_all_Omega[i],VD_all_Omega[i],Vimde_bin)   
    # NE_mean_Omega[i], UE_mean_Omega[i], UE_std_Omega, UE_stderr_Omega[i], ThetaE_mean_Omega[i], ThetaE_std_Omega, ThetaE_stderr_Omega[i], N_Einbin[i], Uplot_NE_Omega[i] = module.BinEjectionData(matched_Vim_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], matched_thetaE_Omega[i], Vim_bin)
    NE_mean_Omega[i], UE_mean_Omega[i], UE_std_Omega, UE_stderr_Omega[i], ThetaE_mean_Omega[i], ThetaE_std_Omega, ThetaE_stderr_Omega[i], N_Einbin[i], Uplot_NE_Omega[i]=module.get_ejection_ratios_equalbinsize(matched_Vim_Omega[i], VD_all_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], matched_thetaE_Omega[i], Vimde_bin)#matched_EE_Omega[i]
   
#fitting functions
def power_fit(U, a, b):
    return a*U**b

def power_law(Omega, A, B, n):
    return A - B * Omega **n

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

# #4 terms with Uim
# COR - Uim
plt.figure(figsize=(6,5.5))
for i in range(5):
    plt.errorbar(Uimplot_Omega[i]/constant,CORmean_Omega[i], yerr=CORstderr_Omega[i], fmt='o', capsize=5, color=colors[i], label=f"$\\Omega$={Omega[i]}%")
plt.xlim(left=0)
plt.ylim(0,1)
plt.legend(fontsize=12)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$e$ [-]', fontsize=14) 
plt.tight_layout()

def GaussianBell(U, a, b, A0, mu, sigma):
    e = a*U**(-b) + A0 * np.exp(-(U - mu)**2 / (2 * sigma**2)) #
    return e

UCOR_fit, COR_fit = defaultdict(list),defaultdict(list)
a_COR, b_COR, A0_COR, mu_COR, sigma_COR = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5)
for i in range(5):
    # --- Remove NaN values before fitting ---
    valid_indices = ~np.isnan(CORmean_Omega[i])  # Get boolean mask where theta_all is NOT NaN
    U_clean = np.array(Uimplot_Omega[i])[valid_indices]/constant       # Keep only valid U values
    # print('i',i,'U_clean', U_clean)
    COR_clean = np.array(CORmean_Omega[i])[valid_indices] # Keep only valid COR values
    # weights = 1/np.sqrt(np.array(N_COR_Omega[i])[valid_indices])
    stderr = np.array(CORstderr_Omega[i])[valid_indices]/constant
    valid_mask = np.where(~np.isnan(stderr))[0]
    # 用该掩码过滤所有列表
    U_clean = U_clean[valid_mask]
    COR_clean = COR_clean[valid_mask]
    stderr = stderr[valid_mask]
    # print('COR_clean',COR_clean)
    # print('weights', weights)
    bounds = ([0, 0, 0, 0, 1], [10, 2, 1, 500, 300])  # lower and upper bounds
    popt, _ = curve_fit(GaussianBell, U_clean, COR_clean, p0 = [3, 0.6, 0.2, 300, 150], bounds=bounds, sigma=stderr, absolute_sigma=True, maxfev=10000)
    a_COR[i], b_COR[i], A0_COR[i], mu_COR[i], sigma_COR[i] = popt 
    # Generate fitted curve
    UCOR_fit[i] = np.linspace(min(U_clean), max(U_clean), 200)
    COR_fit[i] = GaussianBell(UCOR_fit[i], *popt)
    COR_fit_dis = GaussianBell(U_clean, *popt)
    R2 = weighted_r2(COR_clean, COR_fit_dis, 1/stderr**2)
    print('R2_COR1:', R2)
    

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

A0 = 0.12 * np.log(1 + 1061.81*Omega_tbfit)
plt.figure()
plt.plot(Omega_tbfit, mu_COR, 'o')
mu = 312.48 #np.mean(np.delete(mu_COR,1)) # 
print('mu=',mu)

plt.figure()
plt.plot(Omega, sigma_COR, 'o')
sigma = 156.27 #np.mean(np.delete(sigma_COR,1)) #
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
    stderr = np.array(CORstderr_Omega[i])[valid_indices]
    valid_mask = np.where(~np.isnan(stderr))[0]
    # 用该掩码过滤所有列表
    U_clean = U_clean[valid_mask]
    e_clean = e_clean[valid_mask]
    stderr = stderr[valid_mask]
    e_fit_resampled = interpolator(U_clean)
    e_all.append(e_clean)
    e_fit_resampled_all.append(e_fit_resampled)
    weight_e_all.append(1/stderr**2)
    
y_e_all = np.concatenate(e_all)
y_prede_all = np.concatenate(e_fit_resampled_all)
weight_e_glo = np.concatenate(weight_e_all)
# Now compute R²
R2_e = weighted_r2(y_e_all, y_prede_all, weights=weight_e_glo)
print('R2_e:',R2_e)

plt.figure(figsize=(6,5.5))
for i in range(5):
    plt.errorbar(Uimplot_Omega[i]/constant,CORmean_Omega[i], yerr=CORstderr_Omega[i], fmt='o', capsize=5, color=colors[i], label=f"$\\Omega$={Omega[i]}%")
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
    # weights = 1/np.sqrt(np.array(N_COR_Omega[i])[valid_indices])
    stderr = np.radians(Thetarestderr_Omega[i])[valid_indices]
    valid_mask = np.where(~np.isnan(stderr))[0]
    # 用该掩码过滤所有列表
    U_clean = U_clean[valid_mask]
    thetare_clean = thetare_clean[valid_mask]
    stderr = stderr[valid_mask]
    popt, _ = curve_fit(arcsinlinear, U_clean, thetare_clean, p0=[-0.001,1], sigma=stderr, absolute_sigma=True)
    a_thetare[i], b_thetare[i] = popt 
    # Generate fitted curve
    Uthetare_fit[i] = np.linspace(min(U_clean), max(U_clean), 200)
    thetare_fit[i] = arcsinlinear(Uthetare_fit[i], *popt)
    thetare_fit_dis = arcsinlinear(U_clean, *popt)
    R2 = weighted_r2(thetare_clean, thetare_fit_dis, 1/stderr**2)
    print('R2_thetare:', R2)

plt.figure()
for i in range(5):
    plt.scatter(Uimplot_Omega[i]/constant, Thetaremean_Omega[i], s=np.sqrt(N_COR_Omega[i])*5, color=colors[i])
    plt.plot(Uthetare_fit[i], np.degrees(thetare_fit[i]), '--', color=colors[i])
plt.xlabel(r'$U_{im}/\sqrt{gd}$ []')
plt.ylabel(r'$\bar{\theta}_{re}$ [$^\circ$]')

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
    valid_mask = np.where(~np.isnan(stderr))[0]
    # 用该掩码过滤所有列表
    U_clean = U_clean[valid_mask]
    thetare_clean = thetare_clean[valid_mask]
    stderr = stderr[valid_mask]
    thetare_fit_resampled = interpolator(U_clean)
    # Store for global R²
    all_thetare_clean.append(thetare_clean)
    all_thetare_fit_resampled.append(thetare_fit_resampled)
    all_weights_thetare.append(1/stderr**2)
    
y_thetare_all = np.concatenate(all_thetare_clean)
y_predthetare_all = np.concatenate(all_thetare_fit_resampled)
weight_thetare_glo = np.concatenate(all_weights_thetare)
# Now compute R²
R2_thetare = weighted_r2(y_thetare_all, y_predthetare_all, weights=weight_thetare_glo)
print('R2_thetare:',R2_thetare)

plt.figure(figsize=(6,5.5))
for i in range(5):
    plt.errorbar(Uimplot_Omega[i]/constant, Thetaremean_Omega[i], yerr=Thetarestderr_Omega[i], fmt='o', capsize=5, color=colors[i])
    plt.plot(Uthetare_fit_new, np.degrees(thetare_fit_new[i]), '--', color=colors[i])
plt.text(0.02, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.xlim(0,225)
plt.ylim(bottom=0)
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{re}$ [$^\circ$]', fontsize=14)

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

plt.figure(figsize=(6,5))
for i in range(5):
    plt.errorbar(Uimplot_Omega[i]/constant, Thetaremean_Omega[i], yerr=Thetarestderr_Omega[i], fmt='o', capsize=5, color=colors[i])
    plt.plot(Uthetare_fit[i], np.degrees(thetare_fit[i]), '--', color=colors[i])
plt.text(0.03, 0.94, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{re}$ [$^\circ$]', fontsize=14)
plt.tight_layout()
plt.show()

#calculate R^2
# Create interpolator from 100-point fit
interpolator = interp1d(Uimpr_fit, Pr_fit, kind='linear', fill_value='extrapolate')
# Evaluate fit at the same x-values as Uthetaplot
pr_fit_resampled = interpolator(Upr_tbfit_new)
# Now compute R²
mask = N_PrUre_tbfit_new != 0
N_PrUre_tbfit_new, Pr_tbfit_new, pr_fit_resampled = N_PrUre_tbfit_new[mask], Pr_tbfit_new[mask], pr_fit_resampled[mask]
r2_pr = weighted_r2(Pr_tbfit_new, pr_fit_resampled, N_PrUre_tbfit_new)
print('r2_pr:',r2_pr)

#get the first few dots with non-equal bin size to prove the Gompertz
Pr_drybin, Uprdrybin = module.BinUimUd(Vim_all_Omega[4],VD_all_Omega[4],8)
plt.figure(figsize=(6,5))
for i in range(5):
    plt.scatter(Uplot_Omega[i]/constant, Pr_Omega[i], s=np.sqrt(N_PrUre[i])*5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.plot(Uimpr_fit, Pr_fit, 'k--', label='Gompertz fit')
# plt.plot(Uimpr_fit, 0.9945*1.5**(-0.0166)*(1-np.exp(-0.1992*1.5**(-0.8686)*Uimpr_fit)), 'k-', label='Jiang et al. (2024)')
plt.plot(Uprdrybin/constant, Pr_drybin, 'o', color='gray', ms=5, label='Equal-count binning')
plt.xlim(0,250)
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

plt.figure(figsize=(6,5.5))  
for i in range(5):
    plt.scatter(Uplot_NE_Omega[i]/constant, NE_mean_Omega[i], s=np.sqrt(N_Einbin[i])*5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
    plt.plot(UNE_fit_new, NE_fit_new[i], '--', color=colors[i])
plt.xlim(0,225)
plt.ylim(0, 5.75)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$N_\mathrm{E}$ [-]', fontsize=14)

# UE
U_UEfit, UE_fit = defaultdict(list), defaultdict(list)
a_UE, b_UE = np.zeros(5), np.zeros(5)
for i in range(5):
    # --- Remove NaN values before fitting ---
    valid_indices = ~np.isnan(UE_mean_Omega[i])  # Get boolean mask where theta_all is NOT NaN
    U_clean = Uplot_NE_Omega[i][valid_indices]/constant       # Keep only valid U values
    UE_clean = np.array(UE_mean_Omega[i])[valid_indices]/constant     # Keep only valid UE values
    stderr = np.array(UE_stderr_Omega[i])[valid_indices]/constant
    valid_mask = np.where(~np.isnan(stderr))[0]
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
    R2_UE = weighted_r2(UE_clean, UE_fit_dis, weights=1/stderr**2)
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

UUE_fit_new = np.linspace(0, max(Uplot_NE_Omega[0]/constant), 100)
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
    valid_mask = np.where(~np.isnan(stderr))[0]
    # 用该掩码过滤所有列表
    U_clean = U_clean[valid_mask]
    UE_clean = UE_clean[valid_mask]
    stderr = stderr[valid_mask]
    UE_fit_resampled = interpolator(U_clean)
    # Store for global R²
    all_UE_clean.append(UE_clean)
    all_UE_fit_resampled.append(UE_fit_resampled)
    all_weights_UE.append(1/stderr**2)
    
y_ue_all = np.concatenate(all_UE_clean)
y_predue_all = np.concatenate(all_UE_fit_resampled)
weightsue_all = np.concatenate(all_weights_UE)
# Weighted TSS and RSS
r2_ue = weighted_r2(y_ue_all, y_predue_all, weightsue_all)
print(f'Global UE R² (all groups combined): {r2_ue:.3f}')

plt.figure(figsize=(6,5.5))  
for i in range(5):
    plt.errorbar(Uplot_NE_Omega[i]/constant, UE_mean_Omega[i]/constant, yerr=UE_stderr_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
    plt.plot(UUE_fit_new, UE_fit_new[i], '--', color=colors[i])
plt.xlim(0,225)
plt.ylim(bottom=0)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)

# thetaE
plt.figure(figsize=(6,5.5))  
for i in range(5):
    plt.errorbar(Uplot_NE_Omega[i]/constant, ThetaE_mean_Omega[i], yerr=ThetaE_stderr_Omega[i], fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlim(0,225)
plt.ylim(0,50)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{E}$ [$^\circ$]', fontsize=14)

# distribution of theta and Uim
def compute_percentiles_by_omega(data_dict, percentiles=[10, 25, 50, 75, 90]):
    omega_values = sorted(data_dict.keys())
    percentile_data = []
    for omega in omega_values:
        data = np.array(data_dict[omega])
        percentiles_for_omega = [np.percentile(data, p) for p in percentiles]
        percentile_data.append(percentiles_for_omega)
    return percentile_data

percentiles=[10, 25, 50, 75, 90]
VD_per = compute_percentiles_by_omega(VD_all_Omega, percentiles)
percentile_VD = np.array(VD_per)
p10_VD, p25_VD, p50_VD, p75_VD, p90_VD = percentile_VD.T
ThetaD_per = compute_percentiles_by_omega(ThetaD_all_Omega, percentiles)
percentile_ThetaD = np.array(ThetaD_per)
p10_ThetaD, p25_ThetaD, p50_ThetaD, p75_ThetaD, p90_ThetaD = percentile_ThetaD.T

percentiles=[10, 25, 50, 75, 90]
Vim_per = compute_percentiles_by_omega(Vim_all_Omega, percentiles)
percentile_Vim = np.array(Vim_per)
p10_Vim, p25_Vim, p50_Vim, p75_Vim, p90_Vim = percentile_Vim.T
Thetaim_per = compute_percentiles_by_omega(Thetaim_all_Omega, percentiles)
percentile_thetaim = np.array(Thetaim_per)
p10_thetaim, p25_thetaim, p50_thetaim, p75_thetaim, p90_thetaim = percentile_thetaim.T

plt.figure(figsize=(12, 9))
# Plot PDF of U_im/sqrt(gd)
plt.subplot(2, 2, 1)
for i in range(5):
    # Calculate histogram (density=True for probability density)
    counts, bin_edges = np.histogram(Vim_all_Omega[i], bins=50, density=True)
    plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# UD
for i in range(5):
    counts, bin_edges = np.histogram(VD_all_Omega[i], bins=50, density=True)
    plt.step(bin_edges[:-1], counts, where='mid', linestyle='--', color=colors[i])
plt.plot([], [], color='black', label=r"$U_\mathrm{im}$")
plt.plot([], [], '--', color='black', label=r"$U_\mathrm{D}$")
plt.xlabel(r'$U_\mathrm{inc}$ [m/s]', fontsize=14)
plt.ylabel('Probability Density [-]', fontsize=14)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend(fontsize=12)
plt.text(0.03, 0.94, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# Plot PDF of theta_im
plt.subplot(2, 2, 2)
for i in range(5):
    # Calculate histogram (density=True for probability density)
    counts, bin_edges = np.histogram(Thetaim_all_Omega[i], bins=50, density=True)
    # Create the step plot
    plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
for i in range(5):    
    counts, bin_edges = np.histogram(ThetaD_all_Omega[i], bins=50, density=True)
    plt.step(bin_edges[:-1], counts, where='mid', linestyle='--', color=colors[i])
plt.plot([], [], color='black', label=r"$\theta_\mathrm{im}$")
plt.plot([], [], '--', color='black', label=r"$\theta_\mathrm{D}$")
plt.legend(fontsize=12)
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xlabel(r'$\theta_\mathrm{inc}$ [$^\circ$]', fontsize=14)
plt.ylabel('Probability Density [-]', fontsize=14)
plt.text(0.03, 0.94, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(2, 2, 3)
# plt.errorbar(Omega, Vim_mean_glo, yerr=Vim_std_glo, fmt='o', capsize=5, color='#3776ab')
plt.errorbar(Omega,p50_Vim,yerr=[p50_Vim - p25_Vim, p75_Vim - p50_Vim],fmt='ko',capsize=5, label=r'$U_\mathrm{im}$')
plt.errorbar(Omega,p50_VD,yerr=[p50_VD - p25_VD, p75_VD - p50_VD],fmt='o',capsize=5, label=r'$U_\mathrm{D}$')
plt.legend(loc='upper right', fontsize=12)
plt.ylim(0,5.5)
plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
plt.ylabel(r'$U_\mathrm{inc}$ [m/s]', fontsize=14)
plt.text(0.03, 0.94, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(2, 2, 4)
# plt.errorbar(Omega, Thetaim_mean_glo, yerr=Thetaim_std_glo, fmt='o', capsize=5, color='#3776ab')
plt.errorbar(Omega,p50_thetaim,yerr=[p50_thetaim - p25_thetaim, p75_thetaim - p50_thetaim],fmt='ko',capsize=5, label=r'$\theta_\mathrm{im}$')
plt.errorbar(Omega,p50_ThetaD,yerr=[p50_ThetaD - p25_ThetaD, p75_ThetaD - p50_ThetaD],fmt='o',capsize=5, label=r'$\theta_\mathrm{D}$')
plt.ylim(8,48)
plt.legend(loc='upper right', fontsize=12)
plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{inc}$ [$\circ$]', fontsize=14)
plt.text(0.03, 0.94, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()