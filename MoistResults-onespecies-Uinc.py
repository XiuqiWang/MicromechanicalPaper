# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:59:11 2025

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
from scipy.optimize import least_squares

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
dt_output = (5 - 0.01) / N_inter
t_mid = 0.01 + (np.arange(N_inter) + 0.5) * dt_output
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
    EDindices[i], ME[i], MD[i], VExz_mean_t[i], VDxz_mean_t[i], D_vector_t[i], E_vector_t[i], _=module.store_particle_id_data(data,ParticleID_int,coe_h,dt,N_inter,D)
    #cal rebound properties for each Omega
    ParticleID_sal=np.linspace(0,num_p-1,num_p)
    ParticleID_salint = ParticleID_sal.astype(int)
    X[i] = np.array([[time_step['Position'][i][0] for i in ParticleID_salint] for time_step in data])
    Z[i] = np.array([[time_step['Position'][i][2] for i in ParticleID_salint] for time_step in data])
    Par[i], VZ[i], exz_mean_t[i], ez_mean_t[i], VIM_mean_t[i], ThetaIM_mean_t[i], RIM[i], exz_vector_t[i], IM_vector_t[i]=module.store_sal_id_data(data,ParticleID_salint, coe_sal_h, dt, N_inter, D)
    
    
# load CG U-t data
omega_labels = ['Dry', 'M1', 'M5', 'M10', 'M20']
# Initialize lists to hold results for each Omega
U_dpm, C_dpm = [], []
# Loop over all moisture levels
for i in range(2,7):
    for label in omega_labels:
        # --- Load sediment transport data ---
        file_path = f'../ContinuumModel/CGdata/hb=13.5d/Shields00{i}{label}-135d.txt'
        data = np.loadtxt(file_path)
        C = data[:, 1]
        U = data[:, 2]
        # Append to lists
        U_dpm.append(U)
        C_dpm.append(C)
    
t_dpm = np.linspace(0.01, 5, 501)

# calculate mass-weighted mean Uinc = (mim * Uim + mD* UD)/(mim + mD)
def _uinc_one(vim, vd):
    has_vim = isinstance(vim, (list, tuple)) and len(vim) == 2
    has_vd  = isinstance(vd,  (list, tuple)) and len(vd)  == 2

    if has_vim and has_vd:
        num = vim[0] + vd[0]
        den = vim[1] + vd[1]
    elif has_vim:
        num, den = vim
    elif has_vd:
        num, den = vd
    else:
        return np.nan  # 或者返回 0.0

    return num/den if den != 0 else np.nan

Uinc_t = {
    k: [_uinc_one(VIM_mean_t[k][i], VDxz_mean_t[k][i]) for i in range(100)]
    for k in VIM_mean_t
}


plt.close('all')
for i in range(5):  # 5 groups
    plt.figure(figsize=(10, 6))
    for j in range(5):
        plt.subplot(2, 3, j+1)
        index_byS = i*5+j # by Shields
        plt.plot(t_dpm, U_dpm[index_byS], label=r'$U$')
        Uinc = Uinc_t[index_byS] 
        plt.plot(t_mid, Uinc, label=r'$U_{inc}$')
        plt.title(fr'$\Omega$ = {Omega[j]} $\%$')
        plt.ylabel(r'$U$ and $U_{inc}$ [m/s]')
        plt.xlabel(r't [s]')
        plt.xlim(left=0)
        plt.ylim(0,10)
        plt.grid(True)
    plt.legend()
    plt.suptitle(f'Shields=0.0{i+2}')
    plt.tight_layout()
    plt.show()    

def resample_to_measured_times(U_dpm_onecase, t_dpm, t_mid):
    """interpolate U_dpm to match t_mid resolution"""
    f_interp = interp1d(t_dpm, U_dpm_onecase, kind='linear', fill_value='extrapolate')
    return f_interp(t_mid)    
    
def Cal_Uinc_test(U, C, Cref_Uinc):
    eps = 1e-8
    A = 1/(1 + C/(Cref_Uinc + eps))
    # n = 1/np.sqrt(1 + C/(Cref_Uinc_n+ eps))
    Uinc_test = A * U
    return Uinc_test

def residuals(params, U, C, Uinc_meas):
    Cref_Uinc = params
    Uinc_pred = Cal_Uinc_test(U, C, Cref_Uinc)
    res = Uinc_pred - Uinc_meas
    res = np.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
    return res

def fit_Cref_parameters():
    res_list = []

    for idx in range(10, 25):  
        # 1) model U to match measured Uinc times
        U_aligned = resample_to_measured_times(U_dpm[idx], t_dpm, t_mid)
        C_aligned = resample_to_measured_times(C_dpm[idx], t_dpm, t_mid)
        Uinc_meas = Uinc_t[idx]

        # store tuples for global fit
        res_list.append((U_aligned, C_aligned, Uinc_meas))

    # flatten
    def stacked_residuals(params):
        all_res = []
        for U,C,Uinc_meas in res_list:
            all_res.append(residuals(params, U, C, Uinc_meas))
        return np.concatenate(all_res)

    # initial guess + bounds
    x0 = [0.05]
    bounds = ([1e-4],[1.0])

    result = least_squares(stacked_residuals, x0, bounds=bounds, ftol=1e-12, xtol=1e-12)
    return result

result = fit_Cref_parameters()
Cref_opt = result.x
print("Optimal Cref_Uinc          =", Cref_opt)

plt.close('all')
for i in range(5):  # 5 omega
    plt.figure(figsize=(10, 8))
    for j in range(5): # 5 shields
        plt.subplot(3, 2, j+1)
        index_byS = i+j*5 
        Uinc_test = Cal_Uinc_test(U_dpm[index_byS], C_dpm[index_byS], Cref_opt)
        plt.plot(t_dpm, Uinc_test, label=r'$\breve{U}_\mathrm{inc}=\frac{{\hat{U}}}{{\sqrt{{1 + \hat{c}/c_\mathrm{{ref, Uinc}}}}}}$')
        Uinc = Uinc_t[index_byS] 
        plt.plot(t_mid, Uinc, label=r'$\hat{U}_\mathrm{inc}$')
        plt.title(fr'$\tilde{{\Theta}}$=0.0{j+2}')
        plt.ylabel(r'$U_\mathrm{inc}$ [m/s]')
        plt.xlabel(r'$t$ [s]')
        plt.xlim(left=0)
        plt.ylim(0,10)
        plt.grid(True)
        if j == 0:
            plt.legend(fontsize=9, loc='upper right')
    plt.suptitle(fr'$\Omega$={Omega[i]}%')
    plt.tight_layout()
    plt.show()  
    
    
# # Uinc = f(U) by combining all Uinc and U
# U_dpm_mid = [np.interp(t_mid, t_dpm, U) for U in U_dpm]

# Uinc_all_Omega, Udpm_all_Omega = defaultdict(list), defaultdict(list)
# for i in range (5): #loop over Omega 0-20 %
#     selected_indices = list(range(i, 25, 5))  # Get indices like [0,5,10,15,20], [1,6,11,16,21], etc.
#     Uinc_all_Omega[i] = np.concatenate([Uinc_t[j] for j in selected_indices]).tolist()
#     Udpm_all_Omega[i] = np.concatenate([U_dpm_mid[j] for j in selected_indices]).tolist()
    
# def Bin_Uinc_U(Uinc, U, Ubin):
#     Uinc = np.array(Uinc)
#     U = np.array(U)
#     Uinc_mean, Uinc_stderr = [], []
#     for i in range(len(Ubin)-1):
#         # Find indices of velocities within the current range
#         indices = (U >= Ubin[i]) & (U < Ubin[i + 1])
#         idx = np.where(indices)[0]
#         idx = np.array(idx)
      
#         if np.any(idx):  # Check if there are elements in this range
#             Uinc_in_bin = Uinc[idx]
#             Uinc_m, uinc_se = np.mean(Uinc_in_bin), np.std(Uinc_in_bin)/len(Uinc_in_bin)
#             Uinc_mean.append(Uinc_m)
#             Uinc_stderr.append(uinc_se)
#         else:
#             Uinc_mean.append(np.nan)
#             Uinc_stderr.append(np.nan)
            
#     U_median = (Ubin[:-1] + Ubin[1:]) / 2 

#     return np.array(Uinc_mean), np.array(Uinc_stderr), np.array(U_median) 
    
# U_bin = np.linspace(0, max([np.max(u) for u in U_dpm])+1, 20)
# Uinc_mean, Uinc_stderr = defaultdict(list), defaultdict(list)
# for i in range(5):
#     Uinc_mean[i], Uinc_stderr[i], U_Uincplot = Bin_Uinc_U(Uinc_all_Omega[i], Udpm_all_Omega[i], U_bin)
    
# plt.figure(figsize=(6,5))
# for i in range(5):
#     plt.errorbar(U_Uincplot, Uinc_mean[i], yerr=Uinc_stderr[i], fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
# plt.xlabel(r'$U$ [m/s]', fontsize=14)
# plt.ylabel(r'$U_{inc}$ [m/s]', fontsize=14)
# plt.xlim(left=0);plt.ylim(bottom=0)
# plt.legend(fontsize=12)
# plt.tight_layout()   

# def power(U, a, b):
#     return a*U**b

# def linear(U, a):
#     return a*U

# # --- Fit ---
# # --- Remove NaN values before fitting ---
# def fitUincU(U_plot, Uinc_mean, Uinc_stderr):
#     valid_indices = ~np.isnan(Uinc_mean)  
#     U_clean = U_plot[valid_indices]       # Keep only valid U values
#     Uinc_clean = Uinc_mean[valid_indices]# Keep only valid theta values
#     stderr = Uinc_stderr[valid_indices]
#     valid_mask = np.where(~np.isnan(stderr))[0]
#     # 用该掩码过滤所有列表
#     U_clean = U_clean[valid_mask]
#     Uinc_clean = Uinc_clean[valid_mask]
#     stderr = stderr[valid_mask]
#     popt, _ = curve_fit(power, U_clean, Uinc_clean, maxfev=10000)  # no weight since stderr is not significant
#     a, b = popt 
#     return a, b

# #Dry
# a_dry, b_dry = fitUincU(U_Uincplot, Uinc_mean[0], Uinc_stderr[0])
# Uinc_wet = np.array([x for k in range(1, 5) for x in Uinc_mean[k]])
# Uinc_wetstderr = np.array([x for k in range(1, 5) for x in Uinc_stderr[k]])
# U_Uincplot_wet = np.tile(U_Uincplot,4)
# a_wet, b_wet = fitUincU(U_Uincplot_wet, Uinc_wet, Uinc_wetstderr)
# print(f'Uinc = {a_dry:.2f}U**{b_dry:.2f} for Dry')
# print(f'Uinc = {a_wet:.2f}U**{b_wet:.2f} for Moist')

# # Generate fitted curve
# U_fit = np.linspace(0, max(U_Uincplot), 100)
# Uinc_fit_dry = power(U_fit, a_dry, b_dry)
# Uinc_fit_wet = power(U_fit, a_wet, b_wet)
# plt.figure(figsize=(6,5))
# for i in range(5):
#     plt.errorbar(U_Uincplot, Uinc_mean[i], yerr=Uinc_stderr[i], fmt='o', capsize=5, label=rf'$\Omega$={Omega[i]}%', color=colors[i])
# plt.plot(U_fit, Uinc_fit_dry, '--', color=colors[0], label='Dry fit')
# plt.plot(U_fit, Uinc_fit_wet, '--', color=colors[-1], label='Moist fit')
# plt.xlabel(r'$U$ [m/s]', fontsize=14)
# plt.ylabel(r'$U_{inc}$ [m/s]', fontsize=14)
# plt.xlim(left=0);plt.ylim(bottom=0)
# plt.legend(fontsize=12)
# plt.tight_layout()