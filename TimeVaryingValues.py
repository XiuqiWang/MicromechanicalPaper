# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 11:29:00 2025

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
ThetaE_all, UE_all = defaultdict(list), defaultdict(list)
N_range = np.full(25, 0).astype(int)
for i in range (25):
    exz_all[i] = [value for sublist in exz_vector_t[i][N_range[i]:] for value in sublist]
    Vim_all[i] = [value[0] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    VD_all[i] = [value[0] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
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
Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
Vimde_bin = np.linspace(min(Vimde_all_values), max(Vimde_all_values), 10)
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
Vim_all_Omega, exz_all_Omega, Theta_all_Omega, Thetare_all_Omega, VD_all_Omega, ThetaD_all_Omega, matched_Vim_Omega, matched_Thetaim_Omega, matched_NE_Omega, matched_UE_Omega, matched_EE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),defaultdict(list),defaultdict(list)
ThetaE_all_Omega, UE_all_Omega = defaultdict(list), defaultdict(list)
matched_thetaE_Omega = defaultdict(list)
for i in range (5): #loop over Omega 0-20 %
    selected_indices = list(range(i, 25, 5))  # Get indices like [0,5,10,15,20], [1,6,11,16,21], etc.
    Vim_all_Omega[i] = np.concatenate([Vim_all[j] for j in selected_indices]).tolist()
    exz_all_Omega[i] = np.concatenate([exz_all[j] for j in selected_indices]).tolist()
    Theta_all_Omega[i] = np.concatenate([Theta_all[j] for j in selected_indices]).tolist()
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

CORmean_Omega,CORstd_Omega,CORstderr_Omega,Uimplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
COR_theta_mean_Omega, COR_theta_std_Omega, Thetaremean_Omega, Thetarestderr_Omega, Thetaimplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
Pr_Omega,Uplot_Omega, N_PrUre = defaultdict(list), defaultdict(list), defaultdict(list)
Pr_theta_Omega,Theta_pr_Omega = defaultdict(list), defaultdict(list)
NE_mean_Omega, UE_mean_Omega, UE_std_Omega, UE_stderr_Omega, Uplot_NE_Omega, N_Einbin = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),
ThetaE_mean_Omega, ThetaE_stderr_Omega = defaultdict(list), defaultdict(list)
NE_theta_mean_Omega, UE_theta_mean_Omega, UE_theta_std_Omega, Thetaplot_NE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
EE_mean_Omega, EE_stderr_Omega = defaultdict(list), defaultdict(list)
for i in range (5): #loop over Omega 0-20%
    CORmean_Omega[i],CORstd_Omega[i],CORstderr_Omega[i],Thetaremean_Omega[i],Thetarestderr_Omega[i],Uimplot_Omega[i] = module.BinUimCOR_equalbinsize(Vim_all_Omega[i],exz_all_Omega[i],Thetare_all_Omega[i],Vim_bin)
    # COR_theta_mean_Omega[i], COR_theta_std_Omega[i], Thetaimplot_Omega[i] = module.BinThetaimCOR_equalbinsize(Theta_all_Omega[i], exz_all_Omega[i], Thetaim_bin)
    Pr_Omega[i],Uplot_Omega[i],N_PrUre[i] = module.BinUimUd_equalbinsize(Vim_all_Omega[i],VD_all_Omega[i],Vimde_bin)   
    # Pr_theta_Omega[i],Theta_pr_Omega[i] = module.BinThetaimThetad_equalbinsize(Theta_all_Omega[i],ThetaD_all_Omega[i],Thetaimde_bin)   
    NE_mean_Omega[i], UE_mean_Omega[i], UE_std_Omega[i], UE_stderr_Omega[i], ThetaE_mean_Omega[i], ThetaE_stderr_Omega[i], Uplot_NE_Omega[i], N_Einbin[i]=module.get_ejection_ratios_equalbinsize(matched_Vim_Omega[i], VD_all_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], matched_thetaE_Omega[i], Vimde_bin)#matched_EE_Omega[i]
    # NE_theta_mean_Omega[i], UE_theta_mean_Omega[i], UE_theta_std_Omega[i], Thetaplot_NE_Omega[i]=module.get_ejection_theta_equalbinsize(matched_Thetaim_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], Thetaim_bin)

#plot Uim-t
fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axs = axs.flatten()

for i in range(5):  # 5 groups
    ax = axs[i]
    for j in range(5):  # 5 elements per group
        index = i * 5 + j
        ax.plot(t_inter[1:], VIM_mean_t[index], label=fr'$\Omega$={Omega[j]} $\%$')
    ax.set_title(f'$\Theta$=0.0{i+2}')
    ax.set_ylabel(r'$U_{im}$ [m/s]')
    ax.set_xlabel(r't [s]')
    ax.set_xlim(left=0)
    ax.set_ylim(0,12)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

#Steady Uim
Uim_steady = []
for i in range(25):
    Uims = np.nanmean(VIM_mean_t[i][int(3/5*N_inter):])
    Uim_steady.append(Uims)
