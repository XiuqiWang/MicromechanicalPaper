# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:36:39 2025

@author: WangX3
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from module import store_particle_id_data
from module import store_sal_id_data
from module import BinUimCOR
from module import BinThetaimCOR
from module import BinUimUd
from module import match_ejection_to_impact
from module import AverageOverShields
from module import get_ejection_ratios
from module import get_ejection_theta

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
colors = plt.cm.viridis(np.linspace(0, 1, 5))  # 5 colors
colors_n = plt.cm.plasma(np.linspace(0, 1, 5))
t_ver = np.linspace(dt, 5, 500)

#initialize
EDindices = defaultdict(list)
ME,MD,MoE,MoD = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
VExz_mean_t, VDxz_mean_t= defaultdict(list), defaultdict(list)
E_vector_t,VD_vector_t= defaultdict(list),defaultdict(list)

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
    ParticleID=np.linspace(num_p-10-299-1,num_p-10-1,300)
    ParticleID_int = ParticleID.astype(int)
    #cal erosion and deposition properties for each Omega
    #EDindices, E, VX, VExVector, VEzVector, VEx, VEz, ME, MD
    EDindices[i], ME[i], MD[i], VExz_mean_t[i], VDxz_mean_t[i], VD_vector_t[i], E_vector_t[i]=store_particle_id_data(data,ParticleID_int,coe_h,dt,N_inter,D)
    #cal rebound properties for each Omega
    ParticleID_sal=np.linspace(num_p-310,num_p-1,310)
    ParticleID_salint = ParticleID_sal.astype(int)
    X[i] = np.array([[time_step['Position'][i][0] for i in ParticleID_salint] for time_step in data])
    Z[i] = np.array([[time_step['Position'][i][2] for i in ParticleID_salint] for time_step in data])
    Par[i], VZ[i], exz_mean_t[i], ez_mean_t[i], VIM_mean_t[i], ThetaIM_mean_t[i], RIM[i], exz_vector_t[i], IM_vector_t[i]=store_sal_id_data(data,ParticleID_salint, coe_sal_h, dt, N_inter, D)
 
colors_map = plt.cm.viridis(np.linspace(0, 1, 10))
id_case = 24
plt.figure(figsize=(10, 15))
for i in range(300):
    #plt.plot(X[24][:,i]/D, t_ver, '.')   
    color_index = i % len(colors_map)
    if Par[id_case][i][2].any():
        plt.plot(X[id_case][Par[id_case][i][2][:,0],i]/D, t_ver[Par[id_case][i][2][:,0]], 'o',color=colors_map[color_index], fillstyle='none')
        plt.plot(X[id_case][Par[id_case][i][2][:,1],i]/D, t_ver[Par[id_case][i][2][:,1]], 'x',color=colors_map[color_index])
for i in range(300):
    if EDindices[id_case][i][0].any():
        plt.plot(X[id_case][EDindices[id_case][i][0],i]/D, t_ver[EDindices[id_case][i][0]], 's', fillstyle='none', color='k')
plt.xlabel('x/D [m]')
plt.ylabel('t [s]')

plt.figure(figsize=(10, 15))
for i in range(300):
    #plt.plot(X[24][:,i]/D, t_ver, '.')   
    color_index = i % len(colors_map)
    if Par[24][i][2].any():
        plt.plot(Z[id_case][Par[id_case][i][2][:,0],i]/D, t_ver[Par[id_case][i][2][:,0]], 'o',color=colors_map[color_index], fillstyle='none')
        plt.plot(Z[id_case][Par[id_case][i][2][:,1],i]/D, t_ver[Par[id_case][i][2][:,1]], 'x',color=colors_map[color_index])
# for i in range(300):
#     if EDindices[24][i][0].any():
#         plt.plot(X[EDindices[24][i][0],i]/D, t_ver[EDindices[24][i][0]], 'o')
plt.xlabel('z/D [m]')
plt.ylabel('t [s]')

exz_all,Vim_all,VD_all,Theta_all,impact_list,ejection_list = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
# T_range = np.array([1.3, 1.4, 1.8, 2, 2.2, 
#                     1.5, 1.75, 1.9, 2.2, 2.5,
#                     1.5, 1.5, 1.75, 2.25, 2.5,
#                     1.5, 1.75, 1.8, 2, 3.1, 
#                     1.8, 1.8, 2, 2.5, 3.5])
# N_range = ((T_range / 5) * N_inter).astype(int)
#N_range = np.full(25, 3.5/5*N_inter).astype(int)
N_range = np.full(25, 0).astype(int)
for i in range (25):
    exz_all[i] = [value for sublist in exz_vector_t[i][N_range[i]:] for value in sublist]
    Vim_all[i] = [value[0] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    VD_all[i] = [value for sublist in VD_vector_t[i][N_range[i]:] for value in sublist]
    Theta_all[i] = [value[7] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    IDim = [value[1] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    IDre = [value[2] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    xim = [value[3] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    xre = [value[4] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    xcol = [value[5] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Pim = [value[6] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    impact_list[i] = [IDim, IDre, xim, xre, Vim_all[i], xcol, Pim, Theta_all[i]]
    vE = [value[0] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    IDE = [value[1] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    xE = [value[2] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    PE = [value[3] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    ejection_list[i] = [IDE, xE, vE, PE]
    #print('Ne/Nim',len(IDE)/len(IDim))

Vim_all_values = [value for sublist in Vim_all.values() for value in sublist]
Vim_bin = np.linspace(min(Vim_all_values), max(Vim_all_values), 8)
Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
Vimde_bin = np.linspace(min(Vimde_all_values), max(Vimde_all_values), 8)
matched_Vim, matched_thetaim, matched_NE, matched_UE = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
CORmean,CORstd,Uimplot = defaultdict(list),defaultdict(list),defaultdict(list)
COR_theta_mean, COR_theta_std, Thetaimplot = defaultdict(list),defaultdict(list),defaultdict(list)
Pr,Uplot = defaultdict(list),defaultdict(list)
NE_mean,UE_mean,UE_std, Uplot_NE=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
NE_theta_mean,UE_theta_mean,UE_theta_std, Thetaplot_NE=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
impact_ejection_list = defaultdict(list)
for i in range (25):
    impact_ejection_list=match_ejection_to_impact(impact_list[i], ejection_list[i], dt)
    matched_Vim[i] = [element for element in impact_ejection_list[0]]
    matched_thetaim[i] = [element for element in impact_ejection_list[1]]
    matched_NE[i] = [element for element in impact_ejection_list[2]]
    matched_UE[i] = [element for element in impact_ejection_list[3]]
    
for i in range (25):
    CORmean[i],CORstd[i],Uimplot[i] = BinUimCOR(Vim_all[i],exz_all[i],8)
    COR_theta_mean[i], COR_theta_std[i], Thetaimplot[i] = BinThetaimCOR(Theta_all[i], exz_all[i], 8)
    Pr[i],Uplot[i] = BinUimUd(Vim_all[i],VD_all[i],8)    
    NE_mean[i], UE_mean[i], UE_std[i], Uplot_NE[i] = get_ejection_ratios(matched_Vim[i], matched_NE[i], matched_UE[i], 8)
    NE_theta_mean[i], UE_theta_mean[i], UE_theta_std[i], Thetaplot_NE[i] = get_ejection_theta(matched_thetaim[i], matched_NE[i], matched_UE[i], 8)
    
constant = np.sqrt(9.81*D)  
# #plot COR - Uim
# plt.figure(figsize=(16, 13))
# for i in range(5):
#     plt.subplot(2,3,i+1)
#     for j in range(5):
#         plt.errorbar(Uimplot[i*5+j]/constant, CORmean[i*5+j], yerr=CORstd[i*5+j], fmt='o', capsize=5, label=f"$\\Omega$={Omega[j]}%",color=colors[j])
#     plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
#     plt.ylabel(r'$e$ [-]', fontsize=14)
#     plt.title(r'$\tilde{\Theta}=0.0$'f"{i+2}")
#     plt.xlim(0,225)
#     plt.ylim(0,3.5)
# plt.legend(fontsize=14)
# plt.show()

# #plot COR - thetaim
# plt.figure(figsize=(16, 13))
# for i in range(5):
#     plt.subplot(2,3,i+1)
#     for j in range(5):
#         plt.errorbar(Thetaimplot[i*5+j], COR_theta_mean[i*5+j], yerr=COR_theta_std[i*5+j], fmt='o', capsize=5, label=f"$\\Omega$={Omega[j]}%",color=colors[j])
#     plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
#     plt.ylabel(r'$e$ [-]', fontsize=14)
#     plt.title(r'$\tilde{\Theta}=0.0$'f"{i+2}")
#     plt.xlim(0,55)
#     plt.ylim(0,3.5)
# plt.legend(fontsize=14)
# plt.ylim(0,2)
# plt.show()

# #plot Pr - Uim 
# plt.figure(figsize=(16, 13))
# for i in range(5):
#     plt.subplot(2,3,i+1)
#     for j in range(5):
#         plt.plot(Uplot[i*5+j]/constant, Pr[i*5+j], 'o', label=f"$\\Omega$={Omega[j]}%",color=colors[j])
#     plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
#     plt.ylabel(r'$P_\mathrm{r}$ [-]', fontsize=14)
#     plt.title(r'$\tilde{\Theta}=0.0$'f"{i+2}")
#     plt.xlim(0,200)
#     plt.ylim(0,1)
# plt.legend(fontsize=14)
# plt.show()


# #plot NE - Uim 
# plt.figure(figsize=(16, 13))
# for i in range(5):
#     plt.subplot(2,3,i+1)
#     for j in range(5):
#         plt.plot(Uplot_NE[i]/constant, NE_mean[i*5+j], 'o', label=f"$\\Omega$={Omega[j]}%",color=colors[j])
#     plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
#     plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
#     plt.title(r'$\tilde{\Theta}=0.0$'f"{i+2}")
#     plt.xlim(0,120)
#     plt.ylim(0,1.4)
# plt.legend(fontsize=14)
# plt.show()

# #plot UE - Uim
# plt.figure(figsize=(16, 13))
# for i in range(5):
#     plt.subplot(2,3,i+1)
#     for j in range(5):
#         plt.errorbar(Uplot_NE[i]/constant, UE_mean[i*5+j]/constant, yerr=UE_std[i*5+j]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[j]}%",color=colors[j])
#     plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
#     plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
#     plt.title(r'$\tilde{\Theta}=0.0$'f"{i+2}")
#     plt.xlim(0,120)
#     plt.ylim(0,30)
# plt.legend(fontsize=14)
# plt.show()

# #plot NE - theta
# plt.figure(figsize=(16, 13))
# for i in range(5):
#     plt.subplot(2,3,i+1)
#     for j in range(5):
#         plt.plot(Thetaplot_NE[i], NE_theta_mean[i*5+j], 'o', label=f"$\\Omega$={Omega[j]}%",color=colors[j])
#     plt.xlabel(r'$\Theta_{im}/\sqrt{gd}$ [$\circ$]', fontsize=14)
#     plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
#     plt.title(r'$\tilde{\Theta}=0.0$'f"{i+2}")
#     #plt.xlim(0,120)
#     plt.ylim(0,1.4)
# plt.legend(fontsize=14)
# plt.show()

# #plot UE - theta
# plt.figure(figsize=(16, 13))
# for i in range(5):
#     plt.subplot(2,3,i+1)
#     for j in range(5):
#         plt.errorbar(Thetaplot_NE[i], UE_theta_mean[i*5+j]/constant, yerr=UE_theta_std[i*5+j]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[j]}%",color=colors[j])
#     plt.xlabel(r'$\Theta_{im}/\sqrt{gd}$ [$\circ$]', fontsize=14)
#     plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
#     plt.title(r'$\tilde{\Theta}=0.0$'f"{i+2}")
#     #plt.xlim(0,120)
#     plt.ylim(0,30)
# plt.legend(fontsize=14)
# plt.show()

#combine the values from all Shields numbers
Vim_all_Omega, exz_all_Omega, Theta_all_Omega, VD_all_Omega, matched_Vim_Omega, matched_Thetaim_Omega, matched_NE_Omega, matched_UE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),
for i in range (5): #loop over Omega 0-20 %
    selected_indices = list(range(i, 25, 5))  # Get indices like [0,5,10,15,20], [1,6,11,16,21], etc.
    Vim_all_Omega[i] = np.concatenate([Vim_all[j] for j in selected_indices]).tolist()
    exz_all_Omega[i] = np.concatenate([exz_all[j] for j in selected_indices]).tolist()
    Theta_all_Omega[i] = np.concatenate([Theta_all[j] for j in selected_indices]).tolist()
    VD_all_Omega[i] = np.concatenate([VD_all[j] for j in selected_indices]).tolist()
    matched_Vim_Omega[i] = np.concatenate([matched_Vim[j] for j in selected_indices]).tolist()
    matched_Thetaim_Omega[i] = np.concatenate([matched_thetaim[j] for j in selected_indices]).tolist()
    matched_NE_Omega[i] = np.concatenate([matched_NE[j] for j in selected_indices]).tolist()
    matched_UE_Omega[i] = np.concatenate([matched_UE[j] for j in selected_indices]).tolist()

CORmean_Omega,CORstd_Omega,Uimplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list)
COR_theta_mean_Omega, COR_theta_std_Omega, Thetaimplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list)
Pr_Omega,Uplot_Omega = defaultdict(list), defaultdict(list)
NE_mean_Omega, UE_mean_Omega, UE_std_Omega, Uplot_NE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
NE_theta_mean_Omega, UE_theta_mean_Omega, UE_theta_std_Omega, Thetaplot_NE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
for i in range (5): #loop over Omega 0-20 %
    CORmean_Omega[i],CORstd_Omega[i],Uimplot_Omega[i] = BinUimCOR(Vim_all_Omega[i],exz_all_Omega[i],8)
    COR_theta_mean_Omega[i], COR_theta_std_Omega[i], Thetaimplot_Omega[i] = BinThetaimCOR(Theta_all_Omega[i], exz_all_Omega[i], 8)
    Pr_Omega[i],Uplot_Omega[i] = BinUimUd(Vim_all_Omega[i],VD_all_Omega[i],8)    
    NE_mean_Omega[i], UE_mean_Omega[i], UE_std_Omega[i], Uplot_NE_Omega[i]=get_ejection_ratios(matched_Vim_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], 8)
    NE_theta_mean_Omega[i], UE_theta_mean_Omega[i], UE_theta_std_Omega[i], Thetaplot_NE_Omega[i]=get_ejection_theta(matched_Thetaim_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], 8)

# 全局设置 x 轴刻度字体大小
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14  # 也可以同时设置 y 轴刻度字体大小
plt.figure(figsize=(17,11))
plt.subplot(2,2,1)
for i in range(5):
    plt.errorbar(Uimplot_Omega[i]/constant, CORmean_Omega[i], yerr=CORstd_Omega[i], fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlim(0,200)
plt.ylim(0, 3.5)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=24)
plt.ylabel(r'$e$ [-]', fontsize=24)
plt.legend(fontsize=20)
plt.subplot(2,2,2)
for i in range(5):
    plt.plot(Uplot_Omega[i]/constant, Pr_Omega[i], 'o', label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlim(0,200)
plt.ylim(0, 1)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=24)
plt.ylabel(r'$P_\mathrm{r}$ [-]', fontsize=24)
plt.subplot(2,2,3)
for i in range(5):
    plt.plot(Uplot_NE_Omega[i]/constant, NE_mean_Omega[i], 'o', label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlim(0,200)
plt.ylim(0, 1.2)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=24)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=24)
plt.subplot(2,2,4)
for i in range(5):
    plt.errorbar(Uplot_NE_Omega[i]/constant, UE_mean_Omega[i]/constant, yerr=UE_std_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlim(0,200)
plt.ylim(0, 20)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=24)
plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=24)
plt.tight_layout()
plt.show()

plt.figure()
for i in range(5):
    plt.errorbar(Thetaimplot_Omega[i], COR_theta_mean_Omega[i], yerr=COR_theta_std_Omega[i], fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
plt.ylabel(r'$e$ [-]', fontsize=14)
plt.xlim(0,24)

plt.figure()
for i in range(5):
    plt.plot(Thetaplot_NE_Omega[i], NE_mean_Omega[i], 'o', label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)

plt.figure()
for i in range(5):
    plt.errorbar(Thetaimplot_Omega[i], UE_theta_mean_Omega[i]/constant, yerr=UE_theta_std_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)

#dependency of COR on Omega
COR_m, COR_s = defaultdict(list),defaultdict(list)
for i in range (5):
    COR_m[i] = np.nanmean(exz_all_Omega[i])
    COR_s[i] = np.nanstd(exz_all_Omega[i])

cor_values = [COR_m[i] for i in range(len(Omega))]  
plt.figure()
plt.plot(Omega, cor_values, '-')
plt.xlabel(r'$\Omega$ [$\%$]')
plt.ylabel(r'$e$ [-]')
 

#compare with static-bed experiment of Ge (2024)
#try filtering the CORs with Vim=4 m/s (80)
Theta_Ge = [7, 9, 12, 19]
COR_Ge = [0.62, 0.57, 0.49, 0.47]
Theta_GeWet1 = [11, 14, 18]#1.45%
COR_GeWet1 = [0.65, 0.625, 0.525]
Theta_GeWet2 = [7, 13, 21]#22.79%
COR_GeWet2 = [0.69, 0.67, 0.55]
COR_test, COR_test_std, Theta_test = defaultdict(list), defaultdict(list), defaultdict(list)
for i in [0,4]:
    valid_indices = [j for j, val in enumerate(Vim_all_Omega[i]) if 3.5 <= val <= 4.5]
    COR_test[i], COR_test_std[i], Theta_test[i] = BinThetaimCOR([Theta_all_Omega[i][j] for j in valid_indices], [exz_all_Omega[i][j] for j in valid_indices], 8)
plt.figure(figsize=(7,6))
for i in [0,4]:
    plt.errorbar(Theta_test[i], COR_test[i], yerr=COR_test_std[i], fmt='o',capsize=5, label=f'$\Omega$={Omega[i]}% (this study)',color=colors[i])
plt.plot(Theta_Ge, COR_Ge, 'dk', label=r'$\Omega$=0$\%$ (Ge et al., 2024)')
# plt.plot(Theta_GeWet1, COR_GeWet1, '*k', label=r'$\Omega$=1.45$\%$ (Ge et al., 2024)')
plt.plot(Theta_GeWet2, COR_GeWet2, 'sk', label=r'$\Omega$=22.79$\%$ (Ge et al., 2024)')
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
plt.ylabel(r'$e$ [-]', fontsize=14)
plt.legend(fontsize=11)

#try filtering the CORs with thetaim=11.5 degree
U_Ge = [125, 225, 310]
UE_Ge = [0.49, 0.52, 0.75]
U_GeWet = [60, 145]#22.79%
UE_GeWet = [0.7, 2.7]
NE_test, UE_test, UE_test_std, U_testNE = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
for i in [0,4]:
    valid_indices = [j for j, val in enumerate(matched_Thetaim_Omega[i]) if 10 <= val <= 13]
    NE_test[i], UE_test[i], UE_test_std[i], U_testNE[i] = get_ejection_ratios([matched_Vim_Omega[i][j] for j in valid_indices], [matched_NE_Omega[i][j] for j in valid_indices], [matched_UE_Omega[i][j] for j in valid_indices], 8)
plt.figure(figsize=(7,6))
for i in [0,4]:
    plt.errorbar(U_testNE[i]/constant, UE_test[i]/constant, yerr=UE_test_std[i]/constant, fmt='o',capsize=5, label=f'$\Omega$={Omega[i]}% (this study)',color=colors[i])
plt.plot(U_Ge, UE_Ge/np.sqrt(9.81*0.0003), 'dk', label=r'$\Omega$=0$\%$ (Ge et al., 2024)')
plt.plot(U_GeWet, UE_GeWet/np.sqrt(9.81*0.0003), 'sk', label=r'$\Omega$=22.79$\%$ (Ge et al., 2024)')
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
plt.legend(fontsize=11)



#distribution of CORs
# exz_whole, exz_steady = defaultdict(list),defaultdict(list)
# exz_whole_m, exz_whole_s, exz_steady_m, exz_steady_s = [],[], [], []
# theta_whole, theta_steady = defaultdict(list),defaultdict(list)
# theta_whole_m, theta_whole_s, theta_steady_m, theta_steady_s = [], [], [], []
# for i in range (5):
#     exz_whole[i] = [value for sublist in exz_vector_t[i+15][0:] for value in sublist]
#     exz_whole_m.append(np.nanmean(exz_whole[i]))
#     exz_whole_s.append(np.nanstd(exz_whole[i]))
#     exz_steady[i] = [value for sublist in exz_vector_t[i+15][int(3.5/5* N_inter):] for value in sublist]
#     exz_steady_m.append(np.nanmean(exz_steady[i]))
#     exz_steady_s.append(np.nanstd(exz_steady[i]))
    
#     theta_whole[i] = [value for sublist in Thetaim_vector_t[i+15][0:] for value in sublist]
#     theta_whole_m.append(np.nanmean(theta_whole[i]))
#     theta_whole_s.append(np.nanstd(theta_whole[i]))
#     theta_steady[i] = [value for sublist in Thetaim_vector_t[i+15][int(3.5/5* N_inter):] for value in sublist]
#     theta_steady_m.append(np.nanmean(theta_steady[i]))
#     theta_steady_s.append(np.nanstd(theta_steady[i]))
    
# plt.figure(figsize=(10, 15))
# plt.subplot(2, 1, 1)
# for i in range(5):
#     # Calculate histogram (raw counts)
#     counts, bin_edges = np.histogram(exz_whole[i], bins=50)
#     # Normalize the counts so their sum is 1
#     normalized_counts = counts / np.sum(counts)
#     # Create the step plot
#     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
#     # Prepare data for staircase plot
#     #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
#     plt.xlim(-0.5,7.5)
# plt.xlabel('$e_\mathrm{xz,whole}$ [-]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)
# plt.legend(fontsize=14)
# plt.subplot(2, 1, 2)
# for i in range(5):
#     # Calculate histogram (raw counts)
#     counts, bin_edges = np.histogram(exz_steady[i], bins=50)
#     # Normalize the counts so their sum is 1
#     normalized_counts = counts / np.sum(counts)
#     # Create the step plot
#     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
#     # Prepare data for staircase plot
#     #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
#     plt.xlim(-0.5,7.5)
# plt.xlabel('$e_\mathrm{xz,steady}$ [-]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)
# plt.show()

# plt.figure()
# plt.errorbar(Omega, exz_whole_m, yerr=exz_whole_s, fmt='o', capsize=10, label='whole simulation time')
# plt.errorbar(Omega, exz_steady_m, yerr=exz_steady_s, fmt='o', capsize=10, label='steady state')
# plt.xlabel(r'$\Omega$ [%]')
# plt.ylabel(r'$e_\mathrm{xz}$ [-]')
# plt.legend()
# plt.show()

# plt.figure(figsize=(10, 15))
# plt.subplot(2, 1, 1)
# for i in range(5):
#     # Calculate histogram (raw counts)
#     counts, bin_edges = np.histogram(theta_whole[i], bins=50)
#     # Normalize the counts so their sum is 1
#     normalized_counts = counts / np.sum(counts)
#     # Create the step plot
#     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
#     # Prepare data for staircase plot
#     #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
#     plt.xlim(0,75)
# plt.xlabel(r'$\theta_\mathrm{im,whole}$ [$\circ$]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)
# plt.legend(fontsize=14)
# plt.subplot(2, 1, 2)
# for i in range(5):
#     # Calculate histogram (raw counts)
#     counts, bin_edges = np.histogram(theta_steady[i], bins=50)
#     # Normalize the counts so their sum is 1
#     normalized_counts = counts / np.sum(counts)
#     # Create the step plot
#     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
#     # Prepare data for staircase plot
#     #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
#     plt.xlim(0,75)
# plt.xlabel(r'$\theta_\mathrm{im,steady}$ [$\circ$]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)
# plt.show()

# plt.figure()
# plt.errorbar(Omega, theta_whole_m, yerr=theta_whole_s, fmt='o', capsize=10, label='whole simulation time')
# plt.errorbar(Omega, theta_steady_m, yerr=theta_steady_s, fmt='o', capsize=10, label='steady state')
# plt.xlabel(r'$\Omega$ [%]')
# plt.ylabel(r'$\theta_\mathrm{im}$ [$\circ$]')
# plt.legend()
# plt.show()