# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:36:39 2025

@author: WangX3
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from collections import defaultdict
import module
from scipy.interpolate import griddata

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
    ParticleID=np.linspace(num_p-10-299-1,num_p-10-1,300)
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

exz_all,Vim_all,VD_all,ThetaD_all,Theta_all,impact_list,ejection_list = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
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
    VD_all[i] = [value[0] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    ThetaD_all[i] = [value[1] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
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
Vim_bin = np.linspace(min(Vim_all_values), max(Vim_all_values), 20)
Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
Vimde_bin = np.linspace(min(Vimde_all_values), max(Vimde_all_values), 20)
Thetaimde_all_values = [value for sublist in Theta_all.values() for value in sublist] + [value for sublist in ThetaD_all.values() for value in sublist]
Thetaimde_bin = np.linspace(min(Thetaimde_all_values), max(Thetaimde_all_values), 20)
Thetaim_all_values = [value for sublist in Theta_all.values() for value in sublist]
Thetaim_bin = np.linspace(min(Thetaim_all_values), max(Thetaim_all_values), 20)
matched_Vim, matched_thetaim, matched_NE, matched_UE = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
CORmean,CORstd,Uimplot = defaultdict(list),defaultdict(list),defaultdict(list)
COR_theta_mean, COR_theta_std, Thetaimplot = defaultdict(list),defaultdict(list),defaultdict(list)
Pr,Uplot = defaultdict(list),defaultdict(list)
NE_mean,UE_mean,UE_std, Uplot_NE=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
NE_theta_mean,UE_theta_mean,UE_theta_std, Thetaplot_NE=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
impact_ejection_list = defaultdict(list)
for i in range (25):
    impact_ejection_list=module.match_ejection_to_impact(impact_list[i], ejection_list[i], dt)
    matched_Vim[i] = [element for element in impact_ejection_list[0]]
    matched_thetaim[i] = [element for element in impact_ejection_list[1]]
    matched_NE[i] = [element for element in impact_ejection_list[2]]
    matched_UE[i] = [element for element in impact_ejection_list[3]]
    
for i in range (25):
    CORmean[i],CORstd[i],Uimplot[i] = module.BinUimCOR(Vim_all[i],exz_all[i],8)
    COR_theta_mean[i], COR_theta_std[i], Thetaimplot[i] = module.BinThetaimCOR(Theta_all[i], exz_all[i], 8)
    Pr[i],Uplot[i] = module.BinUimUd(Vim_all[i],VD_all[i],8)    
    NE_mean[i], UE_mean[i], UE_std[i], Uplot_NE[i] = module.get_ejection_ratios(matched_Vim[i], matched_NE[i], matched_UE[i], 8)
    NE_theta_mean[i], UE_theta_mean[i], UE_theta_std[i], Thetaplot_NE[i] = module.get_ejection_theta(matched_thetaim[i], matched_NE[i], matched_UE[i], 8)
    
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
Vim_all_Omega, exz_all_Omega, Theta_all_Omega, VD_all_Omega, ThetaD_all_Omega, matched_Vim_Omega, matched_Thetaim_Omega, matched_NE_Omega, matched_UE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),
for i in range (5): #loop over Omega 0-20 %
    selected_indices = list(range(i, 25, 5))  # Get indices like [0,5,10,15,20], [1,6,11,16,21], etc.
    Vim_all_Omega[i] = np.concatenate([Vim_all[j] for j in selected_indices]).tolist()
    exz_all_Omega[i] = np.concatenate([exz_all[j] for j in selected_indices]).tolist()
    Theta_all_Omega[i] = np.concatenate([Theta_all[j] for j in selected_indices]).tolist()
    VD_all_Omega[i] = np.concatenate([VD_all[j] for j in selected_indices]).tolist()
    ThetaD_all_Omega[i] = np.concatenate([ThetaD_all[j] for j in selected_indices]).tolist()
    matched_Vim_Omega[i] = np.concatenate([matched_Vim[j] for j in selected_indices]).tolist()
    matched_Thetaim_Omega[i] = np.concatenate([matched_thetaim[j] for j in selected_indices]).tolist()
    matched_NE_Omega[i] = np.concatenate([matched_NE[j] for j in selected_indices]).tolist()
    matched_UE_Omega[i] = np.concatenate([matched_UE[j] for j in selected_indices]).tolist()


CORmean_Omega,CORstd_Omega,Uimplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list)
COR_theta_mean_Omega, COR_theta_std_Omega, Thetaimplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list)
Pr_Omega,Uplot_Omega = defaultdict(list), defaultdict(list)
Pr_theta_Omega,Theta_pr_Omega = defaultdict(list), defaultdict(list)
NE_mean_Omega, UE_mean_Omega, UE_std_Omega, Uplot_NE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
NE_theta_mean_Omega, UE_theta_mean_Omega, UE_theta_std_Omega, Thetaplot_NE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
for i in range (5): #loop over Omega 0-20 %
    CORmean_Omega[i],CORstd_Omega[i],Uimplot_Omega[i] = module.BinUimCOR_equalbinsize(Vim_all_Omega[i],exz_all_Omega[i],Vim_bin)
    COR_theta_mean_Omega[i], COR_theta_std_Omega[i], Thetaimplot_Omega[i] = module.BinThetaimCOR_equalbinsize(Theta_all_Omega[i], exz_all_Omega[i], Thetaim_bin)
    Pr_Omega[i],Uplot_Omega[i] = module.BinUimUd_equalbinsize(Vim_all_Omega[i],VD_all_Omega[i],Vimde_bin)   
    Pr_theta_Omega[i],Theta_pr_Omega[i] = module.BinThetaimThetad_equalbinsize(Theta_all_Omega[i],ThetaD_all_Omega[i],Thetaimde_bin)   
    NE_mean_Omega[i], UE_mean_Omega[i], UE_std_Omega[i], Uplot_NE_Omega[i]=module.get_ejection_ratios_equalbinsize(matched_Vim_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], Vim_bin)
    NE_theta_mean_Omega[i], UE_theta_mean_Omega[i], UE_theta_std_Omega[i], Thetaplot_NE_Omega[i]=module.get_ejection_theta_equalbinsize(matched_Thetaim_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], Thetaim_bin)

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
plt.ylim(0, 40)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=24)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=24)
plt.tight_layout()
plt.show()

#scatter color when equal bin size
Uim_scatter = np.concatenate(list(Uimplot_Omega.values()))
Thetaim_scatter = np.concatenate(list(Thetaimplot_Omega.values()))
Omega_scatter = np.repeat(Omega, 19)
COR_Uim_scatter = np.concatenate(list(CORmean_Omega.values()))
COR_Thetaim_scatter = np.concatenate(list(COR_theta_mean_Omega.values()))
# Remove NaN values
Uim_valid = Uim_scatter[~np.isnan(COR_Uim_scatter)]
Omega_Uim_valid = Omega_scatter[~np.isnan(COR_Uim_scatter)]
COR_Uim_valid = COR_Uim_scatter[~np.isnan(COR_Uim_scatter)]
Thetaim_valid = Thetaim_scatter[~np.isnan(COR_Thetaim_scatter)]
Omega_thetaim_valid = Omega_scatter[~np.isnan(COR_Thetaim_scatter)]
COR_thetaim_valid = COR_Thetaim_scatter[~np.isnan(COR_Thetaim_scatter)]

fig, ax = plt.subplots()
sc = ax.scatter(Uim_valid, Omega_Uim_valid, c=COR_Uim_valid, cmap='viridis',alpha=0.75)
cbar = plt.colorbar(sc)
cbar.set_label(r'$e$ [-]', fontsize=12)
# Create Delaunay triangulation for contouring
triang = tri.Triangulation(Uim_valid, Omega_Uim_valid)
# Contour plot (for filtered valid data)
# 绘制等值线
contour_levels = [0.6, 0.7, 0.9, 1.4]
contour = ax.tricontour(triang, COR_Uim_valid, levels=contour_levels, colors='black', linewidths=0.7)
# 添加等值线标签
ax.clabel(contour, fmt='%.1f', fontsize=10)
plt.ylim(-0.5,20.5)
plt.xlim(-1,13)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=12)
plt.ylabel(r'$\Omega$ [$\%$]', fontsize=12)


plt.figure(figsize=(17,11))
plt.subplot(2,2,1)
for i in range(5):
    plt.errorbar(Thetaimplot_Omega[i], COR_theta_mean_Omega[i], yerr=COR_theta_std_Omega[i], fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=24)
plt.ylabel(r'$e$ [-]', fontsize=24)
plt.legend(fontsize=20)
plt.subplot(2,2,2)
for i in range(5):
    plt.plot(Theta_pr_Omega[i], Pr_theta_Omega[i], 'o', label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=24)
plt.ylabel(r'$P_\mathrm{r}$ [-]', fontsize=24)
plt.subplot(2,2,3)
for i in range(5):
    plt.plot(Thetaplot_NE_Omega[i], NE_theta_mean_Omega[i], 'o', label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=24)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=24)
plt.subplot(2,2,4)
for i in range(5):
    plt.errorbar(Thetaplot_NE_Omega[i], UE_theta_mean_Omega[i]/constant, yerr=UE_theta_std_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=24)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=24)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
sc = ax.scatter(Thetaim_valid, Omega_thetaim_valid, c=COR_thetaim_valid, cmap='viridis',alpha=0.75)
cbar = plt.colorbar(sc)
cbar.set_label(r'$e$ [-]', fontsize=12)
# Create Delaunay triangulation for contouring
triang = tri.Triangulation(Thetaim_valid, Omega_thetaim_valid)
# Contour plot (for filtered valid data)
# 绘制等值线
contour_levels = [0.7, 1, 3, 4]
contour = ax.tricontour(triang, COR_thetaim_valid, levels=contour_levels, colors='black', linewidths=0.7)
# 添加等值线标签
ax.clabel(contour, fmt='%.1f', fontsize=10)
plt.ylim(-0.5,20.5)
plt.xlim(-5,85)
plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=12)
plt.ylabel(r'$\Omega$ [$\%$]', fontsize=12)

#dependency of COR on Omega
# COR_m, COR_s = defaultdict(list),defaultdict(list)
# for i in range (5):
#     COR_m[i] = np.nanmean(exz_all_Omega[i])
#     COR_s[i] = np.nanstd(exz_all_Omega[i])

# cor_values = [COR_m[i] for i in range(len(Omega))]  
# plt.figure()
# plt.plot(Omega, cor_values, '-')
# plt.xlabel(r'$\Omega$ [$\%$]')
# plt.ylabel(r'$e$ [-]')
 

#Uim - Thetaim apply the colormesh to COR and NE!
mean_thetaim, std_thetaim, Uthetaplot = defaultdict(list),defaultdict(list),defaultdict(list)
for i in range(5):
    mean_thetaim[i], std_thetaim[i], Uthetaplot[i] = module.match_Uim_thetaim(matched_Vim_Omega[i], matched_Thetaim_Omega[i], Vim_bin)

plt.figure(figsize=(8,5))
for i in range(5):
    plt.errorbar(Uthetaplot[i]/constant, mean_thetaim[i], yerr=std_thetaim[i], fmt='o',capsize=5, label=f'$\Omega$={Omega[i]}%',color=colors[i])
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=18)
plt.ylabel(r'$\theta_{im}$ [$\circ$]', fontsize=18)
plt.legend(fontsize=14)

Utheta_scatter = np.concatenate(list(Uthetaplot.values()))
theta_scatter = np.concatenate(list(mean_thetaim.values())) 
theta_scatter_valid = theta_scatter[~np.isnan(theta_scatter)]  
Utheta_scatter_valid = Utheta_scatter[~np.isnan(theta_scatter)]/constant
Omega_scatter_valid = Omega_scatter[~np.isnan(theta_scatter)]
    
fig, ax = plt.subplots()
# grid_x, grid_y = np.meshgrid(
#     np.linspace(Utheta_scatter_valid.min(), Utheta_scatter_valid.max(), 100),
#     np.linspace(Omega_scatter_valid.min(), Omega_scatter_valid.max(), 100)
# )

# # 进行插值
# grid_z = griddata((Utheta_scatter_valid, Omega_scatter_valid), theta_scatter_valid, (grid_x, grid_y), method='cubic')

# # 绘制等高线色块图
# plt.figure(figsize=(6,5))
# contour = plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='viridis')

# 颜色条
# cbar = plt.colorbar(contour)
# cbar.set_label(r'$\theta_{im}$ [°]')
sc = ax.scatter(Utheta_scatter, Omega_scatter, c=theta_scatter, cmap='viridis',alpha=0.75)
cbar = plt.colorbar(sc)
cbar.set_label(r'$\theta_{im}$ [$\circ$]', fontsize=12)
# for i in range(5):
#     plt.errorbar(Uthetaplot[i]/constant, mean_thetaim[i], yerr=std_thetaim[i], fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=12)
plt.ylabel(r'$\Omega$ [$\%$]', fontsize=12)
plt.show()
# plt.legend(fontsize=14)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns
for i in range(5):
    ax = axes[i // 3, i % 3]  # Convert i to 2D grid indexing
    sc = ax.scatter(Vim_all_Omega[i]/constant, Theta_all_Omega[i], c=exz_all_Omega[i], cmap='viridis',
                    alpha=0.75,vmin=min(np.nanmin(arr) for arr in exz_all_Omega.values()),vmax=max(np.nanmax(arr) for arr in exz_all_Omega.values()))
    ax.set_xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=12)
    ax.set_ylabel(r'$\theta_{im}$ [$\circ$]', fontsize=12)
    ax.set_ylim(0,85)
    ax.set_xlim(0,265)
    ax.set_title(f'$\Omega$={Omega[i]}%', fontsize=14)
# Hide the empty 6th subplot
fig.delaxes(axes[1, 2])
# Create a single colorbar for the whole figure
cbar_ax = fig.add_axes([0.7, 0.1, 0.02, 0.4])  # (右侧) [左, 底, 宽, 高]
cbar = fig.colorbar(sc, cax=cbar_ax)  
cbar.set_label(r'$e$ [-]', fontsize=12)
plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()

#filter theta=5, 15, 25 and compare the COR and NE with Uim
filterangle = [r'5$^\circ$', r'15$^\circ$', r'25$^\circ$']
COR_mean_filter,COR_std_filter,Uplot_filter = defaultdict(list),defaultdict(list),defaultdict(list)
NE_filter, UE_mean_filter, UE_std_filter, UNEplot_filter = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
for j in range(3):
    for i in range(5):
        thetamin,thetamax = 5*((j+1)*2-1)-2, 5*((j+1)*2-1)+2
        valid_indices = [j for j, val in enumerate(matched_Thetaim_Omega[i]) if  thetamin <= val <= thetamax]
        COR_mean_filter[5*j+i],COR_std_filter[5*j+i],Uplot_filter[5*j+i] = module.BinUimCOR_equalbinsize(
            [Vim_all_Omega[i][j] for j in valid_indices], [exz_all_Omega[i][j] for j in valid_indices], Vim_bin)
        NE_filter[5*j+i], UE_mean_filter[5*j+i], UE_std_filter[5*j+i], UNEplot_filter[5*j+i] = module.get_ejection_ratios_equalbinsize(
            [matched_Vim_Omega[i][j] for j in valid_indices], [matched_NE_Omega[i][j] for j in valid_indices],
            [matched_UE_Omega[i][j] for j in valid_indices], Vim_bin)

#according to angles
plt.figure(figsize=(13,5))
for j in range(3):
    plt.subplot(1,3,j+1)
    for i in range(5):
        plt.errorbar(Uplot_filter[5*j+i]/constant, COR_mean_filter[5*j+i], yerr=COR_std_filter[5*j+i], fmt='o',capsize=5, label=f'$\Omega$={Omega[i]}%',color=colors[i])
        plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=18)
        plt.ylabel(r'$e$ [-]', fontsize=18)
        plt.xlim(0,250)
        plt.ylim(-0.1,4)
        plt.title(r'$\theta_{im}$ = ' + filterangle[j], fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()

#scatter
fig, ax = plt.subplots()
sc = ax.scatter(np.concatenate([Uplot_filter[i] for i in range(10,15)]/constant), Omega_scatter, 
                c=np.concatenate([COR_mean_filter[i] for i in range(10,15)]), cmap='viridis',alpha=0.75)
cbar = plt.colorbar(sc)
cbar.set_label(r'$e$ [-]', fontsize=12)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=12)
plt.ylabel(r'$\Omega$ [$\%$]', fontsize=12)
plt.title(r'$\theta_{im}$ = ' + filterangle[2], fontsize=12)

plt.figure(figsize=(13,5))
for j in range(3):
    plt.subplot(1,3,j+1)
    for i in range(5):
        plt.plot(UNEplot_filter[5*j+i]/constant, NE_filter[5*j+i], 'o', label=f'$\Omega$={Omega[i]}%',color=colors[i])
    plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=18)
    plt.ylabel(r'$N_\mathrm{E}$ [-]', fontsize=18)
    plt.xlim(0,250)
    plt.ylim(-0.2,5.2)
    plt.title(r'$\theta_{im}$ = ' + filterangle[j], fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()

plt.figure(figsize=(13,5))
for j in range(3):
    plt.subplot(1,3,j+1)
    for i in range(5):
        plt.errorbar(UNEplot_filter[5*j+i]/constant, UE_mean_filter[5*j+i], yerr=UE_std_filter[5*j+i], fmt='o',capsize=5, label=f'$\Omega$={Omega[i]}%',color=colors[i])
    plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=18)
    plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=18)
    plt.xlim(0,275)
    plt.ylim(0,2)
    plt.title(r'$\theta_{im}$ = ' + filterangle[j], fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()

#according to omega
plt.figure(figsize=(13,7))
for i in range(5):
    plt.subplot(2,3,i+1)
    for j in range(3):
        plt.errorbar(Uplot_filter[5*j+i]/constant, COR_mean_filter[5*j+i], yerr=COR_std_filter[5*j+i], fmt='o',capsize=5, label=r'$\theta_{im}$ = ' + filterangle[j],color=colors_n[j])
        plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=18)
        plt.ylabel(r'$e$ [-]', fontsize=18)
        plt.xlim(0,275)
        plt.ylim(-0.1,4)
        plt.title(f'$\Omega$={Omega[i]}%', fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()

#filter Uim=50,100,150  and compare the COR and NE with Uim
filterUim = ['50', '100', '150']
COR_mean_filter,COR_std_filter,Thetaplot_filter = defaultdict(list),defaultdict(list),defaultdict(list)
NE_filter, UE_mean_filter, UE_std_filter, ThetaNEplot_filter = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
for j in range(3):
    for i in range(5):
        Uimmin,Uimmax = (50*(j+1)-5)*constant, (50*(j+1)+5)*constant
        valid_indices = [j for j, val in enumerate(matched_Vim_Omega[i]) if  Uimmin <= val <= Uimmax]
        COR_mean_filter[5*j+i],COR_std_filter[5*j+i],Thetaplot_filter[5*j+i] = module.BinThetaimCOR_equalbinsize(
            [Theta_all_Omega[i][j] for j in valid_indices], [exz_all_Omega[i][j] for j in valid_indices], Thetaim_bin)
        NE_filter[5*j+i], UE_mean_filter[5*j+i], UE_std_filter[5*j+i], ThetaNEplot_filter[5*j+i] = module.get_ejection_theta_equalbinsize(
            [matched_Thetaim_Omega[i][j] for j in valid_indices], [matched_NE_Omega[i][j] for j in valid_indices],
            [matched_UE_Omega[i][j] for j in valid_indices], Thetaim_bin)
        
#according to Uims
plt.figure(figsize=(13,5))
for j in range(3):
    plt.subplot(1,3,j+1)
    for i in range(5):
        plt.errorbar(Thetaplot_filter[5*j+i], COR_mean_filter[5*j+i], yerr=COR_std_filter[5*j+i], fmt='o',capsize=5, label=f'$\Omega$={Omega[i]}%',color=colors[i])
        plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=18)
        plt.ylabel(r'$e$ [-]', fontsize=18)
        plt.xlim(0,70)
        plt.ylim(0,2.25)
        plt.title(r'$U_{im}/\sqrt{gd}$ = ' + filterUim[j], fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()

plt.figure(figsize=(13,5))
for j in range(3):
    plt.subplot(1,3,j+1)
    for i in range(5):
        plt.plot(ThetaNEplot_filter[5*j+i], NE_filter[5*j+i], 'o',label=f'$\Omega$={Omega[i]}%',color=colors[i])
        plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=18)
        plt.ylabel(r'$N_\mathrm{E}$ [-]', fontsize=18)
        plt.xlim(0,45)
        plt.ylim(-0.1,2.1)
        plt.title(r'$U_{im}/\sqrt{gd}$ = ' + filterUim[j], fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()

plt.figure(figsize=(13,5))
for j in range(3):
    plt.subplot(1,3,j+1)
    for i in range(5):
        plt.errorbar(ThetaNEplot_filter[5*j+i], UE_mean_filter[5*j+i], yerr=UE_std_filter[5*j+i], fmt='o',capsize=5, label=f'$\Omega$={Omega[i]}%',color=colors[i])
        plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=18)
        plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=18)
        # plt.xlim(0,70)
        # plt.ylim(0,2.25)
        plt.title(r'$U_{im}/\sqrt{gd}$ = ' + filterUim[j], fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()

#according to omega
plt.figure(figsize=(13,7))
for i in range(5):
    plt.subplot(2,3,i+1)
    for j in range(3):
        plt.errorbar(ThetaNEplot_filter[5*j+i], COR_mean_filter[5*j+i], yerr=COR_std_filter[5*j+i], fmt='o',capsize=5, label=r'$U_{im}/\sqrt{gd}$ = ' + filterUim[j],color=colors_n[j])
        plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=18)
        plt.ylabel(r'$e$ [-]', fontsize=18)
        plt.xlim(0,70)
        plt.ylim(0,2.25)
        plt.title(f'$\Omega$={Omega[i]}%', fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()

