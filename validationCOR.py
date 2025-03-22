# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 11:20:23 2024

@author: WangX3
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
from collections import defaultdict
from module import store_particle_id_data
from module import store_sal_id_data
from module import BinUimCOR
from module import BinUimUd
from module import match_ejection_to_impact
from module import BinThetaimCOR
from module import get_ejection_ratios

# Load the data from the saved file
data_dict = {}  # Dictionary to store data
for i in range(5):
    #read in the files from \Theta = 0.02 - 0.06
    with open(f"input_pkl/data{i+2}_{0}.pkl", 'rb') as f:
        data_dict[f"data{i}"] = pickle.load(f)  # Store in dictionary

#basic parameters
dt = 0.01
D = 0.00025
coe_h = 13.5 #1.5d - critial height for ejection (Beladjine 2007; Ralaiarisoa 2022)
coe_sal_h = 17 #5d - critial height for saltation (Shao 2008; Kok 2012; transformation between different modes)
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
exz_vector_t = defaultdict(list)
IM_vector_t = defaultdict(list)
VIM_mean_t,ThetaIM_mean_t = defaultdict(list), defaultdict(list)
# VIM_t = defaultdict(list)
# Thetaim_t = defaultdict(list)
RIM = defaultdict(list)
Par = defaultdict(list)
VZ = defaultdict(list)

X,Z = defaultdict(list),defaultdict(list)
for i in range(5):
    filename = f"data{i}"
    data = data_dict[filename]
    num_p = 2725
    ParticleID=np.linspace(num_p-10-300,num_p-10-1,300)
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

# colors_map = plt.cm.viridis(np.linspace(0, 1, 10))
# plt.figure(figsize=(10, 15))
# for i in range(300):
#     #plt.plot(X[4][:,i]/D, t_ver, '.') 
#     color_index = i % len(colors_map)
#     if Par[4][i][2].any():
#         plt.plot(X[4][Par[4][i][2][:,0],i]/D, t_ver[Par[4][i][2][:,0]], 'o', color=colors_map[color_index], fillstyle='none')
#         plt.plot(X[4][Par[4][i][2][:,1],i]/D, t_ver[Par[4][i][2][:,1]], 'x', color=colors_map[color_index], fillstyle='none')
#         #plt.plot(X[4][Par[4][i][2][:,0:1],i]/D, t_ver[Par[4][i][2][:,0:1]], '-')
# for i in range(300):
#     color_index = i % len(colors_map)
#     if EDindices[4][i][0].any():
#         plt.plot(X[4][EDindices[4][i][0],i]/D, t_ver[EDindices[4][i][0]], 'd', color=colors_map[color_index])
# plt.xlabel('x/D [m]')
# plt.ylabel('t [s]')
# plt.legend(fontsize=14)

id_p = 296#287
plt.figure(figsize=(10,7))
plt.plot(t_ver, Z[4][:,id_p]/D, linestyle='-', marker='.', color='k', label='vertical position', markersize=5, linewidth=2)
#axs[i].plot(Par[4][id_p][0], np.zeros(len(Par[4][id_p][0])), 'x', label='Collision moments')
plt.plot(t_ver[Par[4][id_p][2][0:2,0]], Z[4][Par[4][id_p][2][0:2,0],id_p]/D, 'Dr', label='Impact', markerfacecolor='none', markersize=10)
plt.plot(t_ver[Par[4][id_p][2][0:2,1]], Z[4][Par[4][id_p][2][0:2,1],id_p]/D, 'sr', label='Rebound', markerfacecolor='none', markersize=10)
#plt.plot(t_ver[Par[4][id_p][1][:,0]], Z[4][Par[4][id_p][1][:,0],id_p]/D, '*', label='Mobile')
#plt.plot(t_ver[Par[4][id_p][1][:,1]], Z[4][Par[4][id_p][1][:,1],id_p]/D, '*', label='Mobile')
plt.plot(t_ver[EDindices[4][id_p][0]], Z[4][EDindices[4][id_p][0],id_p]/D, 'ob', label='Ejection', markerfacecolor='none', markersize=10)
plt.plot(t_ver[EDindices[4][id_p][1]], Z[4][EDindices[4][id_p][1],id_p]/D, 'vb', label='Deposition', markerfacecolor='none', markersize=10)
plt.plot(t_ver[Par[4][id_p][2][4:,0]], Z[4][Par[4][id_p][2][4:,0],id_p]/D, 'Dr', markerfacecolor='none', markersize=10)
plt.plot(t_ver[Par[4][id_p][2][4:,1]], Z[4][Par[4][id_p][2][4:,1],id_p]/D, 'sr', markerfacecolor='none', markersize=10)
plt.xlabel(r'$t$ [s]', fontsize=24)
plt.ylabel(r'$Z_p/d$ [-]',fontsize=24)
plt.legend(fontsize=24)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(t_ver, VZ[4][id_p], linestyle='-', marker='.', label='Particle velocity')
plt.plot(t_ver[Par[4][id_p][2][:, 0]], VZ[4][id_p][Par[4][id_p][2][:, 0]], 'D', label='Impact')
plt.plot(t_ver[Par[4][id_p][2][:, 1]], VZ[4][id_p][Par[4][id_p][2][:, 1]], 'o', label='Rebound')
plt.xlabel('t [s]')
plt.ylabel('Uz [m/s]')
plt.legend()
plt.show()


#get the quantities from the steady state/5s
exz_all,Vim_all,VD_all,Theta_all,zE_all,impact_list,ejection_list = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
for i in range (5):
    N_range = int((3 / 5) * N_inter)
    exz_all[i] = [value for sublist in exz_vector_t[i][N_range:] for value in sublist]
    Vim_all[i] = [value[0] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    VD_all[i] = [value for sublist in VD_vector_t[i][N_range:] for value in sublist]
    Theta_all[i] = [value[7] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    IDim = [value[1] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    IDre = [value[2] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    xim = [value[3] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    xre = [value[4] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    xcol = [value[5] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    Pim = [value[6] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    impact_list[i] = [IDim, IDre, xim, xre, Vim_all[i], xcol, Pim, Theta_all[i]]
    vE = [value[0] for sublist in E_vector_t[i][N_range:] for value in sublist]
    IDE = [value[1] for sublist in E_vector_t[i][N_range:] for value in sublist]
    xE = [value[2] for sublist in E_vector_t[i][N_range:] for value in sublist]
    PE = [value[3] for sublist in E_vector_t[i][N_range:] for value in sublist]
    zE_all[i] = [value[4] for sublist in E_vector_t[i][N_range:] for value in sublist]
    ejection_list[i] = [IDE, xE, vE, PE]
    print('Ne/Nim',len(IDE)/len(IDim))

Vim_all_values = [value for sublist in Vim_all.values() for value in sublist]
# Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
# Vimde_bin = np.linspace(min(Vimde_all_values), max(Vimde_all_values), 8)   
Thetaim_all_values =  [value for sublist in Theta_all.values() for value in sublist]
# Thetaim_bin = np.linspace(min(Thetaim_all_values), max(Thetaim_all_values), 8)
zE_all_values = [value for sublist in zE_all.values() for value in sublist]

impact_ejection_list = defaultdict(list)
for i in range (5):
    impact_ejection_list[i]=match_ejection_to_impact(impact_list[i], ejection_list[i], dt)

#global means and stds at all the wind conditions   
exz_all_values = [value for sublist in exz_all.values() for value in sublist]
VD_all_values = [value for sublist in VD_all.values() for value in sublist]
matched_Vim_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][0]]
matched_thetaim_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][1]]
matched_NE_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][2]]
matched_UE_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][3]]
#get the global NE, UE from all impacts and matched ejections 
NEmean_glo, UEmean_glo, UEstd_glo, Uplot_NE = get_ejection_ratios(matched_Vim_all, matched_NE_all, matched_UE_all, 8)
CORmean_glo,CORstd_glo,Uimplot=BinUimCOR(Vim_all_values,exz_all_values,8)
Pr_glo,Uplot=BinUimUd(Vim_all_values,VD_all_values,8)
COR_theta_mean_glo, COR_theta_std_glo, Thetaimplot = BinThetaimCOR(Thetaim_all_values, exz_all_values, 8)

constant = np.sqrt(9.81*D)
Theta_mean_all = np.nanmean(Thetaim_all_values) #mean theta_im in all cases
#empirical data
Hcr = 1.5
NE_prin = (-0.001*Hcr + 0.012)*Uplot_NE/constant
UIM_UE_prin = np.linspace(0,120, 20)
VE_prin = (0.0538*Hcr + 1.0966)*np.sqrt(UIM_UE_prin)
COR_emp = 0.7469*np.exp(0.1374*Hcr)*(Uimplot/constant)**(-0.0741*np.exp(0.214*Hcr))#Jiang et al. (2024) JGR
Pr_emp = 0.9945*Hcr**(-0.0166)*(1-np.exp(-0.1992*Hcr**(-0.8686)*Uplot/constant))
#Chen 2019
COR_Chen = 0.62 + 0.0084*Uimplot - 0.63*np.sin(Theta_mean_all/180*np.pi)
NE_Chen = np.exp(-0.2 + 1.35*np.log(Uplot_NE)-0.01*Theta_mean_all/180*np.pi)
VE_Chen = np.exp(-1.48 + 0.082*Uplot_NE-0.003*Theta_mean_all/180*np.pi)
#exp data Jiang 2024; Hcr = 1.5D
Unsexp = [15, 35, 55, 75, 95]
Nsexp = [0.05, 0.25, 0.65, 0.625, 1.1]
CORexp_mean = [0.7, 0.7, 0.6, 0.65, 0.5]
CORexp_std = [0.3, 0.2, 0.2, 0.2, 0.1]
Prexp = [0.85, 0.94, 0.98, 0.99, 1.0]
Uvsexp = [25, 45, 70, 95, 125]
Usexp = [5, 7, 8, 12.5, 6]


plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14  # 也可以同时设置 y 轴刻度字体大小
plt.figure(figsize=(17,11))
plt.subplot(2,2,1)
#plot COR - Uim
# for i in range(5):
#     plt.errorbar(Uimplot/constant, CORmean[i], yerr=CORstd[i], fmt='o', capsize=5, label=fr'$\tilde{{\Theta}}=0.0{i+2}$',color=colors[i])
line1 = plt.errorbar(Uimplot/constant, CORmean_glo, yerr=CORstd_glo, fmt='o', capsize=5, label='This study', color='#3776ab')
# line2 = plt.plot(Uimplot/constant, COR_Chen, label='No wind (Chen et al., 2019)',color='k')
line3 = plt.errorbar(Unsexp, CORexp_mean, yerr=CORexp_std, fmt='x', capsize=5, label='Experiment (Jiang et al., 2024)', color='k')
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=24)
plt.ylabel(r'$\bar{e}$ [-]', fontsize=24)
# plt.legend([line1[0],line2[0],line3],['This study','No wind (Chen et al., 2019)','With wind (Jiang et al., 2024)'], fontsize=20)
plt.legend(fontsize=20)
plt.subplot(2,2,2)
#plot Pr - Uim 
# for i in range(5):
#     plt.plot(Uplot/constant, Pr[i], 'o', label=fr'$\tilde{{\Theta}}=0.0{i+2}$',color=colors[i])
plt.plot(Uplot/constant, Pr_glo, 'o', label='This study', color='#3776ab')
#plt.plot(Uplot/constant, Pr_emp, label='empirical Jiang et al. (2024)',color='k')
plt.plot(Unsexp, Prexp, 'x', label='experiment Jiang et al. (2024)', color='k')
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=24)
plt.ylabel(r'$P_\mathrm{r}$ [-]', fontsize=24)
plt.subplot(2,2,3)
#plot \bar{NE} - Uim
plt.plot(Uplot_NE/constant, NEmean_glo, 'o', label='This study', color='#3776ab')
# plt.plot(Uplot_NE/constant, NE_Chen, label='No wind (Chen et al., 2019)', color='k')
plt.plot(Unsexp, Nsexp, 'x', label='With wind (Jiang et al., 2024)',color='k')
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=24)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=24)
#plot \bar{UE} - Uim
plt.subplot(2,2,4)
# for i in range(5):
#     plt.errorbar(Uplot_NE/constant, VE_mean[i]/constant, yerr=VE_std[i]/constant, fmt='o', capsize=5, label=fr'$\tilde{{\Theta}}=0.0{i+2}$',color=colors[i])
plt.errorbar(Uplot_NE/constant, UEmean_glo/constant, yerr=UEstd_glo/constant, fmt='o', capsize=5, label='This study', color='#3776ab')
# plt.plot(Uplot_NE/constant, VE_Chen/constant, label='No wind (Chen et al., 2019)', color='k')
plt.plot(Uvsexp, Usexp, 'x', label='With wind (Jiang et al., 2024)',color='k')
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=24)
plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=24)
plt.tight_layout()
plt.show()


#plot COR - Thetaim
# plt.figure(figsize=(10, 10))
# # for i in range(1,5):
# #     plt.errorbar(Thetaimplot, COR_theta_mean[i], yerr=COR_theta_std[i], fmt='o', capsize=5, label=fr'$\tilde{{\Theta}}=0.0{i+2}$',color=colors[i])
# plt.errorbar(Thetaimplot, COR_theta_mean_glo, yerr=COR_theta_std_glo, fmt='o', capsize=5)
# plt.xlabel(r'$\theta_{IM}$ [deg]', fontsize=14)
# plt.ylabel(r'$\bar{e}_\mathrm{xz,steady}$ [-]', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()

#distribution of impact angles in all cases
plt.figure()
# Calculate histogram (raw counts)
data = Thetaim_all_values
counts, bin_edges = np.histogram(data, bins=15)
# Normalize the counts so their sum is 1
normalized_counts = counts / np.sum(counts)
# Create the step plot
plt.step(bin_edges[:-1], normalized_counts, where='mid')
plt.xlabel(r'$\theta_{IM}$ [deg]', fontsize=14)
plt.ylabel('Normalized Frequency [-]', fontsize=14)

#distribution of ejection height
plt.figure()
# Calculate histogram (raw counts)
data = [x / D for x in zE_all_values]
counts, bin_edges = np.histogram(data, bins=30)
# Normalize the counts so their sum is 1
normalized_counts = counts / np.sum(counts)
# Create the step plot
plt.step(bin_edges[:-1], normalized_counts, where='mid')
plt.xlabel(r'$z_{E}/D$ [-]', fontsize=14)
plt.ylabel('Normalized Frequency [-]', fontsize=14)

plt.figure()
# Calculate histogram (raw counts)
for i in range(5):
    data = [x for x in ejection_list[i][2]]
    counts, bin_edges = np.histogram(data, bins=30)
    # Normalize the counts so their sum is 1
    normalized_counts = counts / np.sum(counts)
    # Create the step plot
    plt.step(bin_edges[:-1], normalized_counts, where='mid',  label=fr'$\tilde{{\Theta}}=0.0{i+2}$',color=colors[i])
plt.xlabel(r'$U_{E}$ [m/s]', fontsize=14)
plt.ylabel('Normalized Frequency [-]', fontsize=14)
plt.legend()