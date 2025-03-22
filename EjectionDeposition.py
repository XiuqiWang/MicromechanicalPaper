# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:51:19 2024

@author: WangX3
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from module import store_particle_id_data
from module import store_sal_id_data
from module import BinUimCOR
from module import BinUimUd
from module import match_ejection_to_impact
from module import AverageOverShields
from module import get_ejection_ratios
from module import get_ejection_theta
import pandas as pd

# Load the data from the saved file
data_dict = {}  # Dictionary to store data
# for i in range(5):  # Loop from 0 to 4
#     filename = f"input_pkl/data_{i}.pkl"  # Generate filename
#     with open(filename, 'rb') as f:
#         data_dict[f"data{i}"] = pickle.load(f)  # Store in dictionary
for i in range(2,7): #Theta = 0.02-0.06
    for j in range(5):
        filename = f"input_pkl/data{i}_{j}.pkl"  # Generate filename
        with open(filename, 'rb') as f:
            data_dict[f"data{5*(i-2)+(j+1)}"] = pickle.load(f)  # Store in dictionary  
for i in range(2,7):  # Loop from 0 to 4
    for j in range(5):
        filename = f"input_pkl/dataLBM{i}_{j}.pkl"  # Generate filename
        with open(filename, 'rb') as f:
            data_dict[f"data{25+5*(i-2)+(j+1)}"] = pickle.load(f)  # Store in dictionary     

#basic parameters
dt = 0.01
D = 0.00025
coe_h = 13.5 #1.5d - critial height for ejection (Beladjine 2007; Ralaiarisoa 2022)
coe_sal_h = 17 #5d - critial height for saltation (Shao 2008; Kok 2012; transformation between different modes)
N_inter = 50 #number of output timesteps for erosion and deposition properties
t_inter = np.linspace(0,5,N_inter+1)
Omega = [0, 1, 5, 10, 20]
colors = plt.cm.viridis(np.linspace(0, 1, 5))  # 5 colors
colors_n = plt.cm.plasma(np.linspace(0, 1, 5))

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
for i in range(50):
    filename = f"data{i+1}"
    data = data_dict[filename]
    if i in [11,17,18]:#0.04 1%, 0.05 5,10%
        num_p = 2714
    elif i < 25:
        num_p = 2725
    else:
        num_p = 2754
    ParticleID=np.linspace(num_p-10-300,num_p-10-1,300)
    ParticleID_int = ParticleID.astype(int)
    #cal erosion and deposition properties for each Omega
    #EDindices, E, VX, VExVector, VEzVector, VEx, VEz, ME, MD
    EDindices[i], ME[i], MD[i], VExz_mean_t[i], VDxz_mean_t[i], VD_vector_t[i], E_vector_t[i] = store_particle_id_data(data,ParticleID_int,coe_h,dt,N_inter,D)
    #cal rebound properties for each Omega
    ParticleID_sal=np.linspace(num_p-310,num_p-1,310)
    ParticleID_salint = ParticleID_sal.astype(int)
    #ez_t[i], exz_t[i], VIM_t[i], Thetaim_t[i], 
    Par[i], VZ[i], exz_mean_t[i], ez_mean_t[i], VIM_mean_t[i], ThetaIM_mean_t[i], RIM[i], exz_vector_t[i], IM_vector_t[i] = store_sal_id_data(data,ParticleID_salint, coe_sal_h, dt, N_inter, D)

# #verification
# id_p = 309
# t=np.linspace(dt, 5, 500);
# plt.figure()
# plt.plot(t, VZ[4][id_p][:], '-', marker='.', label='Vertical velocity')
# par_events = Par[4][id_p][0]
# plt.plot(par_events, np.zeros(len(par_events)), 'x')
# IDI = Par[4][id_p][1][:,0]
# plt.plot(t[IDI], VZ[4][id_p][IDI], 'x', label='impact')
# IDR = Par[4][id_p][1][:,1]
# plt.plot(t[IDR], VZ[4][id_p][IDR], 'x', label='rebound')
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{p,z}$ [m/s]', fontsize=14)
# plt.legend(fontsize=10)

# id_p = 297
# data4 = data_dict['data4'][:]
# Z4 = np.array([time_step['Position'][num_p-13][2] for time_step in data4])
# plt.figure()
# plt.plot(t, Z4, '-', marker='.', label='Vertical position')
# IDE = EDindices[4][id_p][0]
# plt.plot(t[IDE], Z4[IDE], 'x', label='erosion')
# IDD = EDindices[4][id_p][1]
# plt.plot(t[IDD], Z4[IDD], 'o', label='deposition')
# IDI = Par[4][id_p][2][:,0]
# plt.plot(t[IDI], Z4[IDI], 'd', label='impact')
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$Z_\mathrm{p}$ [m/s]', fontsize=14)
# plt.legend(fontsize=10)


#compare the differences of COR between LM and LB 
# plt.figure(figsize=(12, 10))
# plt.subplot(2, 1, 1)
# for i in range(5):
#     plt.plot(
#         t_inter.tolist(),
#         [np.nan] + exz_mean_t[i].tolist(),
#         color=colors[i],
#         label = f"$\\Omega$={Omega[i]}%"
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{e}_\mathrm{xz,LMmodel}$ [-]', fontsize=14)
# plt.ylim(0,2.25)
# plt.legend(fontsize=14)
# plt.subplot(2, 1, 2)
# for i in range(5,10):
#     plt.plot(
#         t_inter.tolist(),
#         [np.nan] + exz_mean_t[i].tolist(),
#         color=colors[i-5],
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{e}_\mathrm{xz, LBmodel}$ [-]', fontsize=14)
# plt.ylim(0,2.25)
# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(12, 10))
# plt.subplot(2, 1, 1)
# for i in range(5):
#     plt.plot(
#         t_inter.tolist(),
#         [np.nan] + ThetaIM_mean_t[i].tolist(),
#         color=colors[i],
#         label = f"$\\Omega$={Omega[i]}%"
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{\theta}_\mathrm{im,LMmodel}$ [$\circ$]', fontsize=14)
# plt.legend(fontsize=14)
# plt.subplot(2, 1, 2)
# for i in range(5,10):
#     plt.plot(
#         t_inter.tolist(),
#         [np.nan] + ThetaIM_mean_t[i].tolist(),
#         color=colors[i-5],
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{\theta}_\mathrm{im, LBmodel}$ [$\circ$]', fontsize=14)
# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()


#distribution of CORs
exz_whole, exz_steady = defaultdict(list),defaultdict(list)
exz_whole_m, exz_whole_s, exz_steady_m, exz_steady_s = [],[], [], []
exz_whole_median, exz_steady_median = [], []
theta_whole, theta_steady = defaultdict(list),defaultdict(list)
theta_whole_m, theta_whole_s, theta_steady_m, theta_steady_s = [], [], [], []
for i in range (10):
    exz_whole[i] = [value for sublist in exz_vector_t[i][0:] for value in sublist]
    exz_whole_m.append(np.nanmean(exz_whole[i]))
    exz_whole_s.append(np.nanstd(exz_whole[i]))
    exz_whole_median.append(np.median(exz_whole[i]))
    exz_steady[i] = [value for sublist in exz_vector_t[i][int(3.5/5* N_inter):] for value in sublist]
    exz_steady_m.append(np.nanmean(exz_steady[i]))
    exz_steady_s.append(np.nanstd(exz_steady[i]))
    exz_steady_median.append(np.median(exz_steady[i]))
    
    theta_whole[i] = [value[7] for sublist in IM_vector_t[i][0:] for value in sublist]
    theta_whole_m.append(np.nanmean(theta_whole[i]))
    theta_whole_s.append(np.nanstd(theta_whole[i]))
    theta_steady[i] = [value[7] for sublist in IM_vector_t[i][int(3.5/5* N_inter):] for value in sublist]
    theta_steady_m.append(np.nanmean(theta_steady[i]))
    theta_steady_s.append(np.nanstd(theta_steady[i]))
    
plt.figure(figsize=(10, 15))
plt.subplot(2, 1, 1)
for i in range(5):
    # Calculate histogram (raw counts)
    counts, bin_edges = np.histogram(exz_whole[i], bins=50)
    # Normalize the counts so their sum is 1
    normalized_counts = counts / np.sum(counts)
    # Create the step plot
    plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
    # Prepare data for staircase plot
    #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
    plt.xlim(-0.5,7.5)
    plt.ylim(0,0.3)
plt.xlabel('$e_\mathrm{xz,whole}$ [-]', fontsize=14)
plt.ylabel('Normalized Frequency [-]', fontsize=14)
plt.title('Liquid migration')
plt.legend(fontsize=14)
plt.subplot(2, 1, 2)
for i in range(5):
    # Calculate histogram (raw counts)
    counts, bin_edges = np.histogram(exz_steady[i], bins=50)
    # Normalize the counts so their sum is 1
    normalized_counts = counts / np.sum(counts)
    # Create the step plot
    plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
    # Prepare data for staircase plot
    #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
    plt.xlim(-0.5,7.5)
    plt.ylim(0,0.16)
plt.xlabel('$e_\mathrm{xz,steady}$ [-]', fontsize=14)
plt.ylabel('Normalized Frequency [-]', fontsize=14)
plt.show()

plt.figure(figsize=(10, 15))
plt.subplot(2, 1, 1)
for i in range(5,10):
    # Calculate histogram (raw counts)
    counts, bin_edges = np.histogram(exz_whole[i], bins=50)
    # Normalize the counts so their sum is 1
    normalized_counts = counts / np.sum(counts)
    # Create the step plot
    plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i-5], label = f"$\\Omega$={Omega[i-5]}%")
    # Prepare data for staircase plot
    #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
    plt.xlim(-0.5,7.5)
    plt.ylim(0,0.3)
plt.xlabel('$e_\mathrm{xz,whole}$ [-]', fontsize=14)
plt.ylabel('Normalized Frequency [-]', fontsize=14)
plt.title('Liquid bridge')
plt.legend(fontsize=14)
plt.subplot(2, 1, 2)
for i in range(5,10):
    # Calculate histogram (raw counts)
    counts, bin_edges = np.histogram(exz_steady[i], bins=50)
    # Normalize the counts so their sum is 1
    normalized_counts = counts / np.sum(counts)
    # Create the step plot
    plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i-5], label = f"$\\Omega$={Omega[i-5]}%")
    # Prepare data for staircase plot
    #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
    plt.xlim(-0.5,7.5)
    plt.ylim(0,0.16)
plt.xlabel('$e_\mathrm{xz,steady}$ [-]', fontsize=14)
plt.ylabel('Normalized Frequency [-]', fontsize=14)
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.errorbar(Omega, exz_whole_m[0:5], yerr=exz_whole_s[0:5], fmt='o', capsize=10, label='whole simulation time')
plt.errorbar(Omega, exz_steady_m[0:5], yerr=exz_steady_s[0:5], fmt='o', capsize=10, label='steady state')
plt.xlabel(r'$\Omega$ [%]')
plt.ylabel(r'$e_\mathrm{xz}$ [-]')
plt.title('Liquid migration')
plt.legend()
plt.subplot(1,2,2)
plt.errorbar(Omega, exz_whole_m[5:10], yerr=exz_whole_s[5:10], fmt='o', capsize=10, label='whole simulation time')
plt.errorbar(Omega, exz_steady_m[5:10], yerr=exz_steady_s[5:10], fmt='o', capsize=10, label='steady state')
plt.xlabel(r'$\Omega$ [%]')
plt.ylabel(r'$e_\mathrm{xz}$ [-]')
plt.title('Liquid bridge')
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.plot(Omega, exz_whole_median[0:5],'o', label='whole simulation time')
plt.plot(Omega, exz_steady_median[0:5], 'o', label='steady state')
plt.xlabel(r'$\Omega$ [%]')
plt.ylabel(r'$e_\mathrm{xz}$ [-]')
plt.title('Liquid migration')
plt.legend()
plt.subplot(1,2,2)
plt.plot(Omega, exz_whole_median[5:10],'o', label='whole simulation time')
plt.plot(Omega, exz_steady_median[5:10],'o', label='steady state')
plt.xlabel(r'$\Omega$ [%]')
plt.ylabel(r'$e_\mathrm{xz}$ [-]')
plt.title('Liquid bridge')
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.errorbar(Omega, theta_whole_m[0:5], yerr=theta_whole_s[0:5], fmt='o', capsize=10, label='whole simulation time')
plt.errorbar(Omega, theta_steady_m[0:5], yerr=theta_steady_s[0:5], fmt='o', capsize=10, label='steady state')
plt.xlabel(r'$\Omega$ [%]')
plt.ylabel(r'$\theta_\mathrm{im}$ [$\circ$]')
plt.title('Liquid migration')
plt.legend()
plt.subplot(1,2,2)
plt.errorbar(Omega, theta_whole_m[5:10], yerr=theta_whole_s[5:10], fmt='o', capsize=10, label='whole simulation time')
plt.errorbar(Omega, theta_steady_m[5:10], yerr=theta_steady_s[5:10], fmt='o', capsize=10, label='steady state')
plt.xlabel(r'$\Omega$ [%]')
plt.ylabel(r'$\theta_\mathrm{im}$ [$\circ$]')
plt.title('Liquid bridge')
plt.show()


exz_all,Vim_all,VD_all,Theta_all,impact_list,ejection_list = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
N_range = np.full(50, 0).astype(int)
# T_range = np.array([1.8, 1.8, 2, 2.5, 3.5, 
#                     1.8, 2, 2, 2, 2])
# N_range = ((T_range / 5) * N_inter).astype(int)
# N_range = np.full(50, 3.5/5*N_inter).astype(int)
for i in range (50):
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

matched_Vim, matched_thetaim, matched_NE, matched_UE = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
CORmean,CORstd,Uimplot = defaultdict(list),defaultdict(list),defaultdict(list)
COR_theta_mean, COR_theta_std, Thetaimplot = defaultdict(list),defaultdict(list),defaultdict(list)
Pr,Uplot = defaultdict(list),defaultdict(list)
NE_mean,UE_mean,UE_std, Uplot_NE=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
# NE_theta_mean, UE_theta_mean, UE_theta_std, Thetaplot_NE=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
impact_ejection_list = defaultdict(list)
for i in range (50):
    impact_ejection_list=match_ejection_to_impact(impact_list[i], ejection_list[i], dt)
    matched_Vim[i] = [element for element in impact_ejection_list[0]]
    matched_thetaim[i] = [element for element in impact_ejection_list[1]]
    matched_NE[i] = [element for element in impact_ejection_list[2]]
    matched_UE[i] = [element for element in impact_ejection_list[3]]
    
for i in range (50):
    CORmean[i],CORstd[i],Uimplot[i] = BinUimCOR(Vim_all[i],exz_all[i],8)
    Pr[i],Uplot[i] = BinUimUd(Vim_all[i],VD_all[i],8)    
    NE_mean[i], UE_mean[i], UE_std[i], Uplot_NE[i] = get_ejection_ratios(matched_Vim[i], matched_NE[i], matched_UE[i], 8)
    # NE_theta_mean[i], UE_theta_mean[i], UE_theta_std[i], Thetaplot_NE[i] = get_ejection_theta(matched_thetaim[i], matched_NE[i], matched_UE[i], 8)
    
constant = np.sqrt(9.81*D)  
# #plot NE - Uim 
# plt.figure(figsize=(15,8))
# plt.subplot(1,2,1)
# for i in range(5):
#     plt.plot(Uplot_NE[i]/constant, NE_mean[i], 'o', label=f"$\\Omega$={Omega[i]}%",color=colors[i])
#     plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
#     plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
#     plt.xlim(0,140)
#     plt.ylim(-0.1,1.4)
# plt.title('Liquid migration')
# plt.legend(fontsize=14)
# plt.subplot(1,2,2)
# for i in range(5,10):
#     plt.plot(Uplot_NE[i]/constant, NE_mean[i], 'o', label=f"$\\Omega$={Omega[i-5]}%",color=colors[i-5])
#     plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
#     plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
#     plt.xlim(0,140)
#     plt.ylim(-0.1,1.4)
# plt.title('Liquid bridge')
# plt.show()


# #plot UE - Uim
# plt.figure(figsize=(15,8))
# plt.subplot(1,2,1)
# for i in range(5):
#     plt.errorbar(Uplot_NE[i]/constant, UE_mean[i]/constant, yerr=UE_std[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
# plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
# plt.xlim(0,225)
# plt.ylim(0,30)
# plt.legend(fontsize=14)
# plt.title('Liquid migration')
# plt.subplot(1,2,2)
# for i in range(5,10):
#     plt.errorbar(Uplot_NE[i]/constant, UE_mean[i]/constant, yerr=UE_std[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i-5]}%",color=colors[i-5])
# plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
# plt.xlim(0,225)
# plt.ylim(0,30)
# plt.title('Liquid bridge')
# plt.show()

# #plot NE - Thetaim 
# plt.figure(figsize=(15,8))
# plt.subplot(1,2,1)
# for i in range(5):
#     plt.plot(Thetaplot_NE[i], NE_theta_mean[i], 'o', label=f"$\\Omega$={Omega[i]}%",color=colors[i])
#     plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
#     plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
#     # plt.xlim(0,140)
#     # plt.ylim(-0.1,1.4)
# plt.title('Liquid migration')
# plt.legend(fontsize=14)
# plt.subplot(1,2,2)
# for i in range(5,10):
#     plt.plot(Thetaplot_NE[i], NE_theta_mean[i], 'o', label=f"$\\Omega$={Omega[i-5]}%",color=colors[i-5])
#     plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
#     plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
#     # plt.xlim(0,140)
#     # plt.ylim(-0.1,1.4)
# plt.title('Liquid bridge')
# plt.show()

# #NE vs Vim and thetaim
# plt.figure(figsize=(8, 6))
# plt.subplot(1,2,1)
# for i in range(1):
#     mask = np.array(matched_NE[i]) != 0  # Boolean mask to exclude zero values
#     plt.scatter(np.array(matched_thetaim[i])[mask], np.array(matched_Vim[i])[mask]/constant, c=np.array(matched_NE[i])[mask], cmap='viridis',vmin=0,vmax=8)
# plt.colorbar(label=r'$\bar{N}_\mathrm{E}$ [-]')  # Add color bar
# plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
# plt.ylabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.xlim(0,55)
# plt.ylim(0,175)
# plt.title("Liquid migration")
# plt.subplot(1,2,2)
# for i in range(5,6):
#      mask = np.array(matched_NE[i]) != 0  # Boolean mask to exclude zero values
#      plt.scatter(np.array(matched_thetaim[i])[mask], np.array(matched_Vim[i])[mask]/constant, c=np.array(matched_NE[i])[mask], cmap='viridis',vmin=0,vmax=8)
# plt.colorbar(label=r'$\bar{N}_\mathrm{E}$ [-]')  # Add color bar
# plt.xlabel(r'$\theta_{im}$ [$\circ$]', fontsize=14)
# plt.ylabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.xlim(0,55)
# plt.ylim(0,175)
# plt.title("Liquid bridge")
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
for i in range (5): #loop over Omega 0-20 %
    selected_indices = list(range(i+25, 50, 5))  # Get indices like [25, 30, 35, 40, 45], [26, 31, 36, 41, 46], etc.
    Vim_all_Omega[i+5] = np.concatenate([Vim_all[j] for j in selected_indices]).tolist()
    exz_all_Omega[i+5] = np.concatenate([exz_all[j] for j in selected_indices]).tolist()
    Theta_all_Omega[i+5] = np.concatenate([Theta_all[j] for j in selected_indices]).tolist()
    VD_all_Omega[i+5] = np.concatenate([VD_all[j] for j in selected_indices]).tolist()
    matched_Vim_Omega[i+5] = np.concatenate([matched_Vim[j] for j in selected_indices]).tolist()
    matched_Thetaim_Omega[i+5] = np.concatenate([matched_thetaim[j] for j in selected_indices]).tolist()
    matched_NE_Omega[i+5] = np.concatenate([matched_NE[j] for j in selected_indices]).tolist()
    matched_UE_Omega[i+5] = [sub for lst in [matched_UE[j] for j in selected_indices] if lst for sub in lst]   

CORmean_Omega,CORstd_Omega,Uimplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list)
COR_theta_mean_Omega, COR_theta_std_Omega, Thetaimplot_Omega = defaultdict(list), defaultdict(list), defaultdict(list)
Pr_Omega,Uplot_Omega = defaultdict(list), defaultdict(list)
NE_mean_Omega, UE_mean_Omega, UE_std_Omega, Uplot_NE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
NE_theta_mean_Omega, UE_theta_mean_Omega, UE_theta_std_Omega, Thetaplot_NE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
for i in range (10): #loop over Omega 0-20 % twice, one for LM one for LB
    CORmean_Omega[i],CORstd_Omega[i],Uimplot_Omega[i] = BinUimCOR(Vim_all_Omega[i],exz_all_Omega[i],8)
    Pr_Omega[i],Uplot_Omega[i] = BinUimUd(Vim_all_Omega[i],VD_all_Omega[i],8)    
    NE_mean_Omega[i], UE_mean_Omega[i], UE_std_Omega[i], Uplot_NE_Omega[i]=get_ejection_ratios(matched_Vim_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], 8)
    # NE_theta_mean_Omega[i], UE_theta_mean_Omega[i], UE_theta_std_Omega[i], Thetaplot_NE_Omega[i]=get_ejection_theta(matched_Thetaim_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], 8)


#plot NE-Uim for LM and LB
plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
for i in range(5):
    plt.plot(Uplot_NE_Omega[i]/constant, NE_mean_Omega[i], 'o', label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlim(0,200)
plt.ylim(0, 1.2)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
plt.legend(fontsize=14)
plt.subplot(1,2,2)
for i in range(5,10):
    plt.plot(Uplot_NE_Omega[i]/constant, NE_mean_Omega[i], 'o', label=f"$\\Omega$={Omega[i-5]}%",color=colors[i-5])
plt.xlim(0,200)
plt.ylim(0, 1.2)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)

plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
for i in range(5):
    plt.errorbar(Uplot_NE_Omega[i]/constant, UE_mean_Omega[i]/constant, yerr=UE_std_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlim(0,200)
plt.ylim(0, 20)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
plt.legend(fontsize=14)
plt.subplot(1,2,2)
for i in range(5,10):
    plt.errorbar(Uplot_NE_Omega[i]/constant, UE_mean_Omega[i]/constant, yerr=UE_std_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i-5]}%",color=colors[i-5])
plt.xlim(0,200)
plt.ylim(0, 20)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)


def match_Uim_thetaim(matched_Vim, matched_thetaim, num_bins):
    if not matched_Vim:
        mean_thetaim, std_thetaim, Uthetaplot =[],[],[]
    else:
        # Combine data into a DataFrame and sort by impact velocity
        data = pd.DataFrame({'Vim': matched_Vim, 'thetaim': matched_thetaim})
        data = data.sort_values(by='Vim').reset_index(drop=True)
    
        # Get bin edges using quantiles
        quantiles = np.linspace(0, 1, num_bins + 1)
        bin_edges = np.quantile(data['Vim'], quantiles)
    
        data['bin'] = pd.cut(data['Vim'], bins=bin_edges, include_lowest=True)
    
        #print(data.columns)  # 检查DataFrame的列名
        # for name, group in data.groupby('bin'):
        #     print(name, group['UE'].values)
            
        mean_thetaim = data.groupby('bin')['thetaim'].mean()
        std_thetaim = data.groupby('bin')['thetaim'].std()
        data['bin'] = data['bin'].astype(str)
    
        Uthetaplot = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return mean_thetaim, std_thetaim, Uthetaplot

mean_thetaim, std_thetaim, Uthetaplot = match_Uim_thetaim(matched_Vim_Omega[0], matched_Thetaim_Omega[0], 8)
plt.figure()
plt.errorbar(Uthetaplot/constant, mean_thetaim, yerr=std_thetaim, fmt='o', capsize=5)
plt.xlabel('Uim')
plt.ylabel('theta_im')
plt.xlim(0,210)
plt.ylim(0,50)

# #plot RE-RD-t
# # Create the figure
# plt.figure(figsize=(10, 12))
# # Subplot 1: ME
# plt.subplot(3, 1, 1)
# for i in range(5,10):
#     plt.plot(
#         t_inter.tolist(),
#         [0] + ME[i].tolist(),
#         color=colors[i-5],
#         label = f"$\\Omega$={Omega[i-5]}%"
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{R}_\mathrm{E}$ [kg/m$^2$/s]', fontsize=14)
# plt.legend(fontsize=14)
# # Subplot 2: MD
# plt.subplot(3, 1, 2)
# for i in range(5,10):
#     plt.plot(
#         t_inter.tolist(),
#         [0] + MD[i].tolist(),
#         color=colors[i-5],
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{R}_\mathrm{D}$ [kg/m$^2$/s]', fontsize=14)
# # Subplot 3: ME - MD
# plt.subplot(3, 1, 3)
# for i in range(5,10):
#     plt.plot(
#         t_inter.tolist(),
#         [0] + (ME[i] - MD[i]).tolist(),
#         color=colors[i-5],
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{R}_\mathrm{E}-\bar{R}_\mathrm{D}$ [kg/m$^2$/s]', fontsize=14)
# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()

# #ez and exz versus t
# plt.figure(figsize=(12, 10))
# plt.subplot(2, 1, 1)
# for i in range(5):
#     plt.plot(
#         t_inter.tolist(),
#         [np.nan] + exz_mean_t[i].tolist(),
#         color=colors[i],
#         label = f"$\\Omega$={Omega[i]}%"
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{e}_\mathrm{xz}$ [-]', fontsize=14)
# plt.legend(fontsize=14)
# plt.subplot(2, 1, 2)
# for i in range(5):
#     plt.plot(
#         t_inter.tolist(),
#         [np.nan] + ez_mean_t[i].tolist(),
#         color=colors[i],
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{e}_\mathrm{z}$ [-]', fontsize=14)
# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()

# #plot impact mass rate
# plt.figure(figsize=(12, 10))
# plt.subplot(3, 1, 1)
# for i in range(5):
#     plt.plot(
#         t_inter.tolist(),
#         [0] + RIM[i].tolist(),
#         color=colors[i],
#         label = f"$\\Omega$={Omega[i]}%"
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{R}_\mathrm{IM}$ [kg/m$^2$/s]', fontsize=14)
# plt.legend(fontsize=14)
# #impact velocity
# plt.subplot(3, 1, 2)
# for i in range(5):
#     plt.plot(
#         t_inter.tolist(),
#         [np.nan] + VIM_mean_t[i].tolist(),
#         color=colors[i],
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{U}_\mathrm{IM}$ [m/s]', fontsize=14)
# #impact angle
# plt.subplot(3, 1, 3)
# for i in range(5):
#     plt.plot(
#         t_inter.tolist(),
#         [np.nan] + ThetaIM_mean_t[i].tolist(),
#         color=colors[i],
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{\theta}_\mathrm{IM}$ [m/s]', fontsize=14)
# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()

# #erosion-impact
# plt.figure(figsize=(12, 10))
# plt.subplot(3, 1, 1)
# for i in range(5,10):
#     plt.plot(
#         t_inter.tolist(),
#         [0] + ME[i].tolist(),
#         color=colors[i-5],
#         label = f"$\\Omega$={Omega[i-5]}%"
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{R}_\mathrm{E}$ [kg/m$^2$/s]', fontsize=14)
# plt.legend(fontsize=14)
# plt.subplot(3, 1, 2)
# for i in range(5,10):
#     plt.plot(
#         t_inter.tolist(),
#         [0] + RIM[i].tolist(),
#         color=colors[i-5],
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{R}_\mathrm{IM}$ [kg/m$^2$/s]', fontsize=14)
# #impact velocity
# plt.subplot(3, 1, 3)
# for i in range(5,10):
#     plt.plot(
#         t_inter.tolist(),
#         [np.nan] + VIM_mean_t[i].tolist(),
#         color=colors[i-5],
#     )
# plt.xlabel('t [s]', fontsize=14)
# plt.ylabel(r'$\bar{U}_\mathrm{IM}$ [m/s]', fontsize=14)
# # Adjust layout and show plot
# plt.tight_layout()
# plt.show()

# #mean RE and RD in steady state
# RE_S,RD_S, RE_Sstd, RD_Sstd,RIM_S,RIM_std = np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10),np.zeros(10),np.zeros(10)
# VExz_all, VDxz_all, ez_all, exz_all, VIM_all, ThetaIM_all = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
# exz_all = defaultdict(list)
# for i in range(10) :
#     # RE_S[i] = np.mean(ME[i][int((3.5 / 5) * N_inter):])
#     # RD_S[i] = np.mean(MD[i][int((3.5 / 5) * N_inter):])
#     # RE_Sstd[i] = np.std(ME[i][int((3.5 / 5) * N_inter):])
#     # RD_Sstd[i] = np.std(MD[i][int((3.5 / 5) * N_inter):])
#     # RIM_S[i] = np.mean(RIM[i][int((3.5 / 5) * N_inter):])
#     # RIM_std[i] = np.std(RIM[i][int((3.5 / 5) * N_inter):])
#     # VExz_all[i] = [value for sublist in VExz_t[i][int((3.5 / 5) * N_inter):] for value in sublist]
#     # VDxz_all[i] = [value for sublist in VDxz_t[i][int((3.5 / 5) * N_inter):] for value in sublist]
#     # ez_all[i] = [value for sublist in ez_t[i][int((3.5 / 5) * N_inter):] for value in sublist]
#     # exz_all[i] = [value for sublist in exz_vector_t[i][int((3.5 / 5) * N_inter):] for value in sublist]
#     # VIM_all[i] = [value for sublist in VIM_t[i][int((3.5 / 5) * N_inter):] for value in sublist]
#     ThetaIM_all[i] = [value for sublist in Thetaim_vector_t[i][:int((1.5 / 5) * N_inter)] for value in sublist]
# VExz_ms, VDxz_ms, exz_ms, ez_ms, VIM_ms, ThetaIM_ms = np.zeros((10,2)), np.zeros((10,2)), np.zeros((10,2)), np.zeros((10,2)), np.zeros((10,2)), np.zeros((10,2))
# for i in range(10):
#     VExz_ms[i,0] = np.nanmean(VExz_mean_t[i])
#     VExz_ms[i,1] = np.nanstd(VExz_mean_t[i])
#     VDxz_ms[i,0] = np.nanmean(VDxz_mean_t[i])
#     VDxz_ms[i,1] = np.nanstd(VDxz_mean_t[i])
#     exz_ms[i,0] = np.nanmean(exz_mean_t[i])#np.mean(exz_all[i])
#     exz_ms[i,1] = np.nanstd(exz_mean_t[i])#np.std(exz_all[i])
#     ez_ms[i,0] = np.nanmean(ez_mean_t[i])#np.mean(ez_all[i])
#     ez_ms[i,1] = np.nanstd(ez_mean_t[i])#np.std(ez_all[i])
#     VIM_ms[i,0] = np.nanmean(VIM_mean_t[i])#np.mean(VIM_all[i])
#     VIM_ms[i,1] = np.nanstd(VIM_mean_t[i])#np.std(VIM_all[i])
#     ThetaIM_ms[i,0] = np.nanmean(ThetaIM_mean_t[i])#np.mean(ThetaIM_all[i])
#     ThetaIM_ms[i,1] = np.nanstd(ThetaIM_mean_t[i])#np.std(ThetaIM_all[i])
    
# #plot RE-Omega, needs more Shields number
# plt.figure(figsize=(10, 10))
# plt.errorbar(Omega, RE_S[0:5], yerr=RE_Sstd[0:5], fmt='o-', capsize=5, label=r'$\bar{R}_\mathrm{mass,E}, \tilde{\Theta}=0.05$', color='#008080')
# plt.errorbar(Omega, RD_S[0:5], yerr=RD_Sstd[0:5], fmt='o-', capsize=5, label=r'$\bar{R}_\mathrm{mass,D}, \tilde{\Theta}=0.05$', color='#fac205')
# plt.errorbar(Omega, RE_S[5:10], yerr=RE_Sstd[5:10], fmt='o-', capsize=5, label=r'$\bar{R}_\mathrm{mass,E}, \tilde{\Theta}=0.06$', color='#add8e6')
# plt.errorbar(Omega, RD_S[5:10], yerr=RD_Sstd[5:10], fmt='o-', capsize=5, label=r'$\bar{R}_\mathrm{mass,D}, \tilde{\Theta}=0.06$', color='#a0522d')
# plt.xticks([0,1,5,10,20])
# plt.xlabel('$\Omega$ [%]', fontsize=14)
# plt.ylabel(r'$\bar{R}_\mathrm{mass,E}, \bar{R}_\mathrm{mass,D}$ [kg/m$^2$/s]', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()
           
# #plot distribution of VExz and VDxz
# plt.figure(figsize=(8, 15))
# plt.subplot(2, 1, 1)
# for i in range(5):
#     # Calculate histogram (raw counts)
#     data = VExz_mean_t[i][-int((3.5 / 5) * N_inter):]
#     filtered_data = data[~np.isnan(data)]
#     counts, bin_edges = np.histogram(filtered_data, bins=50)
#     # Normalize the counts so their sum is 1
#     normalized_counts = counts / np.sum(counts)
#     # Create the step plot
#     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
#     # Prepare data for staircase plot
#     #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
# plt.xlabel('$U_\mathrm{E,xz}$ [m/s]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)
# plt.legend(fontsize=14)
# plt.subplot(2, 1, 2)
# for i in range(5):
#     # Calculate histogram (raw counts)
#     data = VDxz_mean_t[i][-int((3.5 / 5) * N_inter):]
#     filtered_data = data[~np.isnan(data)]
#     counts, bin_edges = np.histogram(filtered_data, bins=50)
#     # Normalize the counts so their sum is 1
#     normalized_counts = counts / np.sum(counts)
#     # Create the step plot
#     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
#     # Prepare data for staircase plot
#     #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
# plt.xlabel('$U_\mathrm{D,xz}$ [m/s]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)
# plt.show()

# #plot mean VExz and VDxz versus Omega
# plt.figure(figsize=(10, 10))
# plt.errorbar(Omega, VExz_ms[0:5,0], yerr=VExz_ms[0:5,1], fmt='o-', capsize=5, label=r'$\bar{U}_\mathrm{E,xz,steady}, \tilde{\Theta}=0.05$', color='#008080')
# plt.errorbar(Omega, VDxz_ms[0:5,0], yerr=VDxz_ms[0:5,1], fmt='o-', capsize=5, label=r'$\bar{U}_\mathrm{D,xz,steady}, \tilde{\Theta}=0.05$', color='#fac205')
# plt.errorbar(Omega, VExz_ms[5:10,0], yerr=VExz_ms[5:10,1], fmt='o-', capsize=5, label=r'$\bar{U}_\mathrm{E,xz,steady}, \tilde{\Theta}=0.06$', color='#add8e6')
# plt.errorbar(Omega, VDxz_ms[5:10,0], yerr=VDxz_ms[5:10,1], fmt='o-', capsize=5, label=r'$\bar{U}_\mathrm{D,xz,steady}, \tilde{\Theta}=0.06$', color='#a0522d')
# plt.xticks([0,1,5,10,20])
# plt.xlabel('$\Omega$ [%]', fontsize=14)
# plt.ylabel(r'$\bar{U}_\mathrm{E,xz,steady}, \bar{U}_\mathrm{D,xz,steady}$ [m/s]', fontsize=14)
# plt.legend(fontsize=14)
# plt.show() 
                                  
# #plot distribution of ez,exz
# exzpor_overone = np.zeros(10)
# for i in range(10):
#     count = sum(1 for x in exz_all[i] if x > 1)
#     exzpor_overone[i]=count/len(exz_all[i])
    
# plt.figure(figsize=(8, 15))
# #plt.subplot(2, 1, 1)
# for i in range(5):
#     # Calculate histogram (raw counts)
#     counts, bin_edges = np.histogram(exz_all[i], bins=50)
#     # Normalize the counts so their sum is 1
#     normalized_counts = counts / np.sum(counts)
#     # Create the step plot
#     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
#     # Prepare data for staircase plot
#     #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
# plt.xlabel('$e_\mathrm{xz}$ [-]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)
# plt.legend(fontsize=14)
# # plt.subplot(2, 1, 2)
# # for i in range(5):
# #     # Calculate histogram (raw counts)
# #     counts, bin_edges = np.histogram(ez_all[i], bins=50)
# #     # Normalize the counts so their sum is 1
# #     normalized_counts = counts / np.sum(counts)
# #     # Create the step plot
# #     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
# #     # Prepare data for staircase plot
# #     #bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # Calculate bin centers
# # plt.xlabel('$e_\mathrm{z}$ [-]', fontsize=14)
# # plt.ylabel('Normalized Frequency [-]', fontsize=14)
# plt.show()

# #distribution of impact velocities and angles
# constant = np.sqrt(9.81*D)

# plt.figure(figsize=(8, 15))
# # plt.subplot(2, 1, 1)
# # for i in range(5):
# #     # Calculate histogram (raw counts)
# #     counts, bin_edges = np.histogram(VIM_all[i], bins=50)
# #     # Normalize the counts so their sum is 1
# #     normalized_counts = counts / np.sum(counts)
# #     # Create the step plot
# #     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
# # plt.xlabel('$U_\mathrm{IM}$ [m/s]', fontsize=14)
# # plt.ylabel('Normalized Frequency [-]', fontsize=14)
# # plt.legend(fontsize=14)
# # plt.subplot(2, 1, 2)
# for i in range(5,10):
#     # Calculate histogram (raw counts)
#     counts, bin_edges = np.histogram(ThetaIM_all[i], bins=50)
#     # Normalize the counts so their sum is 1
#     normalized_counts = counts / np.sum(counts)
#     # Create the step plot
#     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i-5], label = f"$\\Omega$={Omega[i-5]}%")
# plt.xlabel(r'$\theta_\mathrm{IM}$ [$\circ$]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()

# #plot the means of e, VIM and ThetaIM versus Omega
# plt.figure(figsize=(10, 10))
# plt.errorbar(Omega, ez_ms[0:5,0], yerr=ez_ms[0:5,1], fmt='o-', capsize=5, label=r'$\bar{e}_\mathrm{z, steady}, \tilde{\Theta}=0.05$', color='#008080')
# plt.errorbar(Omega, exz_ms[0:5,0], yerr=exz_ms[0:5,1], fmt='o-', capsize=5, label=r'$\bar{e}_\mathrm{xz, steady}, \tilde{\Theta}=0.05$', color='#fac205')
# plt.errorbar(Omega, ez_ms[5:10,0], yerr=ez_ms[5:10,1], fmt='o-', capsize=5, label=r'$\bar{e}_\mathrm{z, steady}, \tilde{\Theta}=0.06$', color='#add8e6')
# plt.errorbar(Omega, exz_ms[5:10,0], yerr=exz_ms[5:10,1], fmt='o-', capsize=5, label=r'$\bar{e}_\mathrm{xz, steady}, \tilde{\Theta}=0.06$', color='#a0522d')
# plt.ylim(0,2)
# plt.xticks([0,1,5,10,20])
# plt.xlabel('$\Omega$ [%]', fontsize=14)
# plt.ylabel(r'$\bar{e}_\mathrm{xz, steady}, \bar{e}_\mathrm{z, steady}$ [-]', fontsize=14)
# plt.legend(fontsize=14)
# plt.show() 

# plt.figure(figsize=(15, 8))
# plt.subplot(1, 2, 1)
# plt.errorbar(Omega, VIM_ms[0:5,0], yerr=VIM_ms[0:5,1], fmt='o-', capsize=5, label=r'$\tilde{\Theta}=0.05$', color='#008080')
# plt.errorbar(Omega, VIM_ms[5:10,0], yerr=VIM_ms[5:10,1], fmt='o-', capsize=5, label=r'$\tilde{\Theta}=0.06$', color='#fac205')
# plt.xticks([0,1,5,10,20])
# plt.xlabel('$\Omega$ [%]', fontsize=14)
# plt.ylabel('$U_\mathrm{IM, steady}$ [m/s]', fontsize=14)
# plt.legend(fontsize=14)
# plt.subplot(1, 2, 2)
# plt.errorbar(Omega, ThetaIM_ms[0:5,0], yerr=ThetaIM_ms[0:5,1], fmt='o-', capsize=5, label=r'$\tilde{\Theta}=0.05$', color='#008080')
# plt.errorbar(Omega, ThetaIM_ms[5:10,0], yerr=ThetaIM_ms[5:10,1], fmt='o-', capsize=5, label=r'$\tilde{\Theta}=0.06$', color='#fac205')
# plt.xticks([0,1,5,10,20])
# plt.xlabel('$\Omega$ [%]', fontsize=14)
# plt.ylabel(r'$\theta_\mathrm{IM, steady}$ [$\circ$]', fontsize=14)
# plt.show()                                  
    
# plt.figure(figsize=(10, 10))
# plt.errorbar(Omega, RIM_S[0:5], yerr=RIM_std[0:5], fmt='o-', capsize=5, label=r'$\bar{R}_\mathrm{IM}, \tilde{\Theta}=0.05$', color='#008080')
# plt.errorbar(Omega, RIM_S[5:10], yerr=RIM_std[5:10], fmt='o-', capsize=5, label=r'$\bar{R}_\mathrm{IM}, \tilde{\Theta}=0.06$', color='#fac205')
# plt.xlabel('$\Omega$ [%]', fontsize=14)
# plt.ylabel(r'$\bar{R}_\mathrm{IM, steady}$ [kg/m$^2$/s]', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()


# #distribution of VDxz and Vim
# # plt.figure(figsize=(10, 10))
# # for i in range(4,5):
# #     # Calculate histogram (raw counts)
# #     counts, bin_edges = np.histogram(VIM_all[i], bins=50)
# #     # Normalize the counts so their sum is 1
# #     normalized_counts = counts / np.sum(counts)
# #     # Create the step plot
# #     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors[i], label = f"$\\Omega$={Omega[i]}%")
# # for i in range(4,5):
# #     # Calculate histogram (raw counts)
# #     counts, bin_edges = np.histogram(VDxz_all[i], bins=50)
# #     # Normalize the counts so their sum is 1
# #     normalized_counts = counts / np.sum(counts)
# #     # Create the step plot
# #     plt.step(bin_edges[:-1], normalized_counts, where='mid', color=colors_n[i], label = f"$\\Omega$={Omega[i]}%")
# # plt.xlabel('$U_\mathrm{D}, U_\mathrm{IM}$ [m/s]', fontsize=14)
# # plt.ylabel('Normalized Frequency [-]', fontsize=14)
# # plt.legend(fontsize=14)
# # plt.show()      