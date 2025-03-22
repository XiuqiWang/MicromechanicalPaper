# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:40:59 2024

@author: WangX3
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import h5py
from module import store_sal_id_data
from module import store_particle_id_data
from module import BinUimCOR
from module import BinUimUd
from module import match_ejection_to_impact
from module import get_ejection_ratios

# Load the data from the mat data
# File paths for the .mat files
file_paths = ['matdata_SA/S005M20LBIni.mat', 'matdata_SA/FTSS005M20LBIni.mat', 'matdata_SA/FiftyTSS005M20LBIni.mat']
# Initialize a list to store the result lists
datalists = []
# Iterate through the file paths
for idx, file_path in enumerate(file_paths):
    # Open the .mat v7.3 file using h5py
    with h5py.File(file_path, 'r') as mat_file:
        # Access the 'data' group (change this if the group name differs)
        data = mat_file['data']       
        # Initialize an empty list to store the extracted data
        result_list = []        
        # Iterate through the cells in 'data' (assuming 'data' is a 1x500 array)
        for i in range(data.shape[0]):  # Assuming 'data' is a 1x500 array
            cell_data = data[i,0]  # Access each cell (a MATLAB struct)            
            # If the cell data is a reference, dereference it
            if isinstance(cell_data, h5py.Reference):
                cell_data = mat_file[cell_data]  # Dereference the object            
            # Extract variables from the struct
            position = cell_data['Position'][:].T
            velocity = cell_data['Velocity'][:].T
            radius = cell_data['Radius'][:].T
            # Append the variables to the result list
            result_list.append({'Position': position, 'Velocity': velocity, 'Radius': radius})        
        # Append the result list to the main list
        datalists.append(result_list)
        
# Now datalists should contain three lists, each with 500 items (datalist0, datalist1, datalist2)
datalist0, datalist1, datalist2 = datalists[0], datalists[1], datalists[2]

#basic parameters
D = 0.00025
coe_h = 13.5 #1.5d critial height for an ejected particle to reach
coe_sal_h = 17 #5d critical height for a particle to be considered saltation
N_inter = 100 #number of output timesteps for erosion and deposition properties
dt0 = 5/500
dt1 = 5/5003
dt2 = 5/50802
dt = [dt0,dt1,dt2]
t0 = np.linspace(dt0, 5, 500)
t1 = np.linspace(dt1, 5, 5003)
t2 = np.linspace(dt2, 5, 50802)
t = [t0, t1, t2]
t_inter = np.linspace(0,5,N_inter+1)
labels = ['500 steps','5000 steps','50000 steps']

#initialize
EDindices = defaultdict(list)
ME,MD,MoE,MoD = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
VExz_mean_t, VDxz_mean_t= defaultdict(list), defaultdict(list)
E_vector_t,VD_vector_t= defaultdict(list),defaultdict(list)

exz_mean_t,ez_mean_t = defaultdict(list),defaultdict(list)
# ez_t = defaultdict(list)
# exz_t = defaultdict(list)
exz_vector_t = defaultdict(list)
IM_vector_t,Thetaim_vector_t = defaultdict(list),defaultdict(list)
VIM_mean_t,ThetaIM_mean_t = defaultdict(list), defaultdict(list)
# VIM_t = defaultdict(list)
# Thetaim_t = defaultdict(list)
RIM = defaultdict(list)
Par = defaultdict(list)
VZ = defaultdict(list)

X,Z = defaultdict(list),defaultdict(list)
for i in range(3):
    #filename = f"data{i}"
    data = datalists[i]
    num_p = 2725    
    ParticleID=np.linspace(num_p-10-300,num_p-10-1,300)
    ParticleID_int = ParticleID.astype(int)
    #cal erosion and deposition properties for each Omega
    #EDindices, E, VX, VExVector, VEzVector, VEx, VEz, ME, MD
    EDindices[i], ME[i], MD[i], VExz_mean_t[i], VDxz_mean_t[i], VD_vector_t[i], E_vector_t[i]=store_particle_id_data(data,ParticleID_int,coe_h,dt[i],N_inter,D)
    ParticleID_sal=np.linspace(num_p-310,num_p-1,310)
    ParticleID_salint = ParticleID_sal.astype(int)
    X[i] = np.array([[time_step['Position'][i][0] for i in ParticleID_salint] for time_step in data])
    Z[i] = np.array([[time_step['Position'][i][2] for i in ParticleID_salint] for time_step in data])
    Par[i], VZ[i], exz_mean_t[i], ez_mean_t[i], VIM_mean_t[i], ThetaIM_mean_t[i], RIM[i], exz_vector_t[i], IM_vector_t[i], Thetaim_vector_t[i]=store_sal_id_data(data,ParticleID_salint, coe_sal_h, dt[i], N_inter, D)


exz_all,Vim_all,VD_all,Theta_all,impact_list,ejection_list = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
for i in range (3):
    N_range = int((3.5 / 5) * N_inter)
    exz_all[i] = [value for sublist in exz_vector_t[i][N_range:] for value in sublist]
    Vim_all[i] = [value[0] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    VD_all[i] = [value for sublist in VD_vector_t[i][N_range:] for value in sublist]
    Theta_all[i] = [value for sublist in Thetaim_vector_t[i][N_range:] for value in sublist]
    IDim = [value[1] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    IDre = [value[2] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    xim = [value[3] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    xre = [value[4] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    xcol = [value[5] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    Pim = [value[6] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    impact_list[i] = [IDim, IDre, xim, xre, Vim_all[i], xcol, Pim]
    vE = [value[0] for sublist in E_vector_t[i][N_range:] for value in sublist]
    IDE = [value[1] for sublist in E_vector_t[i][N_range:] for value in sublist]
    xE = [value[2] for sublist in E_vector_t[i][N_range:] for value in sublist]
    PE = [value[3] for sublist in E_vector_t[i][N_range:] for value in sublist]
    ejection_list[i] = [IDE, xE, vE, PE]
    print('Ne/Nim',len(IDE)/len(IDim))

Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
Vimde_bin = np.linspace(min(Vimde_all_values), max(Vimde_all_values), 8)    
CORmean,CORstd,Uimplot=defaultdict(list),defaultdict(list),defaultdict(list)
Pr,Uplot=defaultdict(list),defaultdict(list)
impact_ejection_list = defaultdict(list)
matched_Vim, matched_NE, matched_UE = defaultdict(list),defaultdict(list),defaultdict(list)
NE_mean,UE_mean,UE_std, Uplot_NE=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
for i in range (3):
    impact_ejection_list=match_ejection_to_impact(impact_list[i], ejection_list[i], dt[i])
    matched_Vim[i] = [element for element in impact_ejection_list[0]]
    matched_NE[i] = [element for element in impact_ejection_list[1]]
    matched_UE[i] = [element for element in impact_ejection_list[2]]

for i in range (3):
    NE_mean[i], UE_mean[i], UE_std[i], Uplot_NE[i] = get_ejection_ratios(matched_Vim[i], matched_NE[i], matched_UE[i], 8)
    CORmean[i],CORstd[i],Uimplot[i] = BinUimCOR(Vim_all[i],exz_all[i], 8)
    Pr[i],Uplot[i] = BinUimUd(Vim_all[i],VD_all[i],Vimde_bin)
 
constant = np.sqrt(9.81*D)
# plt.figure()
# for i in range(3):
#     E_check = [value[0] for value in result[i]]
#     N_check = [len(value) for value in E_check]
#     plt.plot(Vim_all[i]/constant, N_check,'.',label=labels[i])
# plt.xlabel('Uim/sqrt(gd)')
# plt.ylabel('NE')
# plt.legend()


#plot \bar{NE} - Uim
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.plot(Uplot_NE[i]/constant, NE_mean[i], 'o', label=labels[i])
plt.xlabel(r'$U_{IM}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
plt.legend(fontsize=14)
plt.show()

#plot \bar{UE} - Uim
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.errorbar(Uplot_NE[i]/constant, UE_mean[i]/constant, yerr=UE_std[i]/constant, fmt='o', capsize=5, label=labels[i])
plt.xlabel(r'$U_{IM}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{U}_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
plt.legend(fontsize=14)
plt.show()

#plot COR - Uim
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.errorbar(Uimplot[i]/constant, CORmean[i], yerr=CORstd[i], fmt='o', capsize=5, label=labels[i])
plt.xlabel(r'$U_{IM}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{e}_\mathrm{xz,steady}$ [-]', fontsize=14)
plt.legend(fontsize=14)
plt.show()

#plot Pr - Uim 
plt.figure(figsize=(10, 10))
for i in range(3):
    plt.plot(Uplot[i]/constant, Pr[i], 'o', label=labels[i])
plt.xlabel(r'$U_{IM}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$P_\mathrm{r}$ [-]', fontsize=14)
plt.legend(fontsize=14)
plt.show()

#verify the detected impacts and rebounds

plt.figure(figsize=(10, 15))
for j in range(3):
    plt.subplot(3,1,j+1)
    for i in [297,299]:
        plt.plot(X[j][:,i]/D, t[j], '.-')   
        if Par[j][i][2].any():
            plt.plot(X[j][Par[j][i][2][:,0],i]/D, t[j][Par[j][i][2][:,0]], 'D',color='red')
    for i in [297,299]:
        if EDindices[j][i][0].any():
            plt.plot(X[j][EDindices[j][i][0],i]/D, t[j][EDindices[j][i][0]], 'o',color='green')
    plt.title(labels[j])
plt.xlabel('x/D [m]')
plt.ylabel('t [s]')
plt.legend(fontsize=14)


id_p = 297
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.plot(t[i], Z[i][:,id_p]/D, linestyle='-', marker='.', label='vertical position')
    #axs[i].plot(Par[i][id_p][0], np.zeros(len(Par[i][id_p][0])), 'x', label='Collision moments')
    plt.plot(t[i][Par[i][id_p][2][:,0]], Z[i][Par[i][id_p][2][:,0],id_p]/D, 'D', label='Impact')
    plt.plot(t[i][Par[i][id_p][2][:,1]], Z[i][Par[i][id_p][2][:,1],id_p]/D, 'o', label='Rebound')
    #axs[i].plot(t[i][Par[i][id_p][1][:, 0]], Z[i][id_p][Par[i][id_p][1][:, 0]], '*', label='Mobile')
    plt.plot(t[i][EDindices[i][id_p][0]], Z[i][EDindices[i][id_p][0],id_p]/D, 'o', label='Ejection')
    plt.plot(t[i][EDindices[i][id_p][1]], Z[i][EDindices[i][id_p][1],id_p]/D, 'o', label='Deposition')
plt.xlabel('t [s]')
plt.ylabel('Z_p/D [-]')
plt.legend()
plt.show()

id_p = 297
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
for i in range(3):
    # Subplot 1
    axs[i].plot(t[i], VZ[i][id_p], linestyle='-', marker='.', label='Particle velocity')
    #axs[i].plot(Par[i][id_p][0], np.zeros(len(Par[i][id_p][0])), 'x', label='Collision moments')
    axs[i].plot(t[i][Par[i][id_p][2][:, 0]], VZ[i][id_p][Par[i][id_p][2][:, 0]], 'D', label='Impact')
    axs[i].plot(t[i][Par[i][id_p][2][:, 1]], VZ[i][id_p][Par[i][id_p][2][:, 1]], 'o', label='Rebound')
    axs[i].set_xlabel('t [s]')
    axs[i].set_ylabel('Uz [m/s]')
# Add a global legend for the last subplot
axs[2].legend()
# Show the plot
plt.tight_layout()
plt.show()

#verify the detected erosions and depositions
id_pe = 299
# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
# Subplot 1
axs[0].plot(t0, VZ[0][id_pe], linestyle='-', marker='.', label='Particle trajectory')
axs[0].plot(t0[EDindices[0][id_pe][0]], VZ[0][id_pe][EDindices[0][id_pe][0]], 'o', label='Erosion')
axs[0].plot(t0[EDindices[0][id_pe][1]], VZ[0][id_pe][EDindices[0][id_pe][1]], 'x', label='Deposition')
axs[0].set_xlabel(r'$time$ [s]', fontsize=16)
axs[0].set_ylabel('Uz [m/s]', fontsize=16)
axs[0].legend(fontsize=12)
# Subplot 2
axs[1].plot(t1, VZ[1][id_pe], linestyle='-', marker='.', label='Particle trajectory')
axs[1].plot(t1[EDindices[1][id_pe][0]], VZ[1][id_pe][EDindices[1][id_pe][0]], 'o', label='Erosion')
axs[1].plot(t1[EDindices[1][id_pe][1]], VZ[1][id_pe][EDindices[1][id_pe][1]], 'x', label='Deposition')
axs[1].set_xlabel(r'$time$ [s]', fontsize=16)
axs[1].set_ylabel('Uz [m/s]', fontsize=16)
# Subplot 3
axs[2].plot(t2, VZ[2][id_pe], linestyle='-', marker='.', label='Particle trajectory')
axs[2].plot(t2[EDindices[2][id_pe][0]], VZ[2][id_pe][EDindices[2][id_pe][0]], 'o', label='Erosion')
axs[2].plot(t2[EDindices[2][id_pe][1]], VZ[2][id_pe][EDindices[2][id_pe][1]], 'x', label='Deposition')
axs[2].set_xlabel(r'$time$ [s]', fontsize=16)
axs[2].set_ylabel('Uz [m/s]', fontsize=16)

# Adjust layout and display
plt.tight_layout()
plt.show()


#compare ME and MD
fig, axs = plt.subplots(2, 1, figsize=(10, 15))
# Subplot 1
plt.subplot(2, 1, 1)
for i in range(3):
    plt.plot(
        t_inter,
        [0] + ME[i].tolist(),
    )
plt.xlabel('t [s]')
plt.ylabel(r'$R_\mathrm{E}$ [/s]', fontsize=12)
plt.legend(['500 steps','5000 steps','50000 steps'])
# Subplot 2
plt.subplot(2, 1, 2)
for i in range(3):
    plt.plot(
        t_inter,
        [0] + MD[i].tolist(),
    )
plt.xlabel('t [s]')
plt.ylabel(r'$R_\mathrm{D}$ [/s]', fontsize=12)
# Show the plot
plt.tight_layout()
plt.show()

#ejection velocities
# fig, axs = plt.subplots(2, 1, figsize=(10, 12))
# # Subplot 1
# plt.subplot(2, 1, 1)
# for i in range(3):
#     plt.plot(
#         t_inter,
#         [0] + VEx[i].tolist(),
#     )
# plt.xlabel('t [s]')
# plt.ylabel(r'$U_\mathrm{E,x}$ [m$^2$/s]', fontsize=12)
# plt.legend(['500 steps','5000 steps','50000 steps'])
# # Subplot 2
# plt.subplot(2, 1, 2)
# for i in range(3):
#     plt.plot(
#         t_inter,
#         [0] + VEz[i].tolist(),
#     )
# plt.xlabel('t [s]')
# plt.ylabel(r'$U_\mathrm{E,z}$ [m$^2$/s]', fontsize=12)
# # Show the plot
# plt.tight_layout()
# plt.show()


#compare impact rates
fig, axs = plt.subplots(2, 1, figsize=(10, 12))
# Subplot 1
plt.subplot(2, 1, 1)
for i in range(3):
    plt.plot(
        t_inter,
        [0] + RIM[i].tolist(),
    )
plt.xlabel('t [s]')
plt.ylabel(r'$R_\mathrm{IM}$ [/s]', fontsize=12)
plt.legend(['500 steps','5000 steps','50000 steps'])
# Subplot 2
plt.subplot(2, 1, 2)
for i in range(3):
    plt.plot(
        t_inter,
        [0] + VIM_mean_t[i].tolist(),
    )
plt.xlabel('t [s]')
plt.ylabel(r'$U_\mathrm{IM}$ [m/s]', fontsize=12)
# Show the plot
plt.tight_layout()
plt.show()


