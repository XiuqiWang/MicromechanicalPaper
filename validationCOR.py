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
import module

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
t_ver = np.linspace(dt, 5, 500)
constant = np.sqrt(9.81*D)

#initialize
EDindices = defaultdict(list)
ME,MD,MoE,MoD = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
VExz_mean_t, VDxz_mean_t= defaultdict(list), defaultdict(list)
E_vector_t,D_vector_t= defaultdict(list),defaultdict(list)

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
VX = defaultdict(list)

X,Z = defaultdict(list),defaultdict(list)
for i in range(5):
    filename = f"data{i}"
    data = data_dict[filename]
    num_p = 2725
    ParticleID=np.linspace(num_p-10-300,num_p-10-1,300)
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

#trajectory and vertical velocity
id_p = 285
plt.figure(figsize=(12,12))
plt.subplot(3,1,1)
plt.plot(t_ver, Z[4][:,id_p]/D, linestyle='-', marker='.', color='k', markersize=3, linewidth=1)
plt.plot(t_ver[EDindices[4][id_p][0]], Z[4][EDindices[4][id_p][0],id_p]/D, 'ob', label='Ejections', markerfacecolor='none')
plt.plot(t_ver[EDindices[4][id_p][1]], Z[4][EDindices[4][id_p][1],id_p]/D, 'vb', label='Depositions', markerfacecolor='none')
legend_added = False
for t in t_ver[EDindices[4][id_p][0]]:
    if not legend_added:
        plt.axvline(x=t, color='k', linestyle='--', linewidth=1, label='Ejection cessation intervals (start)')
        legend_added = True  # Set flag to True after first legend entry
    else:
        plt.axvline(x=t, color='k', linestyle='--', linewidth=1)
legend_added = False
for t in t_ver[EDindices[4][id_p][1]]:
    if not legend_added:
        plt.axvline(x=t, color='k', linestyle=':', linewidth=1, label='Ejection cessation intervals (end)')
        legend_added = True  # Set flag to True after first legend entry
    else:
        plt.axvline(x=t, color='k', linestyle=':', linewidth=1)
plt.xlabel(r'$t$ [s]', fontsize=14)
plt.ylabel(r'$Z_\mathrm{p}/d$ [-]',fontsize=14)
plt.xlim(0,5)
plt.text(0.02, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.legend(fontsize=10)
plt.subplot(3,1,2)
plt.plot(t_ver, Z[4][:,id_p]/D, linestyle='-', marker='.', color='k', markersize=3, linewidth=1)
plt.plot(t_ver[Par[4][id_p][2][:,0]], Z[4][Par[4][id_p][2][:,0],id_p]/D, 'Dr', label='Impacts', markerfacecolor='none')
plt.plot(t_ver[Par[4][id_p][2][:,1]], Z[4][Par[4][id_p][2][:,1],id_p]/D, 'sr', label='Rebounds', markerfacecolor='none')
index_example2 = np.where(t_ver == 1.27)[0][0]
plt.axvline(t_ver[112],color='b', linestyle='-', label='An example cross-zero interval')
plt.axvline(t_ver[index_example2],color='b', linestyle='-')
legend_added = False
for start, end in t_ver[Par[4][id_p][1]]:
    if not legend_added:
        plt.axvline(x=start, color='k', linestyle='--', linewidth=1, label='Saltation intervals (start)')
        plt.axvline(x=end, color='k', linestyle=':', linewidth=1, label='Saltation intervals (end)')
        legend_added = True  # Set flag to True after first legend entry
    else:
        plt.axvline(x=start, color='k', linestyle='--', linewidth=1)
        plt.axvline(x=end, color='k', linestyle=':', linewidth=1)       
plt.xlabel(r'$t$ [s]', fontsize=14)
plt.ylabel(r'$Z_\mathrm{p}/d$ [-]',fontsize=14)
plt.xlim(0,5)
plt.text(0.02, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.legend(fontsize=10)
plt.subplot(3,1,3)
plt.plot(t_ver, VZ[4][id_p], linestyle='-', marker='.', color='k', markersize=3, linewidth=1)
plt.plot(t_ver[Par[4][id_p][2][:, 0]], VZ[4][id_p][Par[4][id_p][2][:, 0]], 'Dr', label='Impacts',markerfacecolor='none')
plt.plot(t_ver[Par[4][id_p][2][:, 1]], VZ[4][id_p][Par[4][id_p][2][:, 1]], 'sr', label='Rebounds',markerfacecolor='none')
index_example2 = np.where(t_ver == 1.27)[0][0]
plt.axvline(t_ver[112],color='b', linestyle='-')
plt.axvline(t_ver[index_example2],color='b', linestyle='-')
legend_added = False
for start, end in t_ver[Par[4][id_p][1]]:
    if not legend_added:
        plt.axvline(x=start, color='k', linestyle='--', linewidth=1, label='Saltation intervals (start)')
        plt.axvline(x=end, color='k', linestyle=':', linewidth=1, label='Saltation intervals (end)')
        legend_added = True  # Set flag to True after first legend entry
    else:
        plt.axvline(x=start, color='k', linestyle='--', linewidth=1)
        plt.axvline(x=end, color='k', linestyle=':', linewidth=1)       
plt.xlabel(r'$t$ [s]',fontsize=14)
plt.ylabel(r'$U_\mathrm{z}$ [m/s]',fontsize=14)
plt.xlim(0,5)
plt.text(0.02, 0.92, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# check the moments where particle is at the top of the trjectory
plt.figure()
plt.subplot(2,1,1)
plt.plot(t_ver, Z[4][:,id_p], linestyle='-', marker='.', color='k', markersize=3, linewidth=1)
plt.plot(t_ver[Par[4][id_p][3]], Z[4][:,id_p][Par[4][id_p][3]], 'rx',markerfacecolor='none')
plt.plot(t_ver[EDindices[4][id_p][2]], Z[4][EDindices[4][id_p][2],id_p], 'ob', label='Reptations', markerfacecolor='none')
plt.xlabel(r'$t$ [s]',fontsize=14)
plt.ylabel(r'$Z_p$ [m]',fontsize=14)
plt.subplot(2,1,2)
plt.plot(t_ver, VZ[4][id_p], linestyle='-', marker='.', color='k', markersize=3, linewidth=1)
plt.plot(t_ver[Par[4][id_p][3]], VZ[4][id_p][Par[4][id_p][3]], 'rx',markerfacecolor='none')
plt.xlabel(r'$t$ [s]',fontsize=14)
plt.ylabel(r'$U_\mathrm{z}$ [m/s]',fontsize=14)
plt.tight_layout()
plt.show()


#get the quantities from the steady state
exz_all,Vim_all,VD_all,ThetaD_all,Theta_all,zE_all,impact_list,ejection_list = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
Thetare_all, ThetaE_all = defaultdict(list),defaultdict(list)
Vre_all, Vsal_all = defaultdict(list),defaultdict(list)
Vrep_all = defaultdict(list)
for i in range (5):
    N_range = 0#int((3 / 5) * N_inter) #after 3s for dry case
    exz_all[i] = [value for sublist in exz_vector_t[i][N_range:] for value in sublist]
    Vim_all[i] = [value[0] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    Vre_all[i] = [value[10] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    Vsal_all[i] = [value[11] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    VD_all[i] = [value[0] for sublist in D_vector_t[i][N_range:] for value in sublist]
    Vrep_all[i] = [value[-1] for sublist in D_vector_t[i][N_range:] for value in sublist]
    ThetaD_all[i] = [value[1] for sublist in D_vector_t[i][N_range:] for value in sublist]
    Theta_all[i] = [value[7] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    Thetare_all[i] = [value[8] for sublist in IM_vector_t[i][N_range:] for value in sublist]
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
    EE = [value[5] for sublist in E_vector_t[i][N_range:] for value in sublist] #kinetic energy
    # zE_all[i] = [value[4] for sublist in E_vector_t[i][N_range:] for value in sublist]
    ThetaE_all[i] = [value[6] for sublist in E_vector_t[i][N_range:] for value in sublist]
    ejection_list[i] = [IDE, xE, vE, PE, EE, ThetaE_all[i]]
    # print('Ne/Nim',len(IDE)/len(IDim))

impact_ejection_list = defaultdict(list)
for i in range (5):
    impact_ejection_list[i]=module.match_ejection_to_impact(impact_list[i], ejection_list[i], dt)
Vim_all_values = [value for sublist in Vim_all.values() for value in sublist]
Vre_all_values = [value for sublist in Vre_all.values() for value in sublist]
Vsal_all_values = [value for sublist in Vsal_all.values() for value in sublist]
Vrep_all_values = [value for sublist in Vrep_all.values() for value in sublist]
Vim_bin = np.linspace(0, max(Vim_all_values)+1, 7) #make sure the range covers all the events
Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
Vimde_bin = np.linspace(0, max(Vimde_all_values)+1, 7)   

#global means and stds at all the wind conditions   
exz_all_values = [value for sublist in exz_all.values() for value in sublist]
VD_all_values = [value for sublist in VD_all.values() for value in sublist]
Thetare_all_values = [value for sublist in Thetare_all.values() for value in sublist]
Thetaim_all_values = [value for sublist in Theta_all.values() for value in sublist]
ThetaD_all_values = [value for sublist in ThetaD_all.values() for value in sublist]
matched_Vim_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][0]]
# matched_thetaim_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][1]]
matched_NE_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][2]]
matched_UE_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][3]]
matched_thetaE_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][5]]
#get the global NE, UE from all impacts and matched ejections 
CORmean_glo, N_COR, Thetare_mean_glo, Thetare_stderr_glo, Uimplot = module.BinUimCOR_equalbinsize(Vim_all_values,Vre_all_values,Thetare_all_values,Vim_bin)
# Usal_mean, Uincxplot = module.BinUincUsal(Vim_all_values, VD_all_values, Vsal_all_values, Vrep_all_values, Thetaim_all_values, ThetaD_all_values, Vimde_bin)
Pr_glo,Uplot,N_PrUre, Uiminbin, UDinbin = module.BinUimUd_equalbinsize(Vim_all_values,VD_all_values,Vimde_bin)
NEmean_glo, UEmean_glo, UEstd_glo, UEstderr_glo, ThetaEmean_glo, ThetaEstderr_glo, Uplot_NE, N_Einbin = module.get_ejection_ratios_equalbinsize(matched_Vim_all, VD_all_values, matched_NE_all, matched_UE_all, matched_thetaE_all, Vimde_bin)
Nim = np.array([1257, 695, 203, 32, 9, 4])

Thetaim_all_values = [value for sublist in Theta_all.values() for value in sublist]
Theta_mean_all = np.nanmean(Thetaim_all_values) #mean theta_im in all cases
#empirical data
Hcr = 1.5
UIM_UE_prin = np.linspace(0,190, 100)
thetaim_prin = np.arcsin(39.21/(UIM_UE_prin+105.73))
NE_prin = (-0.001*Hcr + 0.012)*UIM_UE_prin
VE_prin = (0.0538*Hcr + 1.0966)*np.sqrt(UIM_UE_prin)
COR_emp = 0.7469*np.exp(0.1374*Hcr)*(UIM_UE_prin)**(-0.0741*np.exp(0.214*Hcr))#Jiang et al. (2024) JGR
Pr_emp = 0.9945*Hcr**(-0.0166)*(1-np.exp(-0.1992*Hcr**(-0.8686)*UIM_UE_prin))
Thetare_emp = 24.56*np.ones(len(UIM_UE_prin))
ThetaE_emp = 0.2*UIM_UE_prin-1.4714*Hcr + 24.2
#anderson
Pr_and = 0.95*(1-np.exp(-2*UIM_UE_prin*constant))
# #Chen 2019
# COR_Chen = 0.62 + 0.0084*Uimplot - 0.63*np.sin(Theta_mean_all/180*np.pi)
NE_Chen = np.exp(-0.2 + 1.35*np.log(UIM_UE_prin*constant)-0.01*thetaim_prin/180*np.pi)
VE_Chen = np.exp(-1.48 + 0.082*UIM_UE_prin*constant-0.003*thetaim_prin/180*np.pi)/constant
#beladijne et al 2007
VE_Bel = 1.18*UIM_UE_prin**0.25
thetare_Bel = np.degrees(np.arcsin((0.3-0.15*np.sin(thetaim_prin))/(0.87-0.72*np.sin(thetaim_prin))))
thetaE_Bel = np.degrees(np.pi/2 + 0.1*(thetaim_prin-np.pi/2))
#exp data Jiang 2024; Hcr = 1.5D
Unsexp = [15, 35, 55, 75, 95, 125]
Nsexp = [0.05, 0.25, 0.65, 0.625, 1.1, 1.25]
CORexp_mean = [0.7, 0.7, 0.6, 0.65, 0.5, 0.6]
CORexp_std = [0.4, 0.3, 0.2, 0.15, 0.25, 0.3]
thetareexp = [25, 26, 24, 24, 34, 37]
thetareexp_std = [15, 20, 30, 28, 18, 27]
Prexp = [0.85, 0.94, 0.98, 0.99, 1.0, 1.0]
Uvsexp = [25, 45, 70, 95, 125]
Usexp = [5, 7, 8, 12.5, 6]
Usexp_std = [3, 4.5, 3.4, 5, 1]
thetaEexp = [28, 33, 35, 28, 20]
thetaEexp_std = [24, 24, 26, 15, 8]
error_hor = np.full(6,4.5)
#exp data Selmani 2024
UNE_Selmani = [20, 25, 55, 72, 100]
NE_Selmani = [0.5, 1, 3, 5, 10]

#only NE
# plt.figure(figsize=(6,5))
# # plt.plot(Uplot_NE/constant, NEmean_glo, 'o', label='This study', color='#3776ab')
# plt.scatter(Uplot_NE/constant, NEmean_glo, s=np.sqrt(N_Einbin)*5, label='This study', color='#3776ab')
# # plt.plot(Uplot_NE/constant, NE_Chen, label='No wind (Chen et al., 2019)', color='k')
# plt.plot(Unsexp, Nsexp,'x', label='With wind (Jiang et al., 2024)',color='k')
# # plt.ylim(0,1.3)
# plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)

plt.close("all")
plt.figure(figsize=(12,13.5))
plt.subplot(3,2,1)
line1 = plt.scatter(Uimplot/constant, CORmean_glo, s=np.sqrt(N_COR)*5, label='This study', color='#3776ab')
line2 = plt.errorbar(Unsexp, CORexp_mean, yerr=CORexp_std, fmt='x', capsize=5, label='Jiang et al. (2024)', color='k')
line3 = plt.plot(UIM_UE_prin, COR_emp, 'k-', label='Jiang et al. (2024)')
plt.xlim(0,190)
plt.ylim(-0.05,2.25)
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{e}$ [-]', fontsize=14)
plt.legend([line2[0],line3[0], line1],['Jiang et al. (2024)','Jiang et al. (2024)', 'This study'], fontsize=12)
plt.text(0.02, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(3,2,2)
line1 = plt.errorbar(Uimplot/constant, Thetare_mean_glo, yerr=Thetare_stderr_glo*np.sqrt(Nim), fmt='o', capsize=5, label='This study', color='#3776ab')
line2 = plt.errorbar(Unsexp, thetareexp, yerr=thetareexp_std, fmt='x', capsize=5, label='Jiang et al. (2024)', color='k')
line3 = plt.plot(UIM_UE_prin, Thetare_emp, 'k-')
line4 = plt.plot(UIM_UE_prin, thetare_Bel, 'k:', label='Beladijne et al. (2007)')
plt.ylim(-10,120)
plt.xlim(0,190)
plt.legend([line2[0],line3[0], line4[0], line1[0]],['Jiang et al. (2024)','Jiang et al. (2024)', 'Beladijne et al. (2007)', 'This study'], loc='upper left', bbox_to_anchor=(0.08, 0.99), fontsize=12)
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{re}$ [$^\circ$]', fontsize=14)
plt.text(0.02, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(3,2,3)
#plot Pr - Uim 
plt.scatter(Uplot/constant, Pr_glo, s=np.sqrt(N_PrUre)*5, label='This study', color='#3776ab')
plt.plot(UIM_UE_prin, Pr_emp, label='Jiang et al. (2024)',color='k')
plt.plot(Unsexp, Prexp, 'x', label='Jiang et al. (2024)', color='k')
plt.plot(UIM_UE_prin, Pr_and, 'k-.', label='Anderson & Haff (1991)')
plt.ylim(0,1.05)
plt.xlim(0,190)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$Pr$ [-]', fontsize=14)
plt.legend(fontsize=12)
plt.text(0.02, 0.92, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(3,2,4)
#plot \bar{NE} - Uim
plt.scatter(Uplot_NE/constant, NEmean_glo, s=np.sqrt(N_Einbin)*5, label='This study', color='#3776ab')
plt.plot(UIM_UE_prin, NE_prin, 'k-', label='Jiang et al. (2024)')
plt.plot(Unsexp, Nsexp,'x', label='Jiang et al. (2024)',color='k')
plt.plot(UNE_Selmani, NE_Selmani, 'dk', label='Selmani et al. (2024)')
plt.plot(UIM_UE_prin, NE_Chen, 'k-.', label='Chen et al. (2019)')
plt.ylim(0,20)
plt.xlim(0,190)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
plt.legend(loc='upper left', bbox_to_anchor=(0.08, 0.99), fontsize=12)
plt.text(0.02, 0.92, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(3,2,5)
#plot \bar{UE} - Uim
line1 = plt.errorbar(Uplot_NE/constant, UEmean_glo/constant, yerr=UEstd_glo/constant, fmt='o', capsize=5, label='This study', color='#3776ab')
line2 = plt.errorbar(Uvsexp, Usexp, yerr=Usexp_std, fmt='x', capsize=5, label='Jiang et al. (2024)', color='k')
line3 = plt.plot(UIM_UE_prin, VE_prin, 'k-')
line4 = plt.plot(UIM_UE_prin, VE_Chen, 'k-.')
line5 = plt.plot(UIM_UE_prin, VE_Bel, 'k:', label='Beladijne et al. (2007)')
plt.ylim(0,32.5)
plt.xlim(0,190)
plt.legend([line2[0],line3[0], line4[0], line5[0], line1[0]],['Jiang et al. (2024)','Jiang et al. (2024)', 'Chen et al. (2019)', 'Beladijne et al. (2007)', 'This study'], loc='upper left', bbox_to_anchor=(0.08, 0.99), fontsize=12)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(e)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(3,2,6)
line1 = plt.errorbar(Uplot_NE/constant, ThetaEmean_glo, yerr=ThetaEstderr_glo*np.sqrt(N_Einbin), fmt='o', capsize=5, label='This study', color='#3776ab')
line2 = plt.errorbar(Uvsexp, thetaEexp, yerr=thetaEexp_std, fmt='x', capsize=5, label='Jiang et al. (2024)', color='k')
line3 = plt.plot(UIM_UE_prin, ThetaE_emp, 'k-')
line4 = plt.plot(UIM_UE_prin, thetaE_Bel, 'k:', label='Beladijne et al. (2007)')
plt.ylim(0,140)
plt.xlim(0,190)
plt.legend([line2[0],line3[0], line4[0], line1[0]],['Jiang et al. (2024)','Jiang et al. (2024)', 'Beladijne et al. (2007)', 'This study'], loc='upper left', bbox_to_anchor=(0.08, 0.99), fontsize=12)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{E}$ [$^\circ$]', fontsize=14)
plt.text(0.02, 0.92, '(f)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

#sensitivity analysis of dx on global mean NE
NE_lim = 2.42
def average(matched_NE_Omega):
    matched_NE_Omega = np.array(matched_NE_Omega)  # Convert to NumPy array
    return np.mean(matched_NE_Omega)

NE_glo = [0.88, 1.31, 1.69, 2.14, 2.18, 2.21, 2.22, 2.22] #average(matched_NE_all)
dx = [5, 10, 20, 60, 70, 80, 100, 200]
plt.figure()
plt.plot(dx,NE_glo, 'o')
plt.plot(np.linspace(0,200, 100), np.ones(100)*NE_lim, '--k', label='limit')
plt.xlabel('dx/d')
plt.ylabel('NE_global')
plt.legend()

#distribution of Uim for different Shields
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# Plot PDF of U_im/sqrt(gd)
for i in range(5):
    # Calculate histogram (density=True for probability density)
    counts, bin_edges = np.histogram(Vim_all[i]/constant, bins=50, density=True)
    # Create the step plot
    plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\tilde{{\\Theta}}$=0.0{i+2}")
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel('Probability Density [-]', fontsize=14)
# plt.ylim(-0.002, 0.03)
plt.legend(fontsize=12)
# Plot PDF of theta_im
plt.subplot(1, 2, 2)
for i in range(5):
    # Calculate histogram (density=True for probability density)
    counts, bin_edges = np.histogram(Theta_all[i], bins=50, density=True)
    # Create the step plot
    plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\tilde{{\\Theta}}$=0.0{i+2}")
plt.xlabel(r'$\theta_\mathrm{im}$ [$^\circ$]', fontsize=14)
plt.ylabel('Probability Density [-]', fontsize=14)
plt.tight_layout()
plt.show()
# plt.subplot(2, 2, 3)
# plt.errorbar(Omega, Vim_mean_glo, yerr=Vim_std_glo, fmt='o', capsize=5, color='#3776ab')
# plt.ylim(0, 5.5)
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.text(0.03, 0.94, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(2, 2, 4)
# plt.errorbar(Omega, Thetaim_mean_glo, yerr=Thetaim_std_glo, fmt='o', capsize=5, color='#3776ab')
# plt.ylim(0, 30)
# plt.xlabel(r'$\Omega$ [$\%$]', fontsize=14)
# plt.ylabel(r'$\theta_\mathrm{im}$ [$\circ$]', fontsize=14)
# plt.text(0.03, 0.94, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')



#for detecting ejection
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

#plot COR - Thetaim
# plt.figure(figsize=(10, 10))
# # for i in range(1,5):
# #     plt.errorbar(Thetaimplot, COR_theta_mean[i], yerr=COR_theta_std[i], fmt='o', capsize=5, label=fr'$\tilde{{\Theta}}=0.0{i+2}$',color=colors[i])
# plt.errorbar(Thetaimplot, COR_theta_mean_glo, yerr=COR_theta_std_glo, fmt='o', capsize=5)
# plt.xlabel(r'$\theta_{IM}$ [deg]', fontsize=14)
# plt.ylabel(r'$\bar{e}_\mathrm{xz,steady}$ [-]', fontsize=14)
# plt.legend(fontsize=14)
# plt.show()

# #distribution of impact angles in all cases
# plt.figure()
# # Calculate histogram (raw counts)
# data = Thetaim_all_values
# counts, bin_edges = np.histogram(data, bins=15)
# # Normalize the counts so their sum is 1
# normalized_counts = counts / np.sum(counts)
# # Create the step plot
# plt.step(bin_edges[:-1], normalized_counts, where='mid')
# plt.xlabel(r'$\theta_{IM}$ [deg]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)

# #distribution of ejection height
# plt.figure()
# # Calculate histogram (raw counts)
# data = [x / D for x in zE_all_values]
# counts, bin_edges = np.histogram(data, bins=30)
# # Normalize the counts so their sum is 1
# normalized_counts = counts / np.sum(counts)
# # Create the step plot
# plt.step(bin_edges[:-1], normalized_counts, where='mid')
# plt.xlabel(r'$z_{E}/D$ [-]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)

# plt.figure()
# # Calculate histogram (raw counts)
# for i in range(5):
#     data = [x for x in ejection_list[i][2]]
#     counts, bin_edges = np.histogram(data, bins=30)
#     # Normalize the counts so their sum is 1
#     normalized_counts = counts / np.sum(counts)
#     # Create the step plot
#     plt.step(bin_edges[:-1], normalized_counts, where='mid',  label=fr'$\tilde{{\Theta}}=0.0{i+2}$',color=colors[i])
# plt.xlabel(r'$U_{E}$ [m/s]', fontsize=14)
# plt.ylabel('Normalized Frequency [-]', fontsize=14)
# plt.legend()