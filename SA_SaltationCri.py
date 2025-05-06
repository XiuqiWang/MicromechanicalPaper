# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:27:48 2025

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

#initialize
EDindices = defaultdict(list)
ME,MD,MoE,MoD = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
VExz_mean_t, VDxz_mean_t= defaultdict(list), defaultdict(list)
E_vector_t,D_vector_t= defaultdict(list),defaultdict(list)

exz_mean_t,ez_mean_t = defaultdict(list),defaultdict(list)
exz_vector_t = defaultdict(list)
IM_vector_t = defaultdict(list)
VIM_mean_t,ThetaIM_mean_t = defaultdict(list), defaultdict(list)
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
    EDindices[i], ME[i], MD[i], VExz_mean_t[i], VDxz_mean_t[i], D_vector_t[i], E_vector_t[i]=module.store_particle_id_data(data,ParticleID_int,coe_h,dt,N_inter,D)
    #cal rebound properties for each Omega
    ParticleID_sal=np.linspace(num_p-310,num_p-1,310)
    ParticleID_salint = ParticleID_sal.astype(int)
    X[i] = np.array([[time_step['Position'][i][0] for i in ParticleID_salint] for time_step in data])
    Z[i] = np.array([[time_step['Position'][i][2] for i in ParticleID_salint] for time_step in data])
    Par[i], VZ[i], exz_mean_t[i], ez_mean_t[i], VIM_mean_t[i], ThetaIM_mean_t[i], RIM[i], exz_vector_t[i], IM_vector_t[i]=module.store_sal_id_data(data,ParticleID_salint, coe_sal_h, dt, N_inter, D)

#get the quantities from the steady state
exz_all,Vim_all,VD_all,ThetaD_all,Theta_all,zE_all,impact_list,ejection_list = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
for i in range (5):
    N_range = 0
    exz_all[i] = [value for sublist in exz_vector_t[i][N_range:] for value in sublist]
    Vim_all[i] = [value[0] for sublist in IM_vector_t[i][N_range:] for value in sublist]
    VD_all[i] = [value[0] for sublist in D_vector_t[i][N_range:] for value in sublist]
    ThetaD_all[i] = [value[1] for sublist in D_vector_t[i][N_range:] for value in sublist]
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
    EE = [value[4] for sublist in E_vector_t[i][N_range:] for value in sublist]
    ejection_list[i] = [IDE, xE, vE, PE, EE]
    
impact_ejection_list = defaultdict(list)
for i in range (5):
    impact_ejection_list[i]=module.match_ejection_to_impact(impact_list[i], ejection_list[i], dt)
Vim_all_values = [value for sublist in Vim_all.values() for value in sublist]
Vim_bin = np.linspace(min(Vim_all_values), max(Vim_all_values), 7)
Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
Vimde_bin = np.linspace(min(Vimde_all_values), max(Vimde_all_values), 7)   

#global means and stds at all the wind conditions   
exz_all_values = [value for sublist in exz_all.values() for value in sublist]
VD_all_values = [value for sublist in VD_all.values() for value in sublist]
matched_Vim_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][0]]
matched_NE_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][2]]
matched_UE_all = [element for key in impact_ejection_list for element in impact_ejection_list[key][3]]
#get the global NE, UE from all impacts and matched ejections 
CORmean_glo,CORstd_glo,CORstderr_glo,Uimplot = module.BinUimCOR_equalbinsize(Vim_all_values,exz_all_values,Vim_bin)
Pr_glo,Uplot,N_PrUre = module.BinUimUd_equalbinsize(Vim_all_values,VD_all_values,Vimde_bin)
NEmean_glo, UEmean_glo, UEstd_glo, UEstderr_glo, Uplot_NE, N_Einbin = module.get_ejection_ratios_equalbinsize(matched_Vim_all, VD_all_values, matched_NE_all, matched_UE_all, Vimde_bin)

constant = np.sqrt(9.81*D)
#empirical data
Hcr = 1.5
NE_prin = (-0.001*Hcr + 0.012)*Uplot_NE/constant
UIM_UE_prin = np.linspace(0,120, 20)
VE_prin = (0.0538*Hcr + 1.0966)*np.sqrt(UIM_UE_prin)
COR_emp = 0.7469*np.exp(0.1374*Hcr)*(Uimplot/constant)**(-0.0741*np.exp(0.214*Hcr))#Jiang et al. (2024) JGR
Pr_emp = 0.9945*Hcr**(-0.0166)*(1-np.exp(-0.1992*Hcr**(-0.8686)*Uplot/constant))
#exp data Jiang 2024; Hcr = 1.5D
Unsexp = [15, 35, 55, 75, 95, 125]
Nsexp = [0.05, 0.25, 0.65, 0.625, 1.1, 1.25]
CORexp_mean = [0.7, 0.7, 0.6, 0.65, 0.5, 0.6]
CORexp_std = [0.4, 0.3, 0.2, 0.15, 0.25, 0.3]
Prexp = [0.85, 0.94, 0.98, 0.99, 1.0, 1.0]
Uvsexp = [25, 45, 70, 95, 125]
Usexp = [5, 7, 8, 12.5, 6]
Usexp_std = [3, 4.5, 3.4, 5, 1]
error_hor = np.full(6,4.5)

#collecting data from the outputs with multiple criteria values
#coe_sal_h = 13.5
U1 = [0.69916092, 2.06192551, 3.4246901 , 4.78745469, 6.15021928,
       7.51298387]
CORmean_glo1 = [1.2675749064308452,
 0.5042023611503276,
 0.3792662977188726,
 0.3211963781790539,
 0.29614444860536787,
 0.22192864762170092]
CORstd_glo1 = [2.056912919997267,
 0.31120276536293484,
 0.22094221654173452,
 0.21227426750456707,
 0.20420081411949395,
 0.08692174498511492]
Upr1 = [0.68881508, 2.05346073, 3.41810639, 4.78275204, 6.14739769,
       7.51204334]
Pr_glo1 = [0.34091493, 0.89015817, 0.95327103, 0.96363636, 0.86956522,
       0.66666667]
NEmean_glo1 = [0.4243125160627088,
 1.2302284710017575,
 1.4236760124610592,
 1.290909090909091,
 1.5217391304347827,
 1.6666666666666667]
UEmean_glo1 = [0.23652281151004947,
 0.2343928646670382,
 0.22445167872038993,
 0.2590552805806972,
 0.1767041834333363,
 0.2904305389214352]
UEstd_glo1 = [0.20336251996938237,
 0.1910938731245986,
 0.17581966643924038,
 0.2591527578208411,
 0.07910517659465834,
 0.2651680593999378]
#coe_sal_h = 17
U2 = [0.81669995, 2.15809381, 3.49948767, 4.84088152, 6.18227538,
       7.52366923]
CORmean_glo2 = [0.9759709842381145,
 0.5250315346428885,
 0.39491516436069574,
 0.3248297281694904,
 0.3344450044307521,
 0.22192864762170092]
CORstd_glo2 = [0.9870135590030893,
 0.29677966624739005,
 0.20763868078976605,
 0.20369267302315688,
 0.2038620151622871,
 0.08692174498511492]
Upr2 = [0.68881508, 2.05346073, 3.41810639, 4.78275204, 6.14739769,
       7.51204334]
Pr_glo2 = [0.17791313, 0.85970819, 0.94444444, 0.96      , 0.84210526,
       0.66666667]
NEmean_glo2 = [0.4093604744350056,
 2.1133557800224465,
 2.414814814814815,
 2.52,
 2.210526315789474,
 2.6666666666666665]
UEmean_glo2 = [0.23929936066072646,
 0.2296063013843892,
 0.2304046922950471,
 0.2597028849615068,
 0.18893098474465753,
 0.2890656938451392]
UEstd_glo2 = [0.20214307141035465,
 0.1995068241078786,
 0.17723247153338031,
 0.22565198763967678,
 0.07780960312218689,
 0.2387434885779707]
#coe_sal_h = 20
U3 = [0.94415999, 2.2623793 , 3.5805986 , 4.8988179 , 6.21703721,
       7.53525651]
CORmean_glo3 = [0.9308738413364859,
 0.5296669816021244,
 0.4067455899818712,
 0.3332527940008383,
 0.3324732743237596,
 0.22192864762170092]
CORstd_glo3 = [0.868096977933892,
 0.2843639374584956,
 0.20681541077362697,
 0.19609015344721534,
 0.18197877247460617,
 0.08692174498511492]
Upr3 = [0.68881508, 2.05346073, 3.41810639, 4.78275204, 6.14739769,
       7.51204334]
Pr_glo3 = [0.10394829, 0.83221477, 0.93723849, 0.95744681, 0.83333333,
       0.66666667]
NEmean_glo3 = [0.34643605870020966,
 2.9758389261744966,
 3.4518828451882846,
 3.425531914893617,
 3.6666666666666665,
 3.1666666666666665]
UEmean_glo3 = [0.23947824196103829,
 0.2313647403282845,
 0.23224694579975455,
 0.2568409955289085,
 0.19621010226692073,
 0.2623559420343883]
UEstd_glo3 = [0.19893133690696568,
 0.20529402807214506,
 0.17727406946515473,
 0.222256166084367,
 0.09826383751204791,
 0.22823785565311747]

plt.figure(figsize=(12,9))
plt.subplot(2,2,1)
#plot COR - Uim
lines = [plt.errorbar(eval(f"U{i}/constant"), eval(f"CORmean_glo{i}"), yerr=eval(f"CORstd_glo{i}"), fmt='o', capsize=5) for i in range(1, 4)]
# line2 = plt.errorbar(Unsexp, CORexp_mean, yerr=CORexp_std, fmt='x', capsize=5, label='Experiment (Jiang et al., 2024)', color='k')
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$e$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.legend(lines, [r'$E_\mathrm{sal,cr}$=1.5$m_\mathrm{p}gd$', '$E_\mathrm{sal,cr}$=5$m_\mathrm{p}gd$', '$E_\mathrm{sal,cr}$=8$m_\mathrm{p}gd$'], fontsize=12)
plt.subplot(2,2,2)
#plot Pr - Uim 
lines = [plt.plot(eval(f"Upr{i}/constant"), eval(f"Pr_glo{i}"), 'o') for i in range(1, 4)]
# plt.plot(Unsexp, Prexp, 'x', color='k')
plt.ylim(0,1.05)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$P_\mathrm{r}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(2,2,3)
#plot \bar{NE} - Uim
lines = [plt.plot(eval(f"Upr{i}/constant"), eval(f"NEmean_glo{i}"), 'o') for i in range(1, 4)]
# plt.plot(Unsexp, Nsexp,'x',color='k')
# plt.ylim(0,1.3)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
#plot \bar{UE} - Uim
plt.subplot(2,2,4)
lines = [plt.errorbar(eval(f"Upr{i}/constant"), eval(f"UEmean_glo{i}/constant"), yerr=eval(f"UEstd_glo{i}/constant"), fmt='o', capsize=5) for i in range(1, 4)]
# plt.errorbar(Uvsexp, Usexp, yerr=Usexp_std, fmt='x', capsize=5, label='With wind (Jiang et al., 2024)', color='k')
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

#h_midair = 20d
Uh1 = [0.68864585, 1.7739315 , 2.85921715, 3.9445028 , 5.02978845,
       6.1150741 ]
CORmean_gloh1 = [1.0547278790842345,
 0.5481677085630939,
 0.4109509541016733,
 0.34675254055795446,
 0.2838405638898634,
 0.40569028488274467]
CORstd_gloh1 = [1.0736363148907826,
 0.3482407721454709,
 0.20844059484581792,
 0.2093232355311031,
 0.14314850890122416,
 0.15768403097944814]
Uprh1 = [0.58564852, 1.74396103, 2.90227355, 4.06058606, 5.21889858,
       6.37721109]
Pr_gloh1 = [0.14703411, 0.81958196, 0.89189189, 0.92207792, 0.83333333,
       0.75      ]
NEmean_gloh1 = [0.398924550495715,
 2.2442244224422443,
 2.7522522522522523,
 2.4025974025974026,
 2.611111111111111,
 2.375]
UEmean_gloh1 = [0.23851625771519683,
 0.22921688292778417,
 0.23872830137699289,
 0.2453553377357668,
 0.22650832801000953,
 0.1857460745766891]
UEstd_gloh1 = [0.20071441513525679,
 0.19525404233189467,
 0.20854571191932267,
 0.18614796483644505,
 0.15022420796324756,
 0.0876945059375047]

#h_midair = 30d
Uh2, CORmean_gloh2, CORstd_gloh2, Uprh2, Pr_gloh2, NEmean_gloh2, UEmean_gloh2, UEstd_gloh2 = U2, CORmean_glo2, CORstd_glo2, Upr2, Pr_glo2, NEmean_glo2, UEmean_glo2, UEstd_glo2

#h_midair = 40d
Uh3 = [0.81669995, 2.15809381, 3.49948767, 4.84088152, 6.18227538,
       7.52366923]
CORmean_gloh3 = [0.9773182476830321,
 0.5488071426664187,
 0.4221684563749771,
 0.36505421341132704,
 0.3577853135259619,
 0.33306242434606653]
CORstd_gloh3 = [0.9852736428934576,
 0.3229656496782703,
 0.2256352842578587,
 0.21941711709999984,
 0.23673641372709617,
 0.23547224277415477]
Uprh3 = [0.68881508, 2.05346073, 3.41810639, 4.78275204, 6.14739769,
       7.51204334]
Pr_gloh3 = [0.17857143, 0.86855941, 0.95283019, 0.97402597, 0.88888889,
       0.71428571]
NEmean_gloh3 = [0.379724535554132,
 2.017875920084122,
 2.3238993710691824,
 2.207792207792208,
 2.074074074074074,
 2.5714285714285716]
UEmean_gloh3 = [0.2388203896244559,
 0.2296019805568729,
 0.23270742620373636,
 0.2569056653391436,
 0.20306025159247104,
 0.28668546261114636]
UEstd_gloh3 = [0.20269856016812368,
 0.1994840052013951,
 0.18225639500472068,
 0.21078600993717508,
 0.10295320678819797,
 0.22903409896613508]

plt.figure(figsize=(12,9))
plt.subplot(2,2,1)
#plot COR - Uim
lines = [plt.errorbar(eval(f"Uh{i}/constant"), eval(f"CORmean_gloh{i}"), yerr=eval(f"CORstd_gloh{i}"), fmt='o', capsize=5) for i in range(1, 4)]
# line2 = plt.errorbar(Unsexp, CORexp_mean, yerr=CORexp_std, fmt='x', capsize=5, label='Experiment (Jiang et al., 2024)', color='k')
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$e$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.legend(lines, [r'$Z_\mathrm{im,cr}$=20$d$', '$Z_\mathrm{im,cr}$=30$d$', '$Z_\mathrm{im,cr}$=40$d$'], fontsize=12)
plt.subplot(2,2,2)
#plot Pr - Uim 
lines = [plt.plot(eval(f"Uprh{i}/constant"), eval(f"Pr_gloh{i}"), 'o') for i in range(1, 4)]
# plt.plot(Unsexp, Prexp, 'x', color='k')
plt.ylim(0,1.05)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$P_\mathrm{r}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(2,2,3)
#plot \bar{NE} - Uim
lines = [plt.plot(eval(f"Uh{i}/constant"), eval(f"NEmean_gloh{i}"), 'o') for i in range(1, 4)]
# plt.plot(Unsexp, Nsexp,'x',color='k')
# plt.ylim(0,1.3)
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
#plot \bar{UE} - Uim
plt.subplot(2,2,4)
lines = [plt.errorbar(eval(f"Uh{i}/constant"), eval(f"UEmean_gloh{i}/constant"), yerr=eval(f"UEstd_gloh{i}/constant"), fmt='o', capsize=5) for i in range(1, 4)]
# plt.errorbar(Uvsexp, Usexp, yerr=Usexp_std, fmt='x', capsize=5, label='With wind (Jiang et al., 2024)', color='k')
plt.xlabel(r'$U_{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()