# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:37:34 2026

@author: WangX3
"""

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

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module_name, class_name):
        if module_name.startswith("numpy._core"):
            module_name = module_name.replace(
                "numpy._core", "numpy.core", 1
            )
        return super().find_class(module_name, class_name)

# Load the data from the saved file
data_dict = {}
moisture_levels = [0, 1, 5, 10]
shield_numbers = range(2, 7)  # S002–S006

data_index = 1

for moisture in moisture_levels:
    for shield in shield_numbers:
        filename = (
            f"input_pkl/Review_P2/"
            f"data4dM{moisture}S{shield:03d}.pkl"
        )

        with open(filename, "rb") as f:
            data_dict[f"data{data_index}"] = NumpyCompatUnpickler(f).load()

        data_index += 1
   
#basic parameters
dt = 0.01
D = 0.00025
coe_h = 13.5 #critial height for a mobile particle to reach
coe_sal_h = 13.5 # treat all hopping rains by one species
N_inter = 100 #number of output timesteps for erosion and deposition properties
dt_output = (5 - 0.01) / N_inter
t_mid = 0.01 + (np.arange(N_inter) + 0.5) * dt_output
Omega = [0, 1, 5, 10]
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
for i in range(len(Omega) * len(shield_numbers)):
    filename = f"data{i+1}"
    data = data_dict[filename]
    num_p = 5147
    id_start = 0
    ParticleID=np.linspace(id_start,num_p-1,num_p-id_start)
    ParticleID_int = ParticleID.astype(int)
    #cal erosion and deposition properties for each Omega
    EDindices[i], ME[i], MD[i], VExz_mean_t[i], VDxz_mean_t[i], D_vector_t[i], E_vector_t[i], _=module.store_particle_id_data(data,ParticleID_int,coe_h,dt,N_inter,D)
    #cal rebound properties for each Omega
    ParticleID_sal=np.linspace(id_start,num_p-1,num_p-id_start)
    ParticleID_salint = ParticleID_sal.astype(int)
    X[i] = np.array([[time_step['Position'][i][0] for i in ParticleID_salint] for time_step in data])
    Z[i] = np.array([[time_step['Position'][i][2] for i in ParticleID_salint] for time_step in data])
    Par[i], VZ[i], exz_mean_t[i], ez_mean_t[i], VIM_mean_t[i], ThetaIM_mean_t[i], RIM[i], exz_vector_t[i], IM_vector_t[i]=module.store_sal_id_data(data,ParticleID_salint, coe_sal_h, dt, N_inter, D)
    
exz_all,Vim_all,VD_all,ThetaD_all,Theta_all,Thetare_all,impact_list,impact_deposition_list,ejection_list = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
Vre_all, Vsal_all = defaultdict(list), defaultdict(list)
Vrep_all = defaultdict(list)
ThetaE_all, UE_all = defaultdict(list), defaultdict(list)
N_range = np.full(len(Omega) * len(shield_numbers), 0).astype(int)
Vimz_all, zim_all = defaultdict(list), defaultdict(list)
for i in range (len(Omega) * len(shield_numbers)):
    exz_all[i] = [value for sublist in exz_vector_t[i][N_range[i]:] for value in sublist]
    Vim_all[i] = [value[0] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Vimz_all[i] = [value[14] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    zim_all[i] = [value[15] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
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
for i in range(len(Omega) * len(shield_numbers)):
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
n_shields = len(shield_numbers)

for i in range(len(Omega)):
    # Data were loaded as M0S002–M0S006, then M1S002–M1S006, etc.
    selected_indices = range(
        i * n_shields,
        (i + 1) * n_shields)
    Vim_all_Omega[i] = np.concatenate(
        [Vim_all[j] for j in selected_indices]
    ).tolist()
    Vre_all_Omega[i] = np.concatenate(
        [Vre_all[j] for j in selected_indices]
    ).tolist()
    Vsal_all_Omega[i] = np.concatenate(
        [Vsal_all[j] for j in selected_indices]
    ).tolist()
    exz_all_Omega[i] = np.concatenate(
        [exz_all[j] for j in selected_indices]
    ).tolist()
    Thetaim_all_Omega[i] = np.concatenate(
        [Theta_all[j] for j in selected_indices]
    ).tolist()
    Thetare_all_Omega[i] = np.concatenate(
        [Thetare_all[j] for j in selected_indices]
    ).tolist()
    VD_all_Omega[i] = np.concatenate(
        [VD_all[j] for j in selected_indices]
    ).tolist()
    ThetaD_all_Omega[i] = np.concatenate(
        [ThetaD_all[j] for j in selected_indices]
    ).tolist()
    ThetaE_all_Omega[i] = np.concatenate(
        [ThetaE_all[j] for j in selected_indices]
    ).tolist()
    UE_all_Omega[i] = np.concatenate(
        [UE_all[j] for j in selected_indices]
    ).tolist()
    matched_Vim_Omega[i] = np.concatenate(
        [matched_Vim[j] for j in selected_indices]
    ).tolist()
    matched_Thetaim_Omega[i] = np.concatenate(
        [matched_thetaim[j] for j in selected_indices]
    ).tolist()
    matched_NE_Omega[i] = np.concatenate(
        [matched_NE[j] for j in selected_indices]
    ).tolist()
    matched_UE_Omega[i] = np.concatenate(
        [matched_UE[j] for j in selected_indices]
    ).tolist()
    matched_EE_Omega[i] = np.concatenate(
        [matched_EE[j] for j in selected_indices]
    ).tolist()
    matched_thetaE_Omega[i] = np.concatenate(
        [matched_thetaE[j] for j in selected_indices]
    ).tolist()

CORmean_Omega,CORstd_Omega,CORstderr_Omega,N_COR_Omega,Uimplot_Omega, Thetaremean_Omega, Thetarestd_Omega, Thetarestderr_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
Pr_Omega,Uplot_Omega, N_PrUre, Uim_meaninbin, UD_meaninbin, N_Princ = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
NE_mean_Omega, UE_mean_Omega, UE_std_Omega, UE_stderr_Omega, Uplot_NE_Omega, N_Einbin, N_Einc = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
ThetaE_mean_Omega, ThetaE_std_Omega, ThetaE_stderr_Omega = defaultdict(list), defaultdict(list), defaultdict(list)
mom_ratio_mean, mom_ratio_std = defaultdict(list), defaultdict(list)
for i in range (len(Omega)): #loop over Omega 0-20%
    CORmean_Omega[i], CORstd_Omega[i], CORstderr_Omega[i], N_COR_Omega[i], Thetaremean_Omega[i], Thetarestd_Omega[i], Thetarestderr_Omega[i], Uimplot_Omega[i] = module.BinUimCOR_equalbinsize(Vim_all_Omega[i],Vre_all_Omega[i],Thetare_all_Omega[i],Vim_bin)
    Pr_Omega[i], Uplot_Omega[i],N_PrUre[i], Uim_meaninbin[i], UD_meaninbin[i], N_Princ[i] = module.BinUimUd_equalbinsize(Vim_all_Omega[i],VD_all_Omega[i],Vimde_bin)   
    NE_mean_Omega[i], UE_mean_Omega[i], UE_std_Omega[i], UE_stderr_Omega[i], ThetaE_mean_Omega[i], ThetaE_std_Omega[i], ThetaE_stderr_Omega[i], N_Einbin[i], Uplot_NE_Omega[i], N_Einc[i], mom_ratio_mean[i], mom_ratio_std[i]= module.get_ejection_ratios_equalbinsize(matched_Vim_Omega[i], VD_all_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], matched_thetaE_Omega[i], Vimde_bin)#matched_EE_Omega[i]

# #6 terms with Uim/Uinc
# COR - Uim
#global figure
plt.figure(figsize=(12,13.5))
plt.subplot(3,2,1)
for i in range(len(Omega)):
    plt.errorbar(Uimplot_Omega[i]/constant, CORmean_Omega[i], yerr=CORstderr_Omega[i], fmt='o', capsize=5, color=colors[i], label=f"$\\Omega$={Omega[i]}%")
plt.text(0.02, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.xlim(0,250)
plt.ylim(0, 1.4)
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$e$ [-]', fontsize=14) 
plt.legend(fontsize=12)
plt.subplot(3,2,2)
for i in range(len(Omega)):
    plt.errorbar(Uimplot_Omega[i]/constant, Thetaremean_Omega[i], yerr=Thetarestderr_Omega[i], fmt='o', capsize=5, color=colors[i])
plt.text(0.02, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.xlim(0,250)
plt.ylim(bottom=0)
plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{re}$ [$^\circ$]', fontsize=14)
plt.subplot(3,2,3)
for i in range(len(Omega)):
    plt.scatter(Uplot_Omega[i]/constant, Pr_Omega[i], s=np.sqrt(N_Princ[i])*3, color=colors[i])
# plt.plot(Uprdrybin/constant, Pr_drybin, 'o', color='gray', ms=5, label='Equal-count binning')
plt.xlim(0,250)
plt.ylim(0, 1.1)
plt.text(0.02, 0.92, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\mathrm{Pr}$ [-]', fontsize=14)
plt.subplot(3,2,4)
for i in range(len(Omega)):
    plt.scatter(Uplot_NE_Omega[i]/constant, NE_mean_Omega[i], s=np.sqrt(N_Einc[i])*3, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlim(0,250)
plt.ylim(0, 5.75)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$N_\mathrm{E}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(3,2,5)
for i in range(len(Omega)):
    plt.errorbar(Uplot_NE_Omega[i]/constant, UE_mean_Omega[i]/constant, yerr=UE_stderr_Omega[i]/constant, fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.xlim(0,250)
# plt.ylim(0, 14.5)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
plt.text(0.02, 0.92, '(e)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.subplot(3,2,6)
for i in range(len(Omega)):
    plt.errorbar(Uplot_NE_Omega[i]/constant, ThetaE_mean_Omega[i], yerr=ThetaE_stderr_Omega[i], fmt='o', capsize=5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
plt.axhline(24, color='k', linestyle='--')
plt.xlim(0,250)
plt.ylim(0,40)
plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
plt.ylabel(r'$\theta_\mathrm{E}$ [$^\circ$]', fontsize=14)
plt.text(0.03, 0.93, '(f)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.tight_layout()

# ============================================================
# Save binned splash statistics for domain-width comparison
# ============================================================
from pathlib import Path
import pandas as pd

domain_width_d = 4
output_dir = Path("output_Review4d")
output_dir.mkdir(parents=True, exist_ok=True)

summary_rows = []

def add_statistics(
    quantity,
    moisture,
    speed,
    mean,
    std=None,
    sample_count=None
):
    speed = np.asarray(speed, dtype=float).ravel()
    mean = np.asarray(mean, dtype=float).ravel()

    if std is None:
        std = np.full(mean.size, np.nan)
    else:
        std = np.asarray(std, dtype=float).ravel()

    if sample_count is None:
        sample_count = np.full(mean.size, np.nan)
    else:
        sample_count = np.asarray(
            sample_count, dtype=float
        ).ravel()

    if not (
        speed.size == mean.size
        == std.size
        == sample_count.size
    ):
        raise ValueError(
            f"Inconsistent array lengths for {quantity}, "
            f"Omega={moisture}%: "
            f"speed={speed.size}, mean={mean.size}, "
            f"std={std.size}, N={sample_count.size}"
        )

    for u, value, stdvalue, n_value in zip(
        speed, mean, std, sample_count
    ):
        if np.isfinite(u) and np.isfinite(value):
            summary_rows.append({
                "width_d": domain_width_d,
                "moisture_pct": moisture,
                "quantity": quantity,
                "speed_normalized": u,
                "mean": value,
                "std": stdvalue,
                "sample_count": n_value
            })


for i, moisture in enumerate(Omega):
    # 1. Restitution coefficient
    add_statistics(
        quantity="e",
        moisture=moisture,
        speed=np.asarray(Uimplot_Omega[i]) / constant,
        mean=CORmean_Omega[i],
        std=CORstd_Omega[i],
        sample_count=N_COR_Omega[i]
    )

    # 2. Rebound angle
    add_statistics(
        quantity="theta_re",
        moisture=moisture,
        speed=np.asarray(Uimplot_Omega[i]) / constant,
        mean=Thetaremean_Omega[i],
        std=Thetarestd_Omega[i],
        sample_count=N_COR_Omega[i]
    )

    # 3. Rebound probability
    pr = np.asarray(Pr_Omega[i], dtype=float)
    n_pr = np.asarray(N_Princ[i], dtype=float)

    # # Binomial standard error
    # pr_stderr = np.full(pr.shape, np.nan)
    # valid = n_pr > 0
    # pr_stderr[valid] = np.sqrt(
    #     pr[valid] * (1.0 - pr[valid]) / n_pr[valid]
    # )

    add_statistics(
        quantity="Pr",
        moisture=moisture,
        speed=np.asarray(Uplot_Omega[i]) / constant,
        mean=pr,
        std=None,
        sample_count=n_pr
    )

    # 4. Mean ejection number
    # Replace stderr=None with NE_stderr_Omega[i] if it is
    # calculated elsewhere in the complete script.
    add_statistics(
        quantity="N_E",
        moisture=moisture,
        speed=np.asarray(Uplot_NE_Omega[i]) / constant,
        mean=NE_mean_Omega[i],
        std=None,
        sample_count=N_Einc[i]
    )

    # 5. Mean normalized ejection speed
    add_statistics(
        quantity="U_E",
        moisture=moisture,
        speed=np.asarray(Uplot_NE_Omega[i]) / constant,
        mean=np.asarray(UE_mean_Omega[i]) / constant,
        std=np.asarray(UE_std_Omega[i]) / constant,
        sample_count=N_Einbin[i]
    )

    # 6. Mean ejection angle
    add_statistics(
        quantity="theta_E",
        moisture=moisture,
        speed=np.asarray(Uplot_NE_Omega[i]) / constant,
        mean=ThetaE_mean_Omega[i],
        std=ThetaE_std_Omega[i],
        sample_count=N_Einbin[i]
    )


summary_df = pd.DataFrame(summary_rows)

output_file = (
    output_dir
    / f"splash_statistics_width_{domain_width_d}d.csv"
)

summary_df.to_csv(output_file, index=False)

print(f"Saved {len(summary_df)} rows to {output_file}")
print(summary_df.groupby(
    ["quantity", "moisture_pct"]
).size())

# # validation dry
# #empirical data
# Hcr = 1.5
# UIM_UE_prin = np.linspace(0,190, 100)
# thetaim_prin = np.arcsin(39.21/(UIM_UE_prin+105.73))
# NE_prin = (-0.001*Hcr + 0.012)*UIM_UE_prin
# VE_prin = (0.0538*Hcr + 1.0966)*np.sqrt(UIM_UE_prin)
# COR_emp = 0.7469*np.exp(0.1374*Hcr)*(UIM_UE_prin)**(-0.0741*np.exp(0.214*Hcr))#Jiang et al. (2024) JGR
# Pr_emp = 0.9945*Hcr**(-0.0166)*(1-np.exp(-0.1992*Hcr**(-0.8686)*UIM_UE_prin))
# Thetare_emp = 24.56*np.ones(len(UIM_UE_prin))
# ThetaE_emp = 0.2*UIM_UE_prin-1.4714*Hcr + 24.2
# #anderson
# Pr_and = 0.95*(1-np.exp(-2*UIM_UE_prin*constant))
# # #Chen 2019
# # COR_Chen = 0.62 + 0.0084*Uimplot - 0.63*np.sin(Theta_mean_all/180*np.pi)
# NE_Chen = np.exp(-0.2 + 1.35*np.log(UIM_UE_prin*constant)-0.01*thetaim_prin/180*np.pi)
# VE_Chen = np.exp(-1.48 + 0.082*UIM_UE_prin*constant-0.003*thetaim_prin/180*np.pi)/constant
# #beladijne et al 2007
# VE_Bel = 1.18*UIM_UE_prin**0.25
# thetare_Bel = np.degrees(np.arcsin((0.3-0.15*np.sin(thetaim_prin))/(0.87-0.72*np.sin(thetaim_prin))))
# thetaE_Bel = np.degrees(np.pi/2 + 0.1*(thetaim_prin-np.pi/2))
# #exp data Jiang 2024; Hcr = 1.5D
# Unsexp = [15, 35, 55, 75, 95, 125]
# Nsexp = [0.05, 0.25, 0.65, 0.625, 1.1, 1.25]
# CORexp_mean = [0.7, 0.7, 0.6, 0.65, 0.5, 0.6]
# CORexp_std = [0.4, 0.3, 0.2, 0.15, 0.25, 0.3]
# thetareexp = [25, 26, 24, 24, 34, 37]
# thetareexp_std = [15, 20, 30, 28, 18, 27]
# Prexp = [0.85, 0.94, 0.98, 0.99, 1.0, 1.0]
# Uvsexp = [25, 45, 70, 95, 125]
# Usexp = [5, 7, 8, 12.5, 6]
# Usexp_std = [3, 4.5, 3.4, 5, 1]
# thetaEexp = [28, 33, 35, 28, 20]
# thetaEexp_std = [24, 24, 26, 15, 8]
# error_hor = np.full(6,4.5)
# #exp data Selmani 2024
# UNE_Selmani = [20, 25, 55, 72, 100]
# NE_Selmani = [0.5, 1, 3, 5, 10]

# # plt.close("all")
# plt.figure(figsize=(12,13.5))
# plt.subplot(3,2,1)
# line1 = plt.errorbar(Uimplot_Omega[0]/constant, CORmean_Omega[0], yerr=CORstd_Omega[0], fmt='o', capsize=5, label='This study', color='#3776ab') 
# line2 = plt.errorbar(Unsexp, CORexp_mean, yerr=CORexp_std, fmt='x', capsize=5, label='Jiang et al. (2024)', color='k')
# line3 = plt.plot(UIM_UE_prin, COR_emp, 'k-', label='Jiang et al. (2024)')
# plt.xlim(0,190)
# plt.ylim(0,2.25)
# plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$e$ [-]', fontsize=14)
# plt.legend([line2[0],line3[0], line1],['Jiang et al. (2024)','Jiang et al. (2024)', 'This study'], fontsize=12)
# plt.text(0.02, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(3,2,2)
# line1 = plt.errorbar(Uimplot_Omega[0]/constant, Thetaremean_Omega[0], yerr=Thetarestd_Omega[0], fmt='o', capsize=5, label='This study', color='#3776ab')
# line2 = plt.errorbar(Unsexp, thetareexp, yerr=thetareexp_std, fmt='x', capsize=5, label='Jiang et al. (2024)', color='k')
# line3 = plt.plot(UIM_UE_prin, Thetare_emp, 'k-')
# line4 = plt.plot(UIM_UE_prin, thetare_Bel, 'k:', label='Beladjine et al. (2007)')
# plt.ylim(-8,120)
# plt.xlim(0,190)
# plt.legend([line2[0],line3[0], line4[0], line1[0]],['Jiang et al. (2024)','Jiang et al. (2024)', 'Beladjine et al. (2007)', 'This study'], loc='upper left', bbox_to_anchor=(0.08, 0.99), fontsize=12)
# plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\theta_\mathrm{re}$ [$^\circ$]', fontsize=14)
# plt.text(0.02, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(3,2,3)
# #plot Pr - Uim 
# plt.scatter(Uplot_Omega[0]/constant, Pr_Omega[0], s=np.sqrt(N_Princ[0])*3, label='This study', color='#3776ab')
# plt.plot(UIM_UE_prin, Pr_emp, label='Jiang et al. (2024)',color='k')
# plt.plot(Unsexp, Prexp, 'x', label='Jiang et al. (2024)', color='k')
# plt.plot(UIM_UE_prin, Pr_and, 'k-.', label='Anderson & Haff (1991)')
# plt.ylim(0,1.05)
# plt.xlim(0,190)
# plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\mathrm{Pr}$ [-]', fontsize=14)
# plt.legend(fontsize=12)
# plt.text(0.02, 0.92, '(c)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(3,2,4)
# #plot \bar{NE} - Uim
# plt.scatter(Uplot_NE_Omega[0]/constant, NE_mean_Omega[0], s=np.sqrt(N_Einc[0])*3, label='This study', color='#3776ab')
# plt.plot(UIM_UE_prin, NE_prin, 'k-', label='Jiang et al. (2024)')
# plt.plot(Unsexp, Nsexp,'x', label='Jiang et al. (2024)',color='k')
# plt.plot(UNE_Selmani, NE_Selmani, 'dk', label='Selmani et al. (2024)')
# plt.plot(UIM_UE_prin, NE_Chen, 'k-.', label='Chen et al. (2019)')
# plt.ylim(0,20)
# plt.xlim(0,190)
# plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$N_\mathrm{E}$ [-]', fontsize=14)
# plt.legend(loc='upper left', bbox_to_anchor=(0.08, 0.99), fontsize=12)
# plt.text(0.02, 0.92, '(d)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(3,2,5)
# #plot \bar{UE} - Uim
# line1 = plt.errorbar(Uplot_NE_Omega[0]/constant, UE_mean_Omega[0]/constant, yerr=UE_std_Omega[0]/constant, fmt='o', capsize=5, label='This study', color='#3776ab')
# line2 = plt.errorbar(Uvsexp, Usexp, yerr=Usexp_std, fmt='x', capsize=5, label='Jiang et al. (2024)', color='k')
# line3 = plt.plot(UIM_UE_prin, VE_prin, 'k-')
# line4 = plt.plot(UIM_UE_prin, VE_Chen, 'k-.')
# line5 = plt.plot(UIM_UE_prin, VE_Bel, 'k:', label='Beladijne et al. (2007)')
# plt.ylim(0,32.5)
# plt.xlim(0,190)
# plt.legend([line2[0],line3[0], line4[0], line5[0], line1[0]],['Jiang et al. (2024)','Jiang et al. (2024)', 'Chen et al. (2019)', 'Beladjine et al. (2007)', 'This study'], loc='upper left', bbox_to_anchor=(0.08, 0.99), fontsize=12)
# plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$U_\mathrm{E}/\sqrt{gd}$ [-]', fontsize=14)
# plt.text(0.02, 0.92, '(e)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.subplot(3,2,6)
# line1 = plt.errorbar(Uplot_NE_Omega[0]/constant, ThetaE_mean_Omega[0], yerr=ThetaE_std_Omega[0], fmt='o', capsize=5, label='This study', color='#3776ab')
# line2 = plt.errorbar(Uvsexp, thetaEexp, yerr=thetaEexp_std, fmt='x', capsize=5, label='Jiang et al. (2024)', color='k')
# line3 = plt.plot(UIM_UE_prin, ThetaE_emp, 'k-')
# line4 = plt.plot(UIM_UE_prin, thetaE_Bel, 'k:', label='Beladjine et al. (2007)')
# plt.ylim(-10,140)
# plt.xlim(0,190)
# plt.legend([line2[0],line3[0], line4[0], line1[0]],['Jiang et al. (2024)','Jiang et al. (2024)', 'Beladjine et al. (2007)', 'This study'], loc='upper left', bbox_to_anchor=(0.08, 0.99), fontsize=12)
# plt.xlabel(r'$U_\mathrm{inc}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\theta_\mathrm{E}$ [$^\circ$]', fontsize=14)
# plt.text(0.02, 0.92, '(f)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()

# # #trajectory and vertical velocity
# id_p = 2700
# plt.figure(figsize=(10,8))
# plt.subplot(2,1,1)
# plt.plot(t_ver, Z[20][:,id_p]/D, linestyle='-', marker='.', color='k', markersize=3, linewidth=1)
# plt.plot(t_ver[EDindices[20][id_p][0]], Z[20][EDindices[20][id_p][0],id_p]/D, 'ob', label='Ejections', markerfacecolor='none')
# plt.plot(t_ver[EDindices[20][id_p][1]], Z[20][EDindices[20][id_p][1],id_p]/D, 'vb', label='Depositions', markerfacecolor='none')
# legend_added = False
# for t in t_ver[EDindices[20][id_p][0]]:
#     if not legend_added:
#         plt.axvline(x=t, color='k', linestyle='--', linewidth=1, label='Mobile intervals (start)')
#         legend_added = True  # Set flag to True after first legend entry
#     else:
#         plt.axvline(x=t, color='k', linestyle='--', linewidth=1)
# legend_added = False
# for t in t_ver[EDindices[20][id_p][1]]:
#     if not legend_added:
#         plt.axvline(x=t, color='k', linestyle=':', linewidth=1, label='Mobile intervals (end)')
#         legend_added = True  # Set flag to True after first legend entry
#     else:
#         plt.axvline(x=t, color='k', linestyle=':', linewidth=1)
# plt.plot(t_ver[Par[20][id_p][2][:,0]], Z[20][Par[20][id_p][2][:,0],id_p]/D, 'Dr', label='Impacts', markerfacecolor='none')
# plt.plot(t_ver[Par[20][id_p][2][:,1]], Z[20][Par[20][id_p][2][:,1],id_p]/D, 'sr', label='Rebounds', markerfacecolor='none')
# index_example2 = np.where(t_ver == 1.54)[0][0]
# plt.axvline(t_ver[139],color='c', linestyle='-', label='An example cross-zero interval')
# plt.axvline(t_ver[index_example2],color='c', linestyle='-')
# plt.xlabel(r'$t$ [s]', fontsize=14)
# plt.ylabel(r'$Z_\mathrm{p}/d$ [-]',fontsize=14)
# plt.xlim(0,5)
# plt.text(0.02, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.legend(fontsize=10)
# plt.subplot(2,1,2)
# plt.plot(t_ver, VZ[20][id_p], linestyle='-', marker='.', color='k', markersize=3, linewidth=1)
# plt.plot(t_ver[Par[20][id_p][2][:, 0]], VZ[20][id_p][Par[20][id_p][2][:, 0]], 'Dr', label='Impacts',markerfacecolor='none')
# plt.plot(t_ver[Par[20][id_p][2][:, 1]], VZ[20][id_p][Par[20][id_p][2][:, 1]], 'sr', label='Rebounds',markerfacecolor='none')
# index_example2 = np.where(t_ver == 1.54)[0][0]
# plt.axvline(t_ver[139],color='c', linestyle='-')
# plt.axvline(t_ver[index_example2],color='c', linestyle='-')
# legend_added = False
# for t in t_ver[EDindices[20][id_p][0]]:
#     if not legend_added:
#         plt.axvline(x=t, color='k', linestyle='--', linewidth=1, label='Mobile intervals (start)')
#         legend_added = True  # Set flag to True after first legend entry
#     else:
#         plt.axvline(x=t, color='k', linestyle='--', linewidth=1)
# legend_added = False
# for t in t_ver[EDindices[20][id_p][1]]:
#     if not legend_added:
#         plt.axvline(x=t, color='k', linestyle=':', linewidth=1, label='Mobile intervals (end)')
#         legend_added = True  # Set flag to True after first legend entry
#     else:
#         plt.axvline(x=t, color='k', linestyle=':', linewidth=1)  
# plt.axhline(0, color='grey', linestyle='-', linewidth=0.5)
# plt.xlabel(r'$t$ [s]',fontsize=14)
# plt.ylabel(r'$U_\mathrm{z}$ [m/s]',fontsize=14)
# plt.xlim(0,5)
# plt.text(0.02, 0.92, '(b)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.show()