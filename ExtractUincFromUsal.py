# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 13:36:04 2025

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
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from typing import Dict, Any

# Load the data from the saved file
data_dict = {}  # Dictionary to store data
for i in range(2,7): #Theta = 0.02-0.06
    for j in range(5):
        filename = f"input_pkl/data{i}_{j}.pkl"  # Generate filename
        with open(filename, 'rb') as f:
            data_dict[f"data{5*(i-2)+(j+1)}"] = pickle.load(f)  # Store in dictionary  
   
#basic parameters
dt = 0.01
g = 9.81
D = 0.00025
coe_h = 13.5 #critial height for a mobile particle to reach
coe_sal_h = 17
N_inter = 100 #number of output timesteps for erosion and deposition properties
t_inter = np.linspace(0,5,N_inter+1)
Omega = [0, 1, 5, 10, 20]
Omega_real = np.array(Omega)*0.01
# Generate smooth curve for plotting
Omega_smooth = np.linspace(min(Omega_real), max(Omega_real), 100)
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
 
    
# load CG Usal-t
# ---- global containers (flat list + nested dict for convenience) ----
U_DPM_FLAT: List[np.ndarray] = []             # 25 sublists in the requested order
U_DPM_TABLE: Dict[str, Dict[int, np.ndarray]] = {}  # e.g. U_DPM_TABLE['Dry'][2] -> array

MOISTURE = ["Dry", "M1", "M5", "M10", "M20"]
SHIELDS  = [2, 3, 4, 5, 6]                    # corresponds to 002..006

def _read_third_column(path: str, col: int = 2) -> np.ndarray:
    """
    Read the 3rd column (0-based index=2) from a text/csv file with unknown delimiter.
    Lines starting with '#' are treated as comments. No header is assumed.
    """
    df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
    if df.shape[1] <= col:
        raise ValueError(f"{os.path.basename(path)} has only {df.shape[1]} columns; need column {col}.")
    return df.iloc[:, col].to_numpy()

def load_shields_series(base_dir: str = ".", extensions: Tuple[str, ...] = ("", ".txt", ".csv", ".dat")) -> None:
    """
    Populate U_DPM_FLAT (25 sublists) and U_DPM_TABLE[moisture][S] with U_dpm=data[:,2]
    from files named: Shields002Dry ... Shields006Dry, Shields002M1 ... Shields006M20.

    Search order per file tries the given extensions in order.
    """
    global U_DPM_FLAT, U_DPM_TABLE
    U_DPM_FLAT.clear()
    U_DPM_TABLE.clear()

    for moist in MOISTURE:
        U_DPM_TABLE[moist] = {}
        for S in SHIELDS:
            # e.g. Shields002Dry(.txt/.csv/...)
            stem = f"Shields{S:03d}{moist}"
            path = None
            for ext in extensions:
                candidate = os.path.join(base_dir, stem + ext)
                if os.path.isfile(candidate):
                    path = candidate
                    break
            if path is None:
                raise FileNotFoundError(f"Could not find file for {stem} with any of {extensions}")

            U_dpm = _read_third_column(path, col=2)
            U_DPM_TABLE[moist][S] = U_dpm
            U_DPM_FLAT.append(U_dpm)  # order: Dry S2..S6, M1 S2..S6, ..., M20 S2..S6

load_shields_series(base_dir="CGdata/", extensions=(".txt",))
# - U_DPM_FLAT is a list of 25 arrays in the required order
# - U_DPM_TABLE["M5"][4] is the array from Shields004M5

exz_all,Vim_all,VD_all,ThetaD_all,Theta_all,Thetare_all,impact_list,impact_deposition_list,ejection_list = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
Vre_all, Vsal_all = defaultdict(list), defaultdict(list)
Tsal_all, Tlast_all = defaultdict(list), defaultdict(list)
Vrep_all = defaultdict(list)
ThetaE_all, UE_all = defaultdict(list), defaultdict(list)
mp_sal_all, mp_rep_all = defaultdict(list), defaultdict(list)
IDim_all, IDD_all = defaultdict(list) , defaultdict(list)
N_range = np.full(25, 0).astype(int)
for i in range (25):
    exz_all[i] = [value for sublist in exz_vector_t[i][N_range[i]:] for value in sublist]
    Vim_all[i] = [value[0] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Vre_all[i] = [value[10] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Vsal_all[i] = [value[11] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Tsal_all[i] = [value[12] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    mp_sal_all[i] = [value[13] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    VD_all[i] = [value[0] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    Vrep_all[i] = [value[5] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    Tlast_all[i] = [value[6] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    mp_rep_all[i] = [value[8] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    ThetaD_all[i] = [value[1] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    IDD_all[i] = [value[2] for sublist in D_vector_t[i][N_range[i]:] for value in sublist]
    Theta_all[i] = [value[7] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Thetare_all[i] = [value[8] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    IDim_all[i] = [value[1] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    IDre = [value[2] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    xim = [value[3] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    xre = [value[4] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    xcol = [value[5] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Pim = [value[6] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    impact_list[i] = [IDim_all[i], IDre, xim, xre, Vim_all[i], xcol, Pim, Theta_all[i]]
    IDE = [value[1] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    xE = [value[2] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    PE = [value[3] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    ThetaE_all[i] = [value[6] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    mE = [value[7] for sublist in E_vector_t[i][N_range[i]:] for value in sublist] #kinetic energy
    UE_all[i] = [value[0] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    ejection_list[i] = [IDE, xE, UE_all[i], PE, mE, ThetaE_all[i]]
# time stamps when impacts and depositions occur
tim_all = {k: [0.01 * x for x in lst] for k, lst in IDim_all.items()}
tD_all = {k: [0.01 * x for x in lst] for k, lst in IDD_all.items()}

Vim_all_values = [value for sublist in Vim_all.values() for value in sublist]
Vim_bin = np.linspace(0, max(Vim_all_values)+1, 10)
VD_all_values = [value for sublist in VD_all.values() for value in sublist]
Vde_bin = np.linspace(0, max(VD_all_values)+1, 15)
Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
Vimde_bin = np.linspace(0, max(Vimde_all_values)+1, 10)
Vre_all_values = [value for sublist in Vre_all.values() for value in sublist]
Vre_bin = np.linspace(0, max(Vre_all_values)+1, 10)
VE_all_values = [value for sublist in UE_all.values() for value in sublist]
VE_bin = np.linspace(0, max(VE_all_values)+1, 10)
# Vsal_all_values = [value for sublist in Vsal_all.values() for value in sublist] + [value for sublist in Vrep_all.values() for value in sublist]
# Vsal_bin = np.linspace(min(Vsal_all_values), max(Vsal_all_values), 10)
Thetaimde_all_values = [value for sublist in Theta_all.values() for value in sublist] + [value for sublist in ThetaD_all.values() for value in sublist]
Thetaimde_bin = np.linspace(min(Thetaimde_all_values), max(Thetaimde_all_values), 10)
Thetaim_all_values = [value for sublist in Theta_all.values() for value in sublist]
Thetaim_bin = np.linspace(min(Thetaim_all_values), max(Thetaim_all_values), 10)
matched_Vim, matched_thetaim, matched_NE, matched_UE, matched_mE, matched_thetaE = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
for i in range (25):
    impact_ejection_list=module.match_ejection_to_impact(impact_list[i], ejection_list[i], dt)
    # impact_ejection_list=module.match_ejection_to_impactanddeposition(impact_deposition_list[i], ejection_list[i])
    matched_Vim[i] = [element for element in impact_ejection_list[0]]
    matched_thetaim[i] = [element for element in impact_ejection_list[1]]
    matched_NE[i] = [element for element in impact_ejection_list[2]]
    matched_UE[i] = [element for element in impact_ejection_list[3]]
    matched_mE[i] = [element for element in impact_ejection_list[4]]
    matched_thetaE[i] = [element for element in impact_ejection_list[5]]
    
    
# match CG Usal to Uim and UD 
def match_Usal_at_Vim_times(
    t_grid: np.ndarray,
    Usal_t: np.ndarray,
    t_im: np.ndarray,
    Vim: np.ndarray,
    *,
    mode: str = "interp"  # "interp" or "snap"
) -> Tuple[np.ndarray, np.ndarray]:
    t_grid = np.asarray(t_grid, float)
    Usal_t = np.asarray(Usal_t, float)
    t_im   = np.asarray(t_im, float)
    Vim    = np.asarray(Vim, float)

    # keep finite grid, sorted & unique
    gmask = np.isfinite(t_grid) & np.isfinite(Usal_t)
    t_grid = t_grid[gmask]; Usal_t = Usal_t[gmask]
    order = np.argsort(t_grid)
    t_grid = t_grid[order]; Usal_t = Usal_t[order]
    uniq = np.r_[True, np.diff(t_grid) > 0]
    t_grid = t_grid[uniq]; Usal_t = Usal_t[uniq]
    if t_grid.size < 2:
        return np.array([]), np.array([])

    # filter events: finite & within grid range
    emin, emax = t_grid[0], t_grid[-1]
    emask = np.isfinite(t_im) & np.isfinite(Vim) & (t_im >= emin) & (t_im <= emax)
    if not np.any(emask):
        return np.array([]), np.array([])
    t_im = t_im[emask]; Vim = Vim[emask]

    if mode == "interp":
        Usal_match = np.interp(t_im, t_grid, Usal_t)
    elif mode == "snap":
        dt = np.median(np.diff(t_grid))
        idx = np.rint((t_im - t_grid[0]) / dt).astype(int)
        idx = np.clip(idx, 0, len(t_grid) - 1)
        Usal_match = Usal_t[idx]
    else:
        raise ValueError("mode must be 'interp' or 'snap'.")

    ok = np.isfinite(Usal_match) & np.isfinite(Vim)
    return Usal_match[ok], Vim[ok]

def counts_per_step(t_grid: np.ndarray,
                    t_im: np.ndarray,
                    t_dep: np.ndarray) -> Dict[str, np.ndarray]:
    t = np.asarray(t_grid, float)
    N = t.size
    t0, tN = t[0], t[-1]

    def bin_indices(te: np.ndarray) -> np.ndarray:
        te = np.asarray(te, float)
        m = np.isfinite(te) & (te >= t0) & (te <= tN)
        te = te[m]
        # [t_i, t_{i+1}) bins, last time goes to last index
        idx = np.searchsorted(t, te, side="right") - 1
        idx = np.clip(idx, 0, N - 1)
        return idx

    idx_im  = bin_indices(t_im)
    idx_dep = bin_indices(t_dep)

    impact   = np.zeros(N, dtype=int)
    incident = np.zeros(N, dtype=int)

    np.add.at(impact,   idx_im,  1)
    np.add.at(incident, idx_im,  1)
    np.add.at(incident, idx_dep, 1)

    return impact, incident


t_dpm = np.linspace(0,5,501)
UsalUim_all, UsalUD_all = defaultdict(list), defaultdict(list)
Nim_all, Ninc_all = defaultdict(list), defaultdict(list)
for i in range(25):
    Usal_t = U_DPM_FLAT[i]
    UsalUim_all[i], _ = match_Usal_at_Vim_times(t_dpm, Usal_t, tim_all[i], Vim_all[i], mode='interp')
    UsalUD_all[i], _ = match_Usal_at_Vim_times(t_dpm, Usal_t, tD_all[i], VD_all[i], mode='interp')
    Nim_all[i], Ninc_all[i] = counts_per_step(t_dpm, tim_all[i], tD_all[i])
    
Usal_all_values = [value for sublist in UsalUim_all.values() for value in sublist] 
Usal_bin = np.linspace(0, max(Usal_all_values), 50)

def BinPrEffUsal_edges(Usal, N_im, N_inc, velsal_bins):
    U    = np.asarray(Usal, float)
    Nim  = np.asarray(N_im, float)
    Ninc = np.asarray(N_inc, float)
    edges = np.asarray(velsal_bins, float)
    # assign each time step to a bin: [edge_i, edge_{i+1}), last edge inclusive
    idx = np.searchsorted(edges, U, side="right") - 1
    nb  = edges.size - 1
    idx = np.clip(idx, 0, nb - 1)

    # sum counts in each bin
    Nim_sum  = np.bincount(idx, weights=Nim,  minlength=nb).astype(float)
    Ninc_sum = np.bincount(idx, weights=Ninc, minlength=nb).astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        Pr_eff = Nim_sum / Ninc_sum
        Pr_se  = np.sqrt(Pr_eff * (1.0 - Pr_eff) / Ninc_sum)

    # clean NaNs where Ninc_sum==0
    empty = (Ninc_sum == 0)
    Pr_eff[empty] = np.nan
    Pr_se[empty]  = np.nan
    
    Pr_se[(Pr_se == 0) & np.isfinite(Pr_se)] = np.nan
    
    Usal_Pr = (edges[:-1] + edges[1:]) / 2 

    return Pr_eff, Pr_se, Usal_Pr

Preff_mean, Preff_se = defaultdict(list), defaultdict(list)
for i in range(5):
    Preff_mean[i], Preff_se[i], Usal_Pr = BinPrEffUsal_edges(U_DPM_FLAT[i], Nim_all[i], Ninc_all[i], Usal_bin)
    
plt.figure()
for i in range(5):
    plt.errorbar(Usal_Pr, Preff_mean[i], yerr=Preff_se[i], fmt='o', capsize=5, color=colors[i], label=f'Shields=0.0{i+2}')
plt.xlabel('Usal [m/s]')
plt.ylabel('Pr_eff')
plt.legend()
plt.axis([0, None, 0, None]) 
plt.tight_layout()

constant = np.sqrt(9.81*D)  
#combine the values from all Shields numbers
Vim_all_Omega, Vre_all_Omega, exz_all_Omega, Thetaim_all_Omega, Thetare_all_Omega, VD_all_Omega, ThetaD_all_Omega, matched_Vim_Omega, matched_Thetaim_Omega, matched_NE_Omega, matched_UE_Omega, matched_mE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),defaultdict(list),defaultdict(list)
Vsal_all_Omega, Vrep_all_Omega = defaultdict(list), defaultdict(list)
Tsal_all_Omega, Tlast_all_Omega = defaultdict(list), defaultdict(list)
ThetaE_all_Omega, UE_all_Omega = defaultdict(list), defaultdict(list)
mpsal_all_Omega, mprep_all_Omega = defaultdict(list), defaultdict(list)
matched_thetaE_Omega = defaultdict(list)
UsalUim_all_Omega, UsalUD_all_Omega  = defaultdict(list), defaultdict(list)
for i in range (5): #loop over Omega 0-20 %
    selected_indices = list(range(i, 25, 5))  # Get indices like [0,5,10,15,20], [1,6,11,16,21], etc.
    Vim_all_Omega[i] = np.concatenate([Vim_all[j] for j in selected_indices]).tolist()
    Vre_all_Omega[i] = np.concatenate([Vre_all[j] for j in selected_indices]).tolist()
    Vsal_all_Omega[i] = np.concatenate([Vsal_all[j] for j in selected_indices]).tolist()
    mpsal_all_Omega[i] = np.concatenate([mp_sal_all[j] for j in selected_indices]).tolist()
    Vrep_all_Omega[i] = np.concatenate([Vrep_all[j] for j in selected_indices]).tolist()
    mprep_all_Omega[i] = np.concatenate([mp_rep_all[j] for j in selected_indices]).tolist()
    Tsal_all_Omega[i] = np.concatenate([Tsal_all[j] for j in selected_indices]).tolist()
    Tlast_all_Omega[i] = np.concatenate([Tlast_all[j] for j in selected_indices]).tolist()
    exz_all_Omega[i] = np.concatenate([exz_all[j] for j in selected_indices]).tolist()
    Thetaim_all_Omega[i] = np.concatenate([Theta_all[j] for j in selected_indices]).tolist()
    Thetare_all_Omega[i] = np.concatenate([Thetare_all[j] for j in selected_indices]).tolist()
    VD_all_Omega[i] = np.concatenate([VD_all[j] for j in selected_indices]).tolist()
    ThetaD_all_Omega[i] = np.concatenate([ThetaD_all[j] for j in selected_indices]).tolist()
    ThetaE_all_Omega[i] = np.concatenate([ThetaE_all[j] for j in selected_indices]).tolist()
    UE_all_Omega[i] = np.concatenate([UE_all[j] for j in selected_indices]).tolist()
    matched_Vim_Omega[i] = np.concatenate([matched_Vim[j] for j in selected_indices]).tolist()
    matched_Thetaim_Omega[i] = np.concatenate([matched_thetaim[j] for j in selected_indices]).tolist()
    matched_NE_Omega[i] = np.concatenate([matched_NE[j] for j in selected_indices]).tolist()
    matched_UE_Omega[i] = np.concatenate([matched_UE[j] for j in selected_indices]).tolist()
    matched_mE_Omega[i] = np.concatenate([matched_mE[j] for j in selected_indices]).tolist()
    matched_thetaE_Omega[i] = np.concatenate([matched_thetaE[j] for j in selected_indices]).tolist()
    UsalUim_all_Omega[i] = np.concatenate([UsalUim_all[j] for j in selected_indices]).tolist()
    UsalUD_all_Omega[i] = np.concatenate([UsalUD_all[j] for j in selected_indices]).tolist()

Uim_mean_Omega, Uim_stderr_Omega, UD_mean_Omega, UD_stderr_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
Tsal_mean_Omega, Tsal_stderr_Omega, Uim_Tsal = defaultdict(list), defaultdict(list), defaultdict(list)
Tlast_mean_Omega, Tlast_stderr_Omega, UD_Tlast = defaultdict(list), defaultdict(list), defaultdict(list)
NE_mean_Omega, UE_mean_Omega, UE_stderr_Omega, Uplot_NE_Omega, N_Einbin, ThetaE_mean_Omega, ThetaE_stderr_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
for i in range (5): #loop over Omega 0-20%
    # Uim_mean_Omega[i], Uim_stderr_Omega[i], UD_mean_Omega[i], UD_stderr_Omega[i], Usal_mean = module.BinUincUsal(Vim_all_Omega[i], VD_all_Omega[i], Vsal_all_Omega[i], Vrep_all_Omega[i], Thetaim_all_Omega[i], ThetaD_all_Omega[i], mpsal_all_Omega[i], mprep_all_Omega[i], Tsal_all_Omega[i], Tlast_all_Omega[i], Vsal_bin)
    Tsal_mean_Omega[i], Tsal_stderr_Omega[i], Uim_Tsal[i] = module.BinTsalUim(Vim_all_Omega[i], Thetaim_all_Omega[i], Tsal_all_Omega[i], mpsal_all_Omega[i], Vim_bin)
    Tlast_mean_Omega[i], Tlast_stderr_Omega[i], UD_Tlast[i] = module.BinTsalUim(VD_all_Omega[i], ThetaD_all_Omega[i], Tlast_all_Omega[i], mprep_all_Omega[i], Vde_bin)
    NE_mean_Omega[i], UE_mean_Omega[i], UE_std_Omega, UE_stderr_Omega[i], ThetaE_mean_Omega[i], ThetaE_std_Omega, ThetaE_stderr_Omega[i], N_Einbin[i], Uplot_NE_Omega[i]=module.BinNEUEmass(matched_Vim_Omega[i], VD_all_Omega[i], mpsal_all_Omega[i], mprep_all_Omega[i], matched_NE_Omega[i], matched_UE_Omega[i], matched_thetaE_Omega[i], matched_mE_Omega[i], Vimde_bin)
    Uim_mean_Omega[i], Uim_stderr_Omega[i], UsalUim_mean = module.BinUincUsal(Vim_all_Omega[i], Thetaim_all_Omega[i], UsalUim_all_Omega[i], Usal_bin)
    UD_mean_Omega[i], UD_stderr_Omega[i], UsalUD_mean = module.BinUincUsal(VD_all_Omega[i], ThetaD_all_Omega[i], UsalUD_all_Omega[i], Usal_bin)

# plot Uim - Usal, UD - Usal
plt.close('all')
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for i in range(5):
    plt.errorbar(UsalUim_mean/constant, Uim_mean_Omega[i]/constant, yerr=Uim_stderr_Omega[i]/constant, fmt='o', capsize=5, color=colors[i], label=f'$\Omega$={Omega[i]}%')
plt.xlabel(r'$U_{sal}$')
plt.ylabel(r'$U_{im}$')
plt.legend()
plt.axis([0, None, 0, None]) 
plt.subplot(1,2,2)
for i in range(5):
    plt.errorbar(UsalUD_mean/constant, UD_mean_Omega[i]/constant, yerr=UD_stderr_Omega[i]/constant, fmt='o', capsize=5, color=colors[i])    
plt.xlabel(r'$U_{sal}$')
plt.ylabel(r'$U_{D}$')
plt.axis([0, None, 0, None]) 
plt.tight_layout()

def weighted_r2(y_true, y_pred, weights):
    y_avg = np.average(y_true, weights=weights)
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - y_avg)**2)
    return 1 - ss_res / ss_tot

def linear_law(x, A, B):
    return A + B * x

def power_intercept(x, n, c):
    return x**n + c

# 1. Uim
n_Uim = np.zeros_like(Omega_real)
c_Uim = np.zeros_like(Omega_real)
plt.figure()
for i, om in enumerate(Omega_real):
    Usal = UsalUim_mean/constant
    Uim = Uim_mean_Omega[i]/constant
    se = Uim_stderr_Omega[i]/constant
    # Remove NaNs
    mask = np.isfinite(Usal) & np.isfinite(Uim) & np.isfinite(se)
    Usal, Uim, se = Usal[mask], Uim[mask], se[mask]
    popt, pcov = curve_fit(power_intercept, Usal, Uim, sigma=se, absolute_sigma=True)
    n_Uim[i], c_Uim[i] = popt
    Uim_pred = power_intercept(Usal, n_Uim[i], c_Uim[i])
    print('R2', weighted_r2(Uim, Uim_pred, 1/se**2))
    plt.errorbar(UsalUim_mean/constant, Uim_mean_Omega[i]/constant, yerr=Uim_stderr_Omega[i]/constant, fmt='o', capsize=5, color=colors[i])
    plt.plot(Usal, Uim_pred, '--', color=colors[i])

def c_bump(omega, A, omega0):
    return c_Uim[0] + A * omega * np.exp(-omega/omega0)

params_, _ = curve_fit(c_bump, Omega_real, c_Uim, p0=[(c_Uim.max()-c_Uim[0])*2.718, 0.01])
A_cUim, omega0_cUim = params_
c_Uim_smooth = c_bump(Omega_smooth, *params_)
c_Uim_fit = c_Uim[0] + A_cUim * Omega_real * np.exp(-Omega_real/omega0_cUim)
print(f'c_Uim_fit = {c_Uim[0]:.2f} + {A_cUim:.2f}*Omega*exp(-Omega/{omega0_cUim:.2f})')
plt.figure()
plt.scatter(Omega_real, c_Uim)
plt.plot(Omega_smooth, c_Uim_smooth)

params_, _ = curve_fit(power_intercept, Omega_real, n_Uim, p0=[1, 0.55])
n_Uim_smooth = power_intercept(Omega_smooth, *params_)
n_nUim, c_nUim = params_
n_Uim_fit = Omega_real**n_nUim + c_nUim
print(f"n_Uim_fit = Omega_real**{n_nUim:.2f} + {c_nUim:.2f}")
plt.figure()
plt.scatter(Omega_real, n_Uim)
plt.plot(Omega_smooth, n_Uim_smooth)

UsalUim_fit = np.linspace(0, max([value for sublist in UsalUim_all_Omega.values() for value in sublist]/constant), 100)
Uim_fit = defaultdict()
# Loop over each element in alpha and multiply with U_fit_new
for i in range(5):
    Uim_fit[i] = power_intercept(UsalUim_fit, n_Uim_fit[i], c_Uim_fit[i])

all_Uim_ori, all_Uim_fit_resampled, weight_Uim_all = [],[],[]
for i in range(5):
    # Create interpolator from 100-point fit
    interpolator = interp1d(UsalUim_fit, Uim_fit[i], kind='linear', fill_value='extrapolate')
    Usal = UsalUim_mean/constant
    Uim = Uim_mean_Omega[i]/constant
    se = Uim_stderr_Omega[i]/constant
    # Remove NaNs
    mask = np.isfinite(Usal) & np.isfinite(Uim) & np.isfinite(se)
    Usal, Uim, se = Usal[mask], Uim[mask], se[mask]
    Uim_fit_resampled = interpolator(Usal)
    all_Uim_ori.append(Uim)
    all_Uim_fit_resampled.append(Uim_fit_resampled)
    weight_Uim_all.append(1/(se**2))
    
y_all = np.concatenate(all_Uim_ori)
y_pred_all = np.concatenate(all_Uim_fit_resampled)
weight_glo = np.concatenate(weight_Uim_all)
# Now compute R²
R2_Uim = weighted_r2(y_all, y_pred_all, weights=weight_glo)
print('R2_Uim=',R2_Uim)

# UD = f(Usal), for now only fit the dry case
n_UD = np.zeros_like(Omega_real)
c_UD = np.zeros_like(Omega_real)
plt.figure()
for i, om in enumerate(Omega_real):
    UsalD = UsalUD_mean/constant
    UD = UD_mean_Omega[i]/constant
    se = UD_stderr_Omega[i]/constant
    # Remove NaNs
    mask = np.isfinite(UsalD) & np.isfinite(UD) & np.isfinite(se)
    UsalD, UD, se = UsalD[mask], UD[mask], se[mask]
    # p0 = [5, 0.0, 0.5, 1.0, 0.6, 0.5, 0.05, 1.0]
    popt, pcov = curve_fit(power_intercept, UsalD, UD, sigma=se, absolute_sigma=True, maxfev=20000)
    n_UD[i], c_UD[i] = popt
    UD_pred = power_intercept(UsalD, n_UD[i], c_UD[i])
    print('R2', weighted_r2(UD, UD_pred, 1/se**2))
    plt.errorbar(UsalUD_mean/constant, UD_mean_Omega[i]/constant, yerr=UD_stderr_Omega[i]/constant, fmt='o', capsize=5, color=colors[i])
    plt.plot(UsalD, UD_pred, '--', color=colors[i])
    
print(f'UD = Usal**{n_UD[0]:.2f}+{c_UD[0]:.2f}')
UD_fit_dry = power_intercept(UsalUim_fit, n_UD[0], c_UD[0])

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for i in range(5):
    plt.errorbar(UsalUim_mean/constant, Uim_mean_Omega[i]/constant, yerr=Uim_stderr_Omega[i]/constant, fmt='o', capsize=5, color=colors[i], label=f'$\Omega$={Omega[i]}%')    
    plt.plot(UsalUim_fit, Uim_fit[i], '--', color=colors[i])
plt.xlabel(r'$U_{sal}/\sqrt{gd}$')
plt.ylabel(r'$U_{im}/\sqrt{gd}$')
plt.axis([0, None, 0, None])
plt.legend()
plt.subplot(1,2,2)
for i in range(5):
    plt.errorbar(UsalUD_mean/constant, UD_mean_Omega[i]/constant, yerr=UD_stderr_Omega[i]/constant, fmt='o', capsize=5, color=colors[i])    
plt.plot(UsalUim_fit, UD_fit_dry, '--', color=colors[0])
plt.xlabel(r'$U_{sal}/\sqrt{gd}$')
plt.ylabel(r'$U_{D}/\sqrt{gd}$')
plt.axis([0, None, 0, None])  
plt.tight_layout()



#compute Tsal = 2*Uim*sin(theta_im)/g
Omega_real = np.array(Omega)*0.01
Tsal_computed, Trep_computed = [], []
for i in range(5):
    alpha = 50.40 - 25.53 * Omega_real[i]**0.5
    beta = -100.12 * (1-np.exp(-2.34*Omega_real[i])) + 159.33
    sin_theta_im = alpha/(Uim_Tsal[i]/constant + beta)
    Tsal_computed.append(2 * Uim_Tsal[i] * sin_theta_im / g)
 
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for i in range(5):
    plt.errorbar(Uim_Tsal[i], Tsal_mean_Omega[i], yerr=Tsal_stderr_Omega[i], fmt='o', capsize=5, color=colors[i], label=f'$\Omega$={Omega[i]}%')
    plt.plot(Uim_Tsal[i], Tsal_computed[i], '--', color=colors[i])
plt.xlabel(r'$U_{im}$ [m/s]')
plt.ylabel(r'$T_{sal}$ [s]')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
plt.subplot(1,2,2)
for i in range(5):
    plt.errorbar(UD_Tlast[i], Tlast_mean_Omega[i], yerr=Tlast_stderr_Omega[i], fmt='o', capsize=5, color=colors[i], label=f'$\Omega$={Omega[i]}%')
plt.xlabel(r'$U_{D}$ [m/s]')
plt.ylabel(r'$T_{dep}$ [s]') 
plt.xlim(left=0)
plt.ylim(bottom=0)   
plt.tight_layout()

def power_model(U, a, b):
    return a * U ** b

# # Store fit results
# a_fit = np.zeros_like(Omega_real)
# b_fit = np.zeros_like(Omega_real)
# R2_fit = np.zeros_like(Omega_real)

# for i, om in enumerate(Omega_real):
#     U = Uim_Tsal[i]
#     T = Tsal_mean_Omega[i]
#     se = Tsal_stderr_Omega[i]

#     # Remove NaNs
#     mask = np.isfinite(U) & np.isfinite(T) & np.isfinite(se)
#     U, T, se = U[mask], T[mask], se[mask]

#     # Fit weighted by 1/se
#     popt, pcov = curve_fit(power_model, U, T, sigma=se, absolute_sigma=True)
#     a_fit[i], b_fit[i] = popt

#     # Calculate weighted R^2
#     T_pred = power_model(U, a_fit[i], b_fit[i])
#     w = 1 / se**2
#     T_mean = np.average(T, weights=w)
#     SS_res = np.sum(w * (T - T_pred)**2)
#     SS_tot = np.sum(w * (T - T_mean)**2)
#     R2_fit[i] = 1 - SS_res / SS_tot

# print("Fitted coes a(Omega):", a_fit, "Fitted coes b(Omega):", b_fit)
# print("Weighted R^2:", R2_fit)

# a_fit_final = np.mean(a_fit[:-1])
# print(f"a_fit_final={a_fit_final:.2f}")
# params_b, _ = curve_fit(linear_law, Omega_real, b_fit, p0=[-1, 1])
# # Generate smooth curve for plotting
# Omega_smooth = np.linspace(min(Omega_real), max(Omega_real), 100)
# b_fit_smooth = linear_law(Omega_smooth, *params_b)
# a_a_opt, b_a_opt = params_b
# b_fit_final = a_a_opt + b_a_opt*Omega_real
# print(f"b_fit_final = {a_a_opt:.2f} {b_a_opt:.2f}*Omega_real")

# plt.figure()
# plt.plot(Omega_real, b_fit, 'o')
# plt.plot(Omega_smooth,b_fit_smooth)
# plt.xlabel('Omega')
# plt.ylabel('b_fit')


# U_fit_im = np.linspace(0, max([value for sublist in Uim_Tsal.values() for value in sublist]), 100)
# Tim_fit = defaultdict()
# # Loop over each element in alpha and multiply with U_fit_new
# for i in range(5):
#     Tim_fit[i] = power_model(U_fit_im, a_fit_final, b_fit_final[i])

# all_Tim_ori, all_Tim_fit_resampled, weight_Tim_all = [],[],[]
# for i in range(5):
#     # Create interpolator from 100-point fit
#     interpolator = interp1d(U_fit_im, Tim_fit[i], kind='linear', fill_value='extrapolate')
#     U = Uim_Tsal[i]
#     T = Tsal_mean_Omega[i]
#     se = Tsal_stderr_Omega[i]
#     # Remove NaNs
#     mask = np.isfinite(U) & np.isfinite(T) & np.isfinite(se)
#     U, T, se = U[mask], T[mask], se[mask]
#     Tim_fit_resampled = interpolator(U)
#     all_Tim_ori.append(T)
#     all_Tim_fit_resampled.append(Tim_fit_resampled)
#     weight_Tim_all.append(1/(se**2))
    
# y_all = np.concatenate(all_Tim_ori)
# y_pred_all = np.concatenate(all_Tim_fit_resampled)
# weight_glo = np.concatenate(weight_Tim_all)
# # Now compute R²
# R2_Tim = weighted_r2(y_all, y_pred_all, weights=weight_glo)
# print('R2_Tim=',R2_Tim)


# plt.figure(figsize=(6,5))
# for i in range(5):
#     plt.errorbar(Uim_Tsal[i], Tsal_mean_Omega[i], yerr=Tsal_stderr_Omega[i], fmt='o', capsize=5, color=colors[i], label=f'$\Omega$={Omega[i]}%')
#     plt.plot(U_fit_im, Tim_fit[i], '--', color=colors[i])
# plt.xlabel(r'$U_{im}$ [m/s]')
# plt.ylabel(r'$T_{sal}$ [s]')
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend()
    

# # TD - UD
# U = np.array(list(chain.from_iterable(UD_Tlast.values())))
# T = np.array(list(chain.from_iterable(Tlast_mean_Omega.values())))
# se = np.array(list(chain.from_iterable(Tlast_stderr_Omega.values())))
# mask = np.isfinite(U) & np.isfinite(T) & np.isfinite(se)
# U, T, se = U[mask], T[mask], se[mask]

# popt, pcov = curve_fit(power_model, U, T, sigma=se, absolute_sigma=True)
# a_Tlast, b_Tlast = popt
# print(f'Tlast={a_Tlast:.2f} *UD **{b_Tlast:.2f}')
# # 3. Plot the fit 
# UDTlast_fit = np.linspace(0, max(np.array(list(chain.from_iterable(UD_Tlast.values())))), 100) #for the fit
# Tlast_fit = a_Tlast * UDTlast_fit ** b_Tlast

# #calculate R^2
# # Create interpolator from 100-point fit
# interpolator = interp1d(UDTlast_fit, Tlast_fit, kind='linear', fill_value='extrapolate')
# # Evaluate fit at the same x-values as Uthetaplot
# Tlast_fit_resampled = interpolator(U)
# # Now compute R²
# r2_Tlast = weighted_r2(T, Tlast_fit_resampled, 1/se**2)
# print('r2_Tlast:',r2_Tlast)


# plt.figure(figsize=(6,5))
# for i in range(5):
#     plt.errorbar(UD_Tlast[i], Tlast_mean_Omega[i], yerr=Tlast_stderr_Omega[i], fmt='o', capsize=5, color=colors[i], label=f'$\Omega$={Omega[i]}%')
# plt.plot(UDTlast_fit, Tlast_fit, 'k--')
# plt.xlabel(r'$U_{D}$ [m/s]')
# plt.ylabel(r'$T_{dep}$ [s]')
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend()

# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# for i in range(5):
#     plt.errorbar(Uim_Tsal[i], Tsal_mean_Omega[i], yerr=Tsal_stderr_Omega[i], fmt='o', capsize=5, color=colors[i], label=f'$\Omega$={Omega[i]}%')
#     plt.plot(U_fit_im, Tim_fit[i], '--', color=colors[i])
# plt.xlabel(r'$U_{im}$ [m/s]')
# plt.ylabel(r'$T_{sal}$ [s]')
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.legend()
# plt.subplot(1,2,2)
# for i in range(5):
#     plt.errorbar(UD_Tlast[i], Tlast_mean_Omega[i], yerr=Tlast_stderr_Omega[i], fmt='o', capsize=5, color=colors[i], label=f'$\Omega$={Omega[i]}%')
# plt.plot(UDTlast_fit, Tlast_fit, 'k--')
# plt.xlabel(r'$U_{D}$ [m/s]')
# plt.ylabel(r'$T_{last}$ [s]')
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.tight_layout()

# # mass-based NE
# plt.figure(figsize=(6,5.5))
# for i in range(5):
#     plt.scatter(Uplot_NE_Omega[i]/constant, NE_mean_Omega[i], s=np.sqrt(N_Einbin[i])*5, label=f"$\\Omega$={Omega[i]}%",color=colors[i])
# plt.xlim(0,225)
# plt.ylim(0, 5.75)
# plt.xlabel(r'$U_\mathrm{im}/\sqrt{gd}$ [-]', fontsize=14)
# plt.ylabel(r'$\bar{N}_\mathrm{E}$ [-]', fontsize=14)
# plt.legend(loc='upper left', fontsize=12)

