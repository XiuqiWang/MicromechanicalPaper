# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 16:38:14 2025

@author: WangX3
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import module
from scipy.stats import expon
from scipy.stats import lognorm
from scipy.stats import gaussian_kde

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
Vre_all = defaultdict(list)
N_range = np.full(25, 0).astype(int)
for i in range (25):
    exz_all[i] = [value for sublist in exz_vector_t[i][N_range[i]:] for value in sublist]
    Vim_all[i] = [value[0] for sublist in IM_vector_t[i][N_range[i]:] for value in sublist]
    Vre_all[i] = [exz * vim[0] for exz_list, vim_list in zip(exz_vector_t[i][N_range[i]:], IM_vector_t[i][N_range[i]:]) for exz, vim in zip(exz_list, vim_list)]
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
    IDE = [value[1] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    xE = [value[2] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    PE = [value[3] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    EE = [value[5] for sublist in E_vector_t[i][N_range[i]:] for value in sublist] #kinetic energy
    ThetaE_all[i] = [value[6] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    UE_all[i] = [value[0] for sublist in E_vector_t[i][N_range[i]:] for value in sublist]
    ejection_list[i] = [IDE, xE, UE_all[i], PE, EE, ThetaE_all[i]]
    #print('Ne/Nim',len(IDE)/len(IDim))

# Vim_all_values = [value for sublist in Vim_all.values() for value in sublist]
# Vim_bin = np.linspace(min(Vim_all_values), max(Vim_all_values), 10)
# Vimde_all_values = [value for sublist in Vim_all.values() for value in sublist] + [value for sublist in VD_all.values() for value in sublist]
# Vimde_bin = np.linspace(min(Vimde_all_values), max(Vimde_all_values), 10)
# Thetaimde_all_values = [value for sublist in Theta_all.values() for value in sublist] + [value for sublist in ThetaD_all.values() for value in sublist]
# Thetaimde_bin = np.linspace(min(Thetaimde_all_values), max(Thetaimde_all_values), 10)
# Thetaim_all_values = [value for sublist in Theta_all.values() for value in sublist]
# Thetaim_bin = np.linspace(min(Thetaim_all_values), max(Thetaim_all_values), 10)
# matched_Vim, matched_thetaim, matched_NE, matched_UE, matched_EE, matched_thetaE = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
# for i in range (25):
#     impact_ejection_list=module.match_ejection_to_impact(impact_list[i], ejection_list[i], dt)
#     # impact_ejection_list=module.match_ejection_to_impactanddeposition(impact_deposition_list[i], ejection_list[i])
#     matched_Vim[i] = [element for element in impact_ejection_list[0]]
#     matched_thetaim[i] = [element for element in impact_ejection_list[1]]
#     matched_NE[i] = [element for element in impact_ejection_list[2]]
#     matched_UE[i] = [element for element in impact_ejection_list[3]]
#     matched_EE[i] = [element for element in impact_ejection_list[4]]
#     matched_thetaE[i] = [element for element in impact_ejection_list[5]]
    
constant = np.sqrt(9.81*D)  
#combine the values from all Shields numbers
Vim_all_Omega, exz_all_Omega, Theta_all_Omega, VD_all_Omega, ThetaD_all_Omega, matched_Vim_Omega, matched_Thetaim_Omega, matched_NE_Omega, matched_UE_Omega, matched_EE_Omega = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list),defaultdict(list),defaultdict(list)
ThetaE_all_Omega, UE_all_Omega = defaultdict(list), defaultdict(list)
Vre_all_Omega , Thetare_all_Omega = defaultdict(list), defaultdict(list)
matched_thetaE_Omega = defaultdict(list)
for i in range (5): #loop over Omega 0-20 %
    selected_indices = list(range(i, 25, 5))  # Get indices like [0,5,10,15,20], [1,6,11,16,21], etc.
    Vim_all_Omega[i] = np.concatenate([Vim_all[j] for j in selected_indices]).tolist()
    exz_all_Omega[i] = np.concatenate([exz_all[j] for j in selected_indices]).tolist()
    Theta_all_Omega[i] = np.concatenate([Theta_all[j] for j in selected_indices]).tolist()
    Thetare_all_Omega[i] = np.concatenate([Thetare_all[j] for j in selected_indices]).tolist()
    Vre_all_Omega[i] = np.concatenate([Vre_all[j] for j in selected_indices]).tolist()
    VD_all_Omega[i] = np.concatenate([VD_all[j] for j in selected_indices]).tolist()
    ThetaD_all_Omega[i] = np.concatenate([ThetaD_all[j] for j in selected_indices]).tolist()
    ThetaE_all_Omega[i] = np.concatenate([ThetaE_all[j] for j in selected_indices]).tolist()
    UE_all_Omega[i] = np.concatenate([UE_all[j] for j in selected_indices]).tolist()
    # matched_Vim_Omega[i] = np.concatenate([matched_Vim[j] for j in selected_indices]).tolist()
    # matched_Thetaim_Omega[i] = np.concatenate([matched_thetaim[j] for j in selected_indices]).tolist()
    # matched_NE_Omega[i] = np.concatenate([matched_NE[j] for j in selected_indices]).tolist()
    # matched_UE_Omega[i] = np.concatenate([matched_UE[j] for j in selected_indices]).tolist()
    # matched_EE_Omega[i] = np.concatenate([matched_EE[j] for j in selected_indices]).tolist()
    # matched_thetaE_Omega[i] = np.concatenate([matched_thetaE[j] for j in selected_indices]).tolist()

def fit_exponential(data, num_points=1000):
    """
    Fit exponential distribution to input data.

    Parameters:
        data (array-like): Input 1D data array.
        num_points (int): Number of points for the fitted PDF line.

    Returns:
        loc (float): Location parameter of the fitted exponential.
        scale (float): Scale parameter (1/lambda) of the fitted exponential.
        x_fit (ndarray): x-values for the fitted curve.
        y_fit (ndarray): Corresponding PDF values for x_fit.
    """
    # Fit the exponential distribution to data
    loc, scale = expon.fit(data)

    # Create x values for plotting PDF
    x_fit = np.linspace(min(data), max(data), num_points)
    y_fit = expon.pdf(x_fit, loc=loc, scale=scale)

    return loc, scale, x_fit, y_fit

def fit_lognormal(data, num_points=1000):
    """
    Fit lognormal distribution to input data.

    Parameters:
        data (array-like): Input 1D data array.
        num_points (int): Number of points for the fitted PDF line.

    Returns:
        shape (float): Shape parameter (sigma) of the lognormal.
        loc (float): Location parameter (usually close to 0).
        scale (float): Scale parameter (exp(mu)) of the lognormal.
        x_fit (ndarray): x-values for the fitted curve.
        y_fit (ndarray): Corresponding PDF values for x_fit.
    """
    # Fit lognormal distribution to data
    shape, loc, scale = lognorm.fit(data, floc=0)  # floc=0 often improves lognormal fit

    # Create x values for plotting PDF
    x_fit = np.linspace(min(data), max(data), num_points)
    y_fit = lognorm.pdf(x_fit, s=shape, loc=loc, scale=scale)

    return shape, loc, scale, x_fit, y_fit

#UE
plt.figure(figsize=(6, 5))
for i in range(5):
    # Calculate histogram (density=True for probability density)
    counts, bin_edges = np.histogram(UE_all_Omega[i], bins=50, density=True)
    # Create the step plot
    plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# plt.ylim(-0.01,0.13)
plt.xlabel(r'$U_\mathrm{E}$ [m/s]', fontsize=14)
plt.ylabel('Probability Density [-]', fontsize=14)
plt.tight_layout()
plt.show()
# Fit
# loc, scale, x_fit, y_fit = fit_exponential(data)
shape, loc, scale, x_fit, y_fit = fit_lognormal(UE_all_Omega[0])

# Plot
plt.hist(UE_all_Omega[0], bins=50, density=True, alpha=0.6, label='Histogram')
# plt.plot(x_fit, y_fit, 'r-', label=f'Exponential Fit\nloc={loc:.2f}, scale={scale:.2f}')
plt.plot(x_fit, y_fit, 'r-', label=f'Lognormal Fit\nσ={shape:.2f}, μ={np.log(scale):.2f}')
plt.xlabel('UE')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Ure
plt.figure(figsize=(6, 5))
for i in range(5):
    # Calculate histogram (density=True for probability density)
    counts, bin_edges = np.histogram(Vre_all_Omega[i], bins=50, density=True)
    # Create the step plot
    plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# plt.ylim(-0.01,0.13)
plt.xlabel(r'$U_\mathrm{re}$ [m/s]', fontsize=14)
plt.ylabel('Probability Density [-]', fontsize=14)
plt.tight_layout()
plt.show()
# Fit
# loc, scale, x_fit, y_fit = fit_exponential(data)
shape, loc, scale, x_fit, y_fit = fit_lognormal(Vre_all_Omega[0])

# Plot
plt.hist(Vre_all_Omega[0], bins=50, density=True, alpha=0.6, label='Histogram')
# plt.plot(x_fit, y_fit, 'r-', label=f'Exponential Fit\nloc={loc:.2f}, scale={scale:.2f}')
plt.plot(x_fit, y_fit, 'r-', label=f'Lognormal Fit\nσ={shape:.2f}, μ={np.log(scale):.2f}')
plt.xlabel('Ure')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#Uim
plt.figure(figsize=(6, 5))
for i in range(5):
    # Calculate histogram (density=True for probability density)
    counts, bin_edges = np.histogram(Vim_all_Omega[i], bins=50, density=True)
    # Create the step plot
    plt.step(bin_edges[:-1], counts, where='mid', color=colors[i], label=f"$\\Omega$={Omega[i]}%")
# plt.ylim(-0.01,0.13)
plt.xlabel(r'$U_\mathrm{re}$ [m/s]', fontsize=14)
plt.ylabel('Probability Density [-]', fontsize=14)
plt.tight_layout()
plt.show()
# Fit
# loc, scale, x_fit, y_fit = fit_exponential(data)
shape, loc, scale, x_fit, y_fit = fit_lognormal(Vim_all_Omega[0])

# Plot
plt.hist(Vim_all_Omega[0], bins=50, density=True, alpha=0.6, label='Histogram')
# plt.plot(x_fit, y_fit, 'r-', label=f'Exponential Fit\nloc={loc:.2f}, scale={scale:.2f}')
plt.plot(x_fit, y_fit, 'r-', label=f'Lognormal Fit\nσ={shape:.2f}, μ={np.log(scale):.2f}')
plt.xlabel('Uim')
plt.ylabel('PDF')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# # U_inc distribution in each bin for dry case
# U_INC = [Vim_all_Omega[0] + VD_all_Omega[0]]
# U_INC = np.array(U_INC)
# # Digitize velocities into bin indices
# # bins[i] <= v < bins[i+1] gets index i+1, so subtract 1 to get 0-based bin index
# bin_indices = np.digitize(U_INC, Vimde_bin) - 1
# # Number of bins
# n_bins = len(Vimde_bin) - 1
# # Plot PDFs for each bin
# fig, axs = plt.subplots(3, 3, figsize=(15, 10))
# axs = axs.flatten()
# for i in range(n_bins):
#     # Get velocities in bin i
#     v_in_bin = U_INC[(bin_indices == i)]   
#     ax = axs[i]
#     if len(v_in_bin) > 1:
#         # Estimate PDF using kernel density estimation
#         kde = gaussian_kde(v_in_bin)
#         x_vals = np.linspace(Vimde_bin[i], Vimde_bin[i+1], 100)
#         ax.plot(x_vals, kde(x_vals), label=f'Bin {i+1}')
#     else:
#         # If not enough data, just mark as empty
#         ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=10)
#     ax.set_title(f'Bin {i+1}: {Vimde_bin[i]:.2f}–{Vimde_bin[i+1]:.2f} m/s')
#     ax.set_xlabel('Velocity')
#     ax.set_ylabel('PDF')
#     ax.grid(True)
# plt.tight_layout()
# plt.show()

