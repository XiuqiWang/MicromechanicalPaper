# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 13:49:01 2025

@author: WangX3
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import module

# Load the data from the saved file
data_dict = {}  # Dictionary to store data
with open("input_pkl/data6_0.pkl", 'rb') as f:
    data_dict["data"] = pickle.load(f)  # Store in dictionary

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

filename = "data"
data = data_dict[filename]
num_p = 2725
ParticleID=np.linspace(num_p-310,num_p-1,310)
ParticleID_int = ParticleID.astype(int)
#cal erosion and deposition properties for each Omega
#EDindices, E, VX, VExVector, VEzVector, VEx, VEz, ME, MD
EDindices, ME, MD, VExz_mean_t, VDxz_mean_t, D_vector_t, E_vector_t=module.store_particle_id_data(data,ParticleID_int,coe_h,dt,N_inter,D)
Z = np.array([[time_step['Position'][i][2] for i in ParticleID_int] for time_step in data])

plt.close('all')
id_p = 285
plt.figure(figsize=(12,5))
plt.plot(t_ver, Z[:,id_p]/D, linestyle='-', marker='.', color='k', markersize=3, linewidth=1)
plt.plot(t_ver[EDindices[id_p][0]], Z[EDindices[id_p][0],id_p]/D, 'ob', label='Ejections', markerfacecolor='none')
plt.plot(t_ver[EDindices[id_p][1]], Z[EDindices[id_p][1],id_p]/D, 'vb', label='Depositions', markerfacecolor='none')
plt.xlabel(r'$t$ [s]', fontsize=14)
plt.ylabel(r'$Z_\mathrm{p}/d$ [-]',fontsize=14)
plt.xlim(0,5)
plt.text(0.02, 0.92, '(a)', transform=plt.gca().transAxes, fontsize=16, fontweight='bold')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()