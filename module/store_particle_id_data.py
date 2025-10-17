# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:32:32 2024

@author: WangX3
"""

import numpy as np
import math
from .hop_average_Usal import hop_average_Usal

def store_particle_id_data(data,ID_Particle, coe_h, dt, N_inter, D):
    X = np.array([[time_step['Position'][i][0] for i in ID_Particle] for time_step in data])
    Z = np.array([[time_step['Position'][i][2] for i in ID_Particle] for time_step in data])
    Vxstored = np.array([[time_step['Velocity'][i][0] for i in ID_Particle] for time_step in data])
    Vzstored = np.array([[time_step['Velocity'][i][2] for i in ID_Particle] for time_step in data])
    Rp = np.array([[time_step['Radius'][i] for i in ID_Particle] for time_step in data])
    Vp = (4/3)*math.pi*Rp**3 
    
    EDindices = []
    VExVector, VEzVector = [], []
    #ME_tot, MD_tot = 0, 0
    VX, VZ, E = [], [], []
    VEx = [[] for _ in range(N_inter)]
    VEz = [[] for _ in range(N_inter)]
    VExz_t = [[] for _ in range(N_inter)]
    VDx = [[] for _ in range(N_inter)]
    VDz = [[] for _ in range(N_inter)]
    VDxz_t = [[] for _ in range(N_inter)]
    VExz_mean_t = np.full(N_inter, np.nan)
    VDxz_mean_t = np.full(N_inter, np.nan)
    ME,MD,MoE,MoD,MpE,MpD = np.zeros(N_inter), np.zeros(N_inter), np.zeros(N_inter), np.zeros(N_inter), np.zeros(N_inter), np.zeros(N_inter)
    E_vector_t,D_vector_t = [[] for _ in range(N_inter)],[[] for _ in range(N_inter)]
    VD_TD_vector_t = [[] for _ in range(N_inter)]

    g = 9.81
    Lx = D * 100
    Ly = 2 * D
    A = Lx * Ly
    
    t = np.linspace(dt, 5, int(5 / dt))

    for i in range(len(ID_Particle)):
        z = Z[:,i]
        x = X[:,i]
        Vx = Vxstored[:,i]
        Vz = Vzstored[:,i]
        # Calculate average particle velocities from position differences/dt
        # Vx = np.concatenate(([0], (X[1:,i] - X[:-1,i]) / dt))
        # Vz = np.concatenate(([0], (Z[1:,i] - Z[:-1,i]) / dt))
        
        ke = 0.5 * Vz**2
        pe = g * z
        e = ke + pe
        d_h = coe_h * D
        thre_e = g * d_h
        ID_Ei, ID_Di = output_id(e, thre_e, dt)

        # Correct Vx when an ejected particle crosses the boundary
        # Index_neg = np.where(Vx < -Lx * 0.25 / dt)[0]
        # Vx[Index_neg] += Lx / dt

        VX.append(Vx)
        VZ.append(Vz)
        E.append(e)

        if ID_Ei.size > 0:
            ID_Eafter = ID_Ei + 1
            ID_Eafter = np.clip(ID_Eafter, 0, len(Vx)-2)
            VExi,VEzi = Vx[ID_Eafter],Vz[ID_Eafter]
            #correct ejection velocities at the evaluation height by extrapolating
            high_Eindices = np.where(z[ID_Eafter] > d_h)
            low_Eindices = np.where(z[ID_Eafter] < d_h)
            VEzi[high_Eindices] = np.sqrt(2*9.81*(z[ID_Eafter[high_Eindices]]-d_h)+Vz[ID_Eafter[high_Eindices]]**2)
            VEzi[low_Eindices] = np.sqrt(Vz[ID_Eafter[low_Eindices]]**2 - 2*9.81*(d_h-z[ID_Eafter[low_Eindices]]))  
            #correct horizontal ejection velocity by interpolating (for the ones lower than 1.5D)
            Vz1_low,Vx1_low,Vx2_low = Vz[ID_Eafter[low_Eindices]],Vx[ID_Eafter[low_Eindices]],Vx[ID_Eafter[low_Eindices]+1]
            z1_low = z[ID_Eafter[low_Eindices]]
            dt_ratio_low = (2*Vz1_low + np.sqrt(4*Vz1_low**2 - 4*9.81*(2*d_h-2*z1_low))) / (2*9.81) /dt
            newlow_Eindices = np.where(dt_ratio_low < 1)
            VExi[low_Eindices[0][newlow_Eindices[0]]] = dt_ratio_low[newlow_Eindices[0]] * (Vx2_low[newlow_Eindices[0]]-Vx1_low[newlow_Eindices[0]]) + Vx1_low[newlow_Eindices[0]]
            #for the ones higher than 1.5D
            Vx0_high, Vx1_high = Vx[ID_Eafter[high_Eindices]-1], Vx[ID_Eafter[high_Eindices]]
            Vz1_high = Vz[ID_Eafter[high_Eindices]]
            z1_high = z[ID_Eafter[high_Eindices]]
            dt_ratio_high = (-2*Vz1_high + np.sqrt(4*Vz1_high**2 - 4*9.81*(2*d_h-2*z1_high))) / (2*9.81) / dt
            newhigh_Eindices = np.where(dt_ratio_high < 1)
            VExi[high_Eindices[0][newhigh_Eindices[0]]] = dt_ratio_high[newhigh_Eindices[0]] * (Vx0_high[newhigh_Eindices[0]]-Vx1_high[newhigh_Eindices[0]]) + Vx1_high[newhigh_Eindices[0]]
            #renew the stored ejection velocities
            VExzi = np.sqrt(VExi**2+VEzi**2)
            
            ID_Dbefore = ID_Di - 1
            VDxi = Vx[ID_Dbefore]
            VDzi = Vz[ID_Dbefore]
            #correct vertical deposition velocities at the evaluation height by extrapolating
            high_Dindices = np.where(z[ID_Dbefore] > d_h)
            low_Dindices = np.where(z[ID_Dbefore] < d_h)
            VDzi[high_Dindices] = -np.sqrt(2*9.81*(z[ID_Dbefore[high_Dindices]]-d_h)+Vz[ID_Dbefore[high_Dindices]]**2)
            VDzi[low_Dindices] = -np.sqrt(Vz[ID_Dbefore[low_Dindices]]**2 - 2*9.81*(d_h-z[ID_Dbefore[low_Dindices]]))    
            VDxzi = np.sqrt(VDxi**2+VDzi**2)
            ThetaDi_radian = np.arctan(np.abs(VDzi/VDxi))
            ThetaDi = np.degrees(ThetaDi_radian)
            # get the id at the top of the reptation hops
            reptop_indices = []
            if ID_Di[0] < ID_Ei[0]:
                ID_Di_new = ID_Di[1:]     # remove the first element
            else:
                ID_Di_new = ID_Di.copy()  # keep the original array
            if ID_Ei[-1] > ID_Di[-1]:
                ID_Ei_new = ID_Ei[:-1]     # remove the last element
            else:
                ID_Ei_new = ID_Ei.copy()  # keep the original array
            for start, end in zip(ID_Ei_new, ID_Di_new):
                Vz_rep = Vz[start:end]
                crossing_indice = np.where((Vz_rep[:-1] >= 0) & (Vz_rep[1:] < 0))[0]
                if len(crossing_indice)>0:
                    reptop_indices.append(start + crossing_indice[-1])
                elif len(Vz_rep)<3:
                    reptop_indices.append(end)
                else:
                    reptop_indices.append(end-1)
            
            IDdepai = np.maximum(-np.asarray(ID_Di_new) + 2*np.asarray(reptop_indices), 0)
            Vrepxi = hop_average_Usal(t, Vx, ID_Di_new, IDdepai)
            Trepi = (ID_Di_new - IDdepai) * dt # reptation time
            Tdepi =  (ID_Di_new - ID_Ei_new) * dt # residence time before depositing
            
            xEi = x[ID_Ei]
            zEi = z[ID_Ei]
            mp = Vp[0, i] * 2650
            xDi = x[ID_Di]
            EEi = 0.5*mp*VExzi**2
            ThetaEi_radian = np.arctan(np.abs(VEzi/VExi))
            ThetaEi = np.degrees(ThetaEi_radian)
            # VExVector.extend(VExi)
            # VEzVector.extend(VEzi)
            #ME_tot += np.sum(mE) / 5 / A
            #MD_tot += np.sum(mD) / 5 / A
            # Distribute values into intervals
            for j, idx in enumerate(ID_Ei):
                interval = min(int(np.ceil((idx + 1) / (len(X[:, i]) / N_inter))), N_inter)
                VEx[interval - 1].append(VExi[j]*mp)
                VEz[interval - 1].append(VEzi[j]*mp)
                VExz_t[interval - 1].append(VExzi[j]*mp)
                ME[interval - 1] += mp / (5 / N_inter) / A
                MpE[interval - 1] += mp
                E_vector_t[interval - 1].append([VExzi[j], ID_Eafter[j], xEi[j], i, zEi[j], EEi[j], ThetaEi[j], mp])
            for j, idx in enumerate(ID_Di):
                interval = min(int(np.ceil((idx - 1) / (len(X[:, i]) / N_inter))), N_inter)
                VDx[interval - 1].append(VDxi[j]*mp)
                VDz[interval - 1].append(VDzi[j]*mp)
                VDxz_t[interval - 1].append(VDxzi[j]*mp)
                MD[interval - 1] += mp / (5 / N_inter) / A
                MpD[interval - 1] += mp
                D_vector_t[interval-1].append([VDxzi[j], ThetaDi[j], ID_Dbefore[j], xDi[j], i, mp])
            for j, dix in enumerate(ID_Di_new):
                interval = min(int(np.ceil((idx - 1) / (len(X[:, i]) / N_inter))), N_inter)
                VD_TD_vector_t[interval-1].append([VDxzi[j], abs(Vrepxi[j]), Trepi[j], Tdepi[j]])
                
            
        else:
            IDdepai = []
                
        EDindices.append((ID_Ei + 1, ID_Di - 1, np.array(IDdepai)))
    
    for i in range(N_inter):
        if VExz_t[i]:
            VExz_mean_t[i] = np.sum(VExz_t[i])/MpE[i]
        if VDxz_t[i]:
            VDxz_mean_t[i] = np.sum(VDxz_t[i])/MpD[i]      
            
    #VExz_mean_t, VEz_mean_t
    return EDindices, ME, MD, VExz_mean_t, VDxz_mean_t, D_vector_t, E_vector_t, VD_TD_vector_t

def output_id(e, thre_e, dt):
    ID_E, ID_D = [], []
    t = np.linspace(dt, 5, len(e))
    g = 9.81

    condition_indices = np.where(e <= thre_e)[0]
    segments = [] # static intervals

    if condition_indices.size > 0:
        start_idx = condition_indices[0]
        for i in range(1, len(condition_indices)):
            #the interval between every E and D should be long enough to be considered saltation
            if (t[condition_indices[i]] - t[condition_indices[i - 1]]) > np.sqrt((thre_e / g - 12 * 0.00025) * 2 / g)*2:
                end_idx = condition_indices[i - 1]
                #set an interval between every ID_D and ID_E to avoid counting rebounds
                # the particle has to be static over 0.02s between a deposition and an ejection
                #for now use 2*0.01s from the coarsest output steps
                if (t[end_idx] - t[start_idx]) > 0.02:
                    segments.append((start_idx, end_idx))
                start_idx = condition_indices[i]
        if (t[condition_indices[-1]] - t[start_idx]) > 0.02:
            segments.append((start_idx, condition_indices[-1]))
    
    if len(segments) > 1:
        ID_E = [seg[1] for seg in segments[:]]
        ID_D = [seg[0] for seg in segments[:]]
        #delete the ID_D if it appears at the first time step and the ID_E if it appears at the last time step
        ID_D = [id_d for id_d in ID_D if id_d > 1]
        ID_E = [id_e for id_e in ID_E if id_e < len(e)-1]

    return np.array(ID_E), np.array(ID_D)
