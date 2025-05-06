# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:31:52 2024

@author: WangX3
"""
import numpy as np
import math

def store_sal_id_data(data, ID_Particle, coe_h, dt, N_inter, D):
    X = np.array([[time_step['Position'][i][0] for i in ID_Particle] for time_step in data])
    Z = np.array([[time_step['Position'][i][2] for i in ID_Particle] for time_step in data])
    VXstored = np.array([[time_step['Velocity'][i][0] for i in ID_Particle] for time_step in data])
    VZstored = np.array([[time_step['Velocity'][i][2] for i in ID_Particle] for time_step in data])
    Rp = np.array([[time_step['Radius'][i] for i in ID_Particle] for time_step in data])
    Vp = (4/3)*math.pi*Rp**3 
    
    Par = []
    VX = []
    VZ = []
    E = []
    ez = []
    exz = []
    Vxr = []
    Vxi = []
    ez_t = [[] for _ in range(N_inter)]
    exz_t = [[] for _ in range(N_inter)]
    Vim_t = [[] for _ in range(N_inter)]
    Thetaim_t = [[] for _ in range(N_inter)]
    exz_vector_t,IM_vector_t = [[] for _ in range(N_inter)],[[] for _ in range(N_inter)]
    exz_mean_t,ez_mean_t = np.full(N_inter, np.nan), np.full(N_inter, np.nan)
    Vim_mean_t, Thetaim_mean_t = np.full(N_inter, np.nan), np.full(N_inter, np.nan)
    RIM,Mp = np.zeros(N_inter),np.zeros(N_inter)
    
    g = 9.81
    Lx = D * 100
    Ly = 2 * D
    A = Lx * Ly
    
    for i in range(len(ID_Particle)):
        z = Z[:, i]
        x = X[:,i]
        Vx = VXstored[:,i]
        Vz = VZstored[:,i]
        # Calculate average particle velocities from position differences/dt
        # Vz = np.concatenate(([0], (Z[1:, i] - Z[:-1, i]) / dt))
        # Vx = np.concatenate(([0], (X[1:, i] - X[:-1, i]) / dt))
        
        # Correct the Vx when a saltating particle crosses the boundary in positive direction
        # Index_neg = np.where(Vx < -Lx * 0.1 / dt)[0]
        # F_amp = np.ceil(0.5 * (VXstored[Index_neg, i] + VXstored[Index_neg, i]) * dt / Lx)
        # Vx[Index_neg] = Vx[Index_neg] + F_amp * Lx / dt
        
        ke = 0.5 * Vz**2
        pe = g * z
        e = ke + pe
        d_h = coe_h * D
        thre_e = g * d_h
        mp = Vp[0, i] * 2650
        
        # Call findSaltationID function
        Moms_coli, IDmobile_vec, IDvzri_vec, ezi, exzi, Vi_i, Vr_i, dxim_i, theta_i, thetare_i = findSaltationID(e, Vx, Vz, z, thre_e, dt)
        
        # Collect the global vectors
        Par.append([Moms_coli, IDmobile_vec, IDvzri_vec])
        VX.append(Vx)
        VZ.append(Vz)
        E.append(e)
        ez.extend(ezi)
        exz.extend(exzi)
        #Vxi.extend(Vi_i)
        #Vxr.extend(Vxr_i)
        
        if IDvzri_vec.size > 0:
            IDim_i = IDvzri_vec[:,0]
            IDre_i = IDvzri_vec[:,1]
            xim_i = x[IDim_i]
            xre_i = x[IDre_i]
            x_col = xim_i + dxim_i
            Eim_i = 0.5*mp*Vi_i**2 #kinetic energy 
            # Time-varying: determine which interval the values should go
            for j in range(IDvzri_vec.shape[0]):
                idx = int(np.ceil((IDvzri_vec[j, 1] + 1) / (X.shape[0] / N_inter)))
                idx = min(idx, N_inter)  # Prevent overflow
                ez_t[idx - 1].append(ezi[j]*mp)
                exz_t[idx - 1].append(exzi[j])
                Vim_t[idx - 1].append(Vi_i[j]*mp)
                Thetaim_t[idx - 1].append(theta_i[j]*mp)
                Mp[idx - 1] += mp #for counting the sum of particles' masses
                RIM[idx - 1] += 1 / (5 / N_inter)
                exz_vector_t[idx - 1].append(exzi[j])
                IM_vector_t[idx - 1].append([Vi_i[j], IDim_i[j], IDre_i[j], xim_i[j], xre_i[j], x_col[j], i, theta_i[j], thetare_i[j], Eim_i[j]])
    
    #mass-weighted avergae values in each output time step
    for i in range(N_inter):
        if exz_t[i]:
            exz_mean_t[i] = np.mean(exz_t[i])
        if ez_t[i]:
            ez_mean_t[i] = np.sum(ez_t[i])/Mp[i]
        if Vim_t[i]:
            Vim_mean_t[i] = np.sum(Vim_t[i])/Mp[i]
        if Thetaim_t[i]:
            Thetaim_mean_t[i] = np.sum(Thetaim_t[i])/Mp[i]
            
    return Par, VZ, exz_mean_t, ez_mean_t, Vim_mean_t, Thetaim_mean_t, RIM, exz_vector_t,IM_vector_t#ez_t, exz_t, Vim_t, Thetaim_t #E

# Helper function 1: findSaltationID
def findSaltationID(e, Vxi, Vzi, Zi, thre_e, dt):
    Moments_col = []
    ez_vector = []
    exz_vector = []
    IDmobile = []
    IDvzri_vec = []
    Vi_vec,Vr_vec = [],[]
    dxim_vec = []
    Thetai_vec,Thetare_vec = [],[]
    
    IntervalMobile = OutputMobileInterval(e, thre_e)
    if len(IntervalMobile)>0:
        IDmobile.extend(IntervalMobile)
    t = np.linspace(dt, 5, int(5 / dt))
    h = 13.5*0.00025 #the height for evaluating velocities 13.5D
    
    for interval in IntervalMobile:
        Vz_sal = Vzi[interval[0]:interval[1]]
        ti = t[interval[0]:interval[1]]
        
        # Find highest point indices
        crossing_indices = np.where((Vz_sal[:-1] >= 0) & (Vz_sal[1:] < 0))[0]
        
        # Calculate crossing times
        ratioST = Vz_sal[crossing_indices] / (Vz_sal[crossing_indices + 1] - Vz_sal[crossing_indices])
        crossing_times = ti[crossing_indices] - dt * ratioST
        
        Moments_col.extend(crossing_times)
        
        # Locate the next IDs (global) of the crossing moments
        ID_next = (np.ceil(crossing_times / dt)-1).astype(int)
        Vxzi = np.sqrt(Vxi**2 + Vzi**2)
        IDvzri = Findez(ID_next, Vzi, Vxzi, Zi, thre_e)
        
        if len(IDvzri) > 0:
            Vz_im = Vzi[IDvzri[:, 0]]
            Vz_re,Vx_re = Vzi[IDvzri[:, 1]],Vxi[IDvzri[:, 1]]
            
            #calculate the horizontal travel distance of the impact particle before collision
            dh_im = Zi[IDvzri[:,0]]-12*0.00025 #vertical distance between the impact particle and the bed surface
            dt_im = (np.sqrt(4*Vz_im**2+8*9.81*dh_im) - 2*Vz_im)/2/9.81
            dx_im = Vxi[IDvzri[:,0]]*dt_im
            
            #correct the vertical impact velocities to the values at the evaluation height 13.5D
            high_iindices = np.where(Zi[IDvzri[:,0]] > h)
            low_iindices = np.where(Zi[IDvzri[:,0]] < h)
            Vz_im[high_iindices] = -np.sqrt(2*9.81*(Zi[IDvzri[high_iindices,0]]-h)+Vzi[IDvzri[high_iindices,0]]**2)
            Vz_im[low_iindices] = -np.sqrt(Vzi[IDvzri[low_iindices,0]]**2 - 2*9.81*(h-Zi[IDvzri[low_iindices,0]]))          
            
            #correct the vertical rebound velocities
            high_rindices = np.where(Zi[IDvzri[:,1]] > h)
            low_rindices = np.where(Zi[IDvzri[:,1]] < h)
            Vz_re[high_rindices] = np.sqrt(2*9.81*(Zi[IDvzri[high_rindices,1]]-h)+Vzi[IDvzri[high_rindices,1]]**2)
            Vz_re[low_rindices] = np.sqrt(Vzi[IDvzri[low_rindices,1]]**2 - 2*9.81*(h-Zi[IDvzri[low_rindices,1]]))
            #correct the horizontal rebound velocities
            Vz1_low,Vx1_low,Vx2_low = Vzi[IDvzri[low_rindices,1]],Vxi[IDvzri[low_rindices,1]],Vxi[IDvzri[low_rindices,1]+1]
            z1_low = Zi[IDvzri[low_rindices,1]]
            dt_ratio_low = (2*Vz1_low + np.sqrt(4*Vz1_low**2 - 4*9.81*(2*h-2*z1_low))) / (2*9.81) /dt
            newlow_rindices = np.where(dt_ratio_low < 1)
            if len(newlow_rindices[0]) > 0:
                Vx_re[low_rindices[0][newlow_rindices[0]]] = dt_ratio_low[newlow_rindices[0]] * (Vx2_low[newlow_rindices[0]]-Vx1_low[newlow_rindices[0]]) + Vx1_low[newlow_rindices[0]]
            #higher than 1.5D
            Vx0_high,Vx1_high = Vxi[IDvzri[high_rindices,1]-1],Vxi[IDvzri[high_rindices,1]]
            Vz1_high = Vzi[IDvzri[high_rindices,1]]
            z1_high = Zi[IDvzri[high_rindices,1]]
            dt_ratio_high = (2*Vz1_high + np.sqrt(4*Vz1_high**2 - 4*9.81*(2*h-2*z1_high))) / (2*9.81) /dt
            newhigh_rindices = np.where(dt_ratio_high < 1)
            if len(newhigh_rindices[0]) > 0:
                Vx_re[high_rindices[0][newhigh_rindices[0]]] = dt_ratio_high[newhigh_rindices[0]] * (Vx0_high[newhigh_rindices[0]]-Vx1_high[newhigh_rindices[0]]) + Vx1_high[newhigh_rindices[0]]
            
            ez = np.abs(Vz_re/Vz_im)#np.abs(Vzi[IDvzri[:, 1]] / Vzi[IDvzri[:, 0]])
            Vxzi_imnew = np.sqrt(Vxi[IDvzri[:, 0]]**2 + Vz_im**2)
            Vxzi_renew = np.sqrt(Vx_re**2 + Vz_re**2)
            exz = Vxzi_renew/Vxzi_imnew #Vxzi[IDvzri[:, 1]] / Vxzi[IDvzri[:, 0]]
            
            #impact angles
            Thetaii_radian = np.arctan(np.abs(Vz_im/Vxi[IDvzri[:, 0]]))
            Thetaii = np.degrees(Thetaii_radian)
            #rebound angles
            Thetarei_radian = np.arctan(np.abs(Vz_re/Vx_re))
            Thetarei = np.degrees(Thetarei_radian)
            
            #filter the exz, keep only the ones < 2
            # valid_indices = np.where(exz < 2)
            # exz = exz[valid_indices]
            # ez = ez[valid_indices]
            # IDvzri = IDvzri[valid_indices]
            # dx_im = dx_im[valid_indices]
            # Vxzi_imnew = Vxzi_imnew[valid_indices]
            # Vxzi_renew = Vxzi_renew[valid_indices]
            # Thetaii = Thetaii[valid_indices]
            
            ez_vector.extend(ez)
            exz_vector.extend(exz)
            IDvzri_vec.extend(IDvzri)
            Vi_vec.extend(Vxzi_imnew) # Vxzi[IDvzri[:, 0]]#impact velocity and angle are corrected at 1.5D height
            Vr_vec.extend(Vxzi_renew)
            dxim_vec.extend(dx_im)
            Thetai_vec.extend(Thetaii)
            Thetare_vec.extend(Thetarei)
    
    return Moments_col, np.array(IDmobile), np.array(IDvzri_vec), np.array(ez_vector), np.array(exz_vector), np.array(Vi_vec), np.array(Vr_vec), np.array(dxim_vec), np.array(Thetai_vec), np.array(Thetare_vec)

# Helper function 2: OutputMobileInterval
def OutputMobileInterval(e, thre_e):
    t = np.linspace(5 / len(e), 5, len(e))
    condition_indices = np.where(e > thre_e)[0]
    segments = []
    
    if len(condition_indices) > 0:
        start_idx = condition_indices[0]
        for i in range(1, len(condition_indices)):
            #extract the continous intervals
            if condition_indices[i] != condition_indices[i-1] + 1:#t[condition_indices[i]] - t[condition_indices[i - 1]] > 0.03:
                end_idx = condition_indices[i - 1]
                if t[end_idx] - t[start_idx] > np.sqrt((thre_e/9.81 - 12 * 0.00025) * 2 / 9.81)*2:
                    segments.append([start_idx, end_idx])
                start_idx = condition_indices[i]
        
        if t[condition_indices[-1]] - t[start_idx] > 0.03:
            segments.append([start_idx, condition_indices[-1]])
    
    return np.array(segments)

# Helper function 3: Findez
def Findez(ID_next, Vzi, Vxzi, Zi, thre_e):
    IDvzr = []
    IDvzi = []
    IDvzri = []
    
    #search for the maximum Vz and minimum Vz to calculate the COR
    for i in range(len(ID_next)-1):
        ID_range = np.linspace(ID_next[i]-1, ID_next[i + 1], ID_next[i + 1] - ID_next[i] + 2).astype(int)
        
        #print('ID_range:',ID_range)
        if len(ID_range) > 2:
            local_maxima = np.logical_and(Vzi[ID_range[1:-1]] > Vzi[ID_range[:-2]], Vzi[ID_range[1:-1]] > Vzi[ID_range[2:]])
            local_minima = np.logical_and(Vzi[ID_range[1:-1]] < Vzi[ID_range[:-2]], Vzi[ID_range[1:-1]] < Vzi[ID_range[2:]])
            #make sure maxima and minima are equal-size (empty or with only 1 value)
            if np.sum(local_maxima) > 1:
                true_indices = np.where(local_maxima)[0]
                max_index = true_indices[np.argmax(Vzi[ID_range[true_indices+1]])]#np.argmin(ID_range[true_indices+1])] # Keep only the index correponding to the highest value in Vzi
                local_maxima[:] = False
                local_maxima[max_index] = True
            # Check and reduce local_minima to one element
            if np.sum(local_minima) > 1:
                true_indices = np.where(local_minima)[0]
                min_index = true_indices[np.argmin(Vzi[ID_range[true_indices+1]])]#np.argmax(ID_range[true_indices+1])] # Keep only the index correponding to the lowest value in Vzi
                local_minima[:] = False
                local_minima[min_index] = True
            if np.sum(local_maxima) == 0 or np.sum(local_minima) == 0:
                local_maxima[:] = False # Set to all false
                local_minima[:] = False  # Set to all false
          
            IDvzr.extend(ID_range[np.where(local_maxima)[0]+1])
            IDvzi.extend(ID_range[np.where(local_minima)[0]+1])
    
    # print('IDvzi:',IDvzi)
    # print('IDvzr:',IDvzr)
    
    IDvzri = np.array([IDvzi, IDvzr]).T
        
    if IDvzri.size > 0:
        IDvzri = IDvzri[Vzi[IDvzri[:, 0]] < 0]#impact velocity should be negative
        #print('Vzr:',Vzi[IDvzri[:, 1]])
        IDvzri = IDvzri[Vzi[IDvzri[:, 1]] > 0]#rebound velocity should be positive
        vz_crit = np.sqrt((thre_e/9.81 - 12 * 0.00025) * 2 * 9.81)
        #the rebound velocity should be high enough to reach (coe_h-12)D
        IDvzri = IDvzri[Vzi[IDvzri[:, 1]] > vz_crit]
        #the impact should be lower than 20D to be close to the bed, for excluding mid-air collision
        IDvzri = IDvzri[Zi[IDvzri[:, 0]] <= 42*0.00025]
        # IDvzri = IDvzri[Zi[IDvzri[:, 1]] <= 50*0.00025]
        #exclude the ones with exz > 1
        # exz = Vxzi[IDvzri[:, 1]] / Vxzi[IDvzri[:, 0]]
        # IDvzri = IDvzri[exz < 1]
    
    return IDvzri
