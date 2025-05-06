# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:00:10 2025

@author: WangX3
"""

import numpy as np
import matplotlib.pyplot as plt

U      = [1,5,6,8]
NE     = [2,4,8,9]
Weight = [3,2,1,1]

def our_fit(x):
    return 1+1.1*x

def weighted_mean(NE,Weight):
    weight_sum = sum(Weight)
    weighted_sum = sum(np.array(Weight)*np.array(NE))
    weighted_mean = weighted_sum/weight_sum
    return weighted_mean

fig,ax = plt.subplots()
ax.plot(U,NE,'k.') 
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

x_vec = np.linspace(0,8,100)
ax.plot(x_vec,our_fit(x_vec),'r-')
ax.plot(x_vec,weighted_mean(NE,Weight)*np.ones_like(x_vec),'b-')

def calc_SSres(NE,U,Weight):
    SSres = 0
    for n in range(len(U)):
        SSres += Weight[n]*( NE[n]-our_fit(U[n]) ) **2
    return SSres

def calc_SStot(NE,U,Weight):
    SStot = 0
    Weighted_mean = weighted_mean(NE,Weight)
    for n in range(len(U)):
        SStot += Weight[n]*( NE[n]-Weighted_mean ) **2
    return SStot

SSres = calc_SSres(NE,U,Weight)
SStot = calc_SStot(NE,U,Weight)

R2 = 1- SSres/SStot

print('R2='+str(R2))
