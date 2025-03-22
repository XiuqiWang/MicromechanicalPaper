# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:42:52 2025

@author: WangX3
"""
import numpy as np

def AverageOverShields(Values):
    Shields_means = []  # 用于存储均值
    Shields_stds = []   # 用于存储标准差

# 遍历每组子数列
    for group_start in range(5):  # 每组 5 个数列
    # 取出每组索引的子数列
        group_indices = list(range(group_start, 25, 5))  # 如 [0, 5, 10, 15, 20]
        group_values = np.array([Values[i] for i in group_indices])  # 提取对应子数列
    
    # 计算每组的均值和标准差
        group_mean = np.nanmean(group_values, axis=0)
        group_std = np.nanstd(group_values, axis=0)
    
    # 存储结果
        Shields_means.append(group_mean)
        Shields_stds.append(group_std)
    
    Shields_means = np.array(Shields_means)
    Shields_stds = np.array(Shields_stds)
    
    return Shields_means, Shields_stds