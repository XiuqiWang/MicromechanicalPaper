# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:34:57 2025

@author: WangX3
"""
import numpy as np
import itertools

# def match_ejection_to_impact(Vim_all_i, impact_list, ejection_list, vel_bins,dt):#theta_all_i,
#     #impact_thetas = np.array(theta_all_i)
#     ejection_ids, ejection_positions, ejection_velocities, ejectionpar_ids = ejection_list
#     impact_ids, impact_positions, impactpar_ids = impact_list
    
#     impact_positions = np.array(impact_positions)
#     impact_ids = np.array(impact_ids)
#     # initialize the result list to be equally long as the impact_list
#     result = [[[], []] for _ in range(len(impact_ids))]
#     thre_pos = 3*0.00025
#     # loop over every ejection
#     for ejection_id, ejection_pos, ejection_vel, epar_id in zip(ejection_ids, ejection_positions, ejection_velocities, ejectionpar_ids):
#         #ejection_id = ejection_id -1#static moment of the ejection
#         # find valid impact indices
#         valid_indices = np.where(
#             (impactpar_ids != epar_id) &
#             ## impact ID should be smaller than or equal to ejection ID 
#             (impact_ids <= ejection_id) &
#             (ejection_id - impact_ids <= 0.02/dt) &
#             (np.abs(ejection_pos - impact_positions) < thre_pos)
#             )[0]

#         #if valid impacts are found, find the temporally closest impact 
#         if len(valid_indices) > 0:
#             # 计算 id 差值
#             id_differences = ejection_id - impact_ids[valid_indices]
#             # 找到最小 id 差值
#             min_id_diff = np.min(id_differences)
#             # 获取所有具有相同最小 id 差值的索引
#             min_id_indices = np.where(id_differences == min_id_diff)[0]

#             # 如果有多个最小差值，进一步比较空间距离
#             # print('ejection pos:',ejection_pos)
#             # print('impact id:',impact_ids[valid_indices[min_id_indices]])
#             # print('impact pos:',impact_positions[valid_indices[min_id_indices]])
#             if len(min_id_indices) > 1:
#                 position_differences = np.abs(ejection_pos - impact_positions[valid_indices[min_id_indices]])
#                 closest_index = valid_indices[min_id_indices[np.argmin(position_differences)]]
#             else:
#                 closest_index = valid_indices[min_id_indices[0]]  # 只有一个最小值时直接取该索引
#             result[closest_index][0].append(ejection_id)
#             result[closest_index][1].append(ejection_vel)
#             # print('ejection_id:',ejection_id)
#             # print('impact_id:',impact_ids[valid_indices])
#             # print('position diff:',np.abs(ejection_pos - impact_positions[valid_indices]))
#             # print('closest_index:',impact_ids[closest_index])
#             # print('impact_positions[closest_index]', impact_positions[closest_index])
            
#     Uplot, NE_i,VE_i,VE_std_i = get_ejection_ratios(np.array(Vim_all_i), result, vel_bins)
            
#     return Uplot, NE_i,VE_i,VE_std_i,result



def match_ejection_to_impact(impact_list, ejection_list, dt):#theta_all_i,
    #impact_thetas = np.array(theta_all_i)
    ejection_ids, ejection_positions, ejection_velocities, ejectionpar_ids = ejection_list
    impact_ids, rebound_ids, impact_positions, rebound_positions, Vimpacts, collision_positions, impactpar_ids, Thetaimpacts = impact_list
    
    impact_positions = np.array(impact_positions)
    impact_ids = np.array(impact_ids)
    rebound_positions = np.array(rebound_positions)
    rebound_ids = np.array(rebound_ids)
    Vimpacts = np.array(Vimpacts)
    collision_positions = np.array(collision_positions)
    Thetaimpacts = np.array(Thetaimpacts)
    Xmax = 100*0.00025
    thres_pos = 7*0.00025
    #calculate how many times the impact particles crossed the domain before colliding
    Ncycle = np.floor(collision_positions/Xmax)
    collision_positions = collision_positions - Xmax*Ncycle
    # initialize the result list to be equally long as the impact_list
    result = [[[], []] for _ in range(len(impact_ids))]
    
    # loop over every ejection
    for ejection_id, ejection_pos, ejection_vel, epar_id in zip(ejection_ids, ejection_positions, ejection_velocities, ejectionpar_ids):
        # find valid impact indices
        # valid_indices = np.where(
        #     (impactpar_ids != epar_id) &
        #     ## impact ID should be smaller than or equal to ejection ID 
        #     (impact_ids <= ejection_id) &
        #     (ejection_id - impact_ids <= 0.03/dt) &
        #     (((ejection_pos >= thre_pos) & (ejection_pos <= 100*0.00025-thre_pos) & (np.abs(ejection_pos - impact_positions) < thre_pos))
        #     |  
        #     ((ejection_pos < thre_pos) & (impact_positions < 50*0.00025) & (np.abs(ejection_pos - impact_positions) < thre_pos))
        #     |   
        #     ((ejection_pos < thre_pos) & (impact_positions >= 50*0.00025) & (np.abs(ejection_pos + 100 * 0.00025 - impact_positions) < thre_pos))
        #     |
        #     ((ejection_pos >= 100*0.00025-thre_pos) & (impact_positions > 50*0.00025) & (np.abs(ejection_pos - impact_positions) < thre_pos))
        #     |
        #     ((ejection_pos >= 100*0.00025-thre_pos) & (impact_positions <= 50*0.00025) & (np.abs(impact_positions + 100 * 0.00025 - ejection_pos) < thre_pos)))
        # )[0]
        mask = (
            (impactpar_ids != epar_id) &
            (impact_ids <= ejection_id) &
            (ejection_id <= rebound_ids) &
            # (ejection_id - impact_ids <= 0.01/dt) &
            # (ejection_pos >= collision_positions) &
            (np.abs(ejection_pos - collision_positions) <= thres_pos)
            # ((impact_positions <= ejection_pos) & (rebound_positions >= ejection_pos))
            # ( (impact_positions < rebound_positions) & (impact_positions <= ejection_pos) & (rebound_positions >= ejection_pos))
            # |
            # (((np.floor(L/Xmax) <= 1) & (impact_positions >= rebound_positions)) & ((ejection_pos >= impact_positions) | (ejection_pos < rebound_positions)))
            # | 
            # (((np.floor(L/Xmax) > 1)) & ((ejection_pos >= impact_positions) | (ejection_pos < rebound_positions))) 
    )
        valid_indices = np.where(mask)[0]
        
        #if valid impacts are found, find the temporally closest impact 
        if len(valid_indices) > 0:
            # 计算 id 差值
            id_differences = ejection_id - impact_ids[valid_indices]
            # 找到最小 id 差值
            min_id_diff = np.min(id_differences)
            # 获取所有具有相同最小 id 差值的索引
            min_id_indices = np.where(id_differences == min_id_diff)[0]

            # 如果有多个最小差值，进一步比较空间距离
            # print('ejection pos:',ejection_pos)
            # print('impact id:',impact_ids[valid_indices[min_id_indices]])
            # print('impact pos:',impact_positions[valid_indices[min_id_indices]])
            if len(min_id_indices) > 1:
                position_differences = np.zeros(len(min_id_indices))
                for i in range(len(min_id_indices)):
                    # if ((L[valid_indices[i]] < Xmax and impact_positions[valid_indices[i]]<rebound_positions[valid_indices[i]] and impact_positions[valid_indices[i]] < ejection_pos and ejection_pos < rebound_positions[valid_indices[i]])
                    # or (L[valid_indices[i]] <= Xmax and impact_positions[valid_indices[i]] >= rebound_positions[valid_indices[i]] and impact_positions[valid_indices[i]] <= ejection_pos)
                    # or (L[valid_indices[i]] > Xmax and impact_positions[valid_indices[i]] <= ejection_pos)):
                    position_differences[i] = np.abs(ejection_pos - collision_positions[valid_indices[i]])
                    # if (L[valid_indices[i]] <= Xmax and impact_positions[valid_indices[i]] >= rebound_positions[valid_indices[i]] and ejection_pos <= rebound_positions[valid_indices[i]])
                    # or (L[valid_indices[i]] > Xmax and ejection_pos < rebound_positions[valid_indices[i]]):
                    #     position_differences[i] = np.abs(ejection_pos + Xmax - impact_positions[valid_indices[i]])
                
                closest_index = valid_indices[min_id_indices[np.argmin(position_differences)]]
            else:
                closest_index = valid_indices[min_id_indices[0]]  # 只有一个最小值时直接取该索引
                
            result[closest_index][0].append(ejection_id)
            result[closest_index][1].append(ejection_vel)
            # print('ejection_id:',ejection_id)
            # print('impact_id:& rebound_id:',impact_ids[closest_index],rebound_ids[closest_index])
            # print('ejection_pos:',ejection_pos/0.00025)
            # print('impact_pos:& rebound_pos:',impact_positions[closest_index]/0.00025,rebound_positions[closest_index]/0.00025)
            # print('impact particle id:', impactpar_ids[closest_index])
            # print('impact_id:& rebound_id:',impact_ids[valid_indices],rebound_ids[valid_indices])
            # print('position diff:',np.abs(ejection_pos - impact_positions[valid_indices]))
            # print('closest_index:',impact_ids[closest_index])
            # print('impact_positions[closest_index]', impact_positions[closest_index])
    
    impact_ejection_list = [[], [], [], []]
    impact_ejection_list[0].extend(Vimpacts)
    impact_ejection_list[1].extend(Thetaimpacts)
    for sublist in result:
        impact_ejection_list[2].append(len(sublist[0]))
        impact_ejection_list[3].append(sublist[1])
    
    return impact_ejection_list


    ## 遍历每个ejection
    # for ej_id, ej_pos, ej_vel in zip(ejection_ids, ejection_positions, ejection_velocities):
        # closest_impact_id = None
        # closest_impact_idx = None
        # closest_distance = float('inf')#infinite
        # search_range = 4 * 0.00025  # Example range for distance constraint
        # # 遍历impact列表
        # for i, (imp_id, imp_pos) in enumerate(zip(impact_ids, impact_positions)):
        #     # 检查ID是否满足条件 (impact发生在ejection之前)
        #     if imp_id >= ej_id:
        #         break
        #     # 计算距离条件
        #     #if ej_pos >= 10 * 0.00025:
        #         # 正常情况：impact位置在ejection附近
        #         #if imp_pos < ej_pos:
        #     distance = np.abs(ej_pos - imp_pos)
        #         #else:
        #         #    continue
        #     #else:
        #         # 特殊情况：允许100D附近的impact
        #         # if imp_pos < 50*0.00025:
        #         #     distance = ej_pos - imp_pos
        #         # else:
        #         #     distance = ej_pos + 100 * 0.00025 - imp_pos
        #     if distance > search_range:
        #         break
        #     # 检查是否找到更近的impact
        #     if distance < closest_distance:
        #         closest_distance = distance
        #         closest_impact_id = imp_id
        #         closest_impact_idx = i
        # # 如果找到满足条件的impact，将ejection ID存入对应位置
        # if closest_impact_idx is not None:
        #     result[closest_impact_idx][0].append(ej_id)
        #     result[closest_impact_idx][1].append(ej_vel)
            
    #Find indices where theta_all_i is within the range 9.5 to 10.5
    # condition = (impact_thetas >= 9) & (impact_thetas <= 11)
    # filtered_ids = np.where(condition)[0]
    # # Filter both arrays
    # filtered_Vim = np.array(Vim_all_i)[filtered_ids]
    # result = [result[i] for i in filtered_ids]
    #print('filtered_Vim',filtered_Vim)
    #print('result',result)

#loop over the ejection list to find the nearest (temporally and spatially) impact for each ejection
    # convert list into NumPy array
    # ejection_ids = np.array(ejection_list[0])
    # ejection_positions = np.array(ejection_list[1])
    # ejection_velocities = np.array(ejection_list[2])
    # impact_ids = np.array(impact_list[0])
    # impact_positions = np.array(impact_list[1])
    
    # # Define thresholds
    # threshold_tem = 3
    # threshold = D * 10
    # result = [[[], []] for _ in range(len(impact_ids))]  # 初始化结果列表

    
    
   