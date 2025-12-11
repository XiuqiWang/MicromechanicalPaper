# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:32:04 2025

@author: WangX3
"""
import numpy as np
import itertools

def match_ejection_to_impactanddeposition(impact_deposition_list, ejection_list):
    ejection_ids, ejection_positions, ejection_velocities, ejectionpar_ids, ejection_energy = ejection_list
    incidence_ids, incidence_positions, VDs, depositionpar_ids, Thetaincidence = impact_deposition_list
    
    incidence_positions = np.array(incidence_positions)
    incidence_ids = np.array(incidence_ids)
    VDs = np.array(VDs)
    Xmax = 100*0.00025
    thres_pos = 5*0.00025
    
    # initialize the result list to be equally long as the impact_list
    result = [[[], [], []] for _ in range(len(incidence_ids))]
    
    # loop over every ejection
    for ejection_id, ejection_pos, ejection_vel, epar_id, ejection_ene in zip(ejection_ids, ejection_positions, ejection_velocities, ejectionpar_ids, ejection_energy):
        mask = (
            (depositionpar_ids != epar_id) &
            (incidence_ids <= ejection_id) 
            # (ejection_id <= incidence_ids+2) &
            # (np.abs(ejection_pos - incidence_positions) <= thres_pos)
    )
        valid_indices = np.where(mask)[0]
        #if there are found impacts
        if len(valid_indices) > 0:
            # 计算所有位置差值
            position_differences = np.zeros(len(valid_indices))
            for i in range(len(valid_indices)):
                position_differences[i] = np.abs(ejection_pos - incidence_positions[valid_indices[i]])
            # 找到最小距离
            min_pos_diff = np.min(position_differences)
            # 获取所有具有相同最小空间距离的索引
            min_pos_indices = np.where(position_differences == min_pos_diff)[0]

            # 如果有多个最小空间距离，进一步比较 ID 差值（即时间）
            if len(min_pos_indices) > 1:
                id_differences = ejection_id - np.array([incidence_ids[valid_indices[i]] for i in min_pos_indices])
                closest_index = valid_indices[min_pos_indices[np.argmin(id_differences)]]
            else:
                closest_index = valid_indices[min_pos_indices[0]]
                
            result[closest_index][0].append(ejection_id)
            result[closest_index][1].append(ejection_vel)
            result[closest_index][2].append(ejection_ene)
            # print('ejection_id:',ejection_id)
            # print('impact_id:& rebound_id:',impact_ids[closest_index],rebound_ids[closest_index])
            # print('ejection_pos:',ejection_pos/0.00025)
            # print('impact_pos:& rebound_pos:',impact_positions[closest_index]/0.00025,rebound_positions[closest_index]/0.00025)
            # print('impact particle id:', impactpar_ids[closest_index])
            # print('impact_id:& rebound_id:',impact_ids[valid_indices],rebound_ids[valid_indices])
            # print('position diff:',np.abs(ejection_pos - impact_positions[valid_indices]))
            # print('closest_index:',impact_ids[closest_index])
            # print('impact_positions[closest_index]', impact_positions[closest_index])
    
    impact_ejection_list = [[], [], [], [], []]
    impact_ejection_list[0].extend(VDs)
    impact_ejection_list[1].extend(Thetaincidence)
    for sublist in result:
        impact_ejection_list[2].append(len(sublist[0]))#NE
        impact_ejection_list[3].append(sublist[1])#UE
        impact_ejection_list[4].append(sublist[2])#EE
    
    return impact_ejection_list
    #     valid_indices = np.where(mask)[0]
        
    #     #if valid impacts are found, find the temporally closest impact 
    #     if len(valid_indices) > 0:
    #         # 计算 id 差值
    #         id_differences = ejection_id - deposition_ids[valid_indices]
    #         # 找到最小 id 差值
    #         min_id_diff = np.min(id_differences)
    #         # 获取所有具有相同最小 id 差值的索引
    #         min_id_indices = np.where(id_differences == min_id_diff)[0]

    #         # 如果有多个最小差值，进一步比较空间距离
    #         if len(min_id_indices) > 1:
    #             position_differences = np.zeros(len(min_id_indices))
    #             for i in range(len(min_id_indices)):
    #                 position_differences[i] = np.abs(ejection_pos - deposition_positions[valid_indices[i]])
                   
    #             closest_index = valid_indices[min_id_indices[np.argmin(position_differences)]]
    #         else:
    #             closest_index = valid_indices[min_id_indices[0]]  # 只有一个最小值时直接取该索引
                
    #         result[closest_index][0].append(ejection_id)
    #         result[closest_index][1].append(ejection_vel)
    #         result[closest_index][2].append(ejection_ene)
    
    # impact_ejection_list = [[], [], [], []]
    # impact_ejection_list[0].extend(VDs)
    # for sublist in result:
    #     impact_ejection_list[1].append(len(sublist[0]))#NE
    #     impact_ejection_list[2].append(sublist[1])#UE
    #     impact_ejection_list[3].append(sublist[2])#EE
    
    # return impact_ejection_list