# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:50:19 2024

@author: WangX3
"""

import pickle
import pandas as pd
import numpy as np
from module import read_data

# data0 = read_data('S002DryLBIni.data', 14, (1, 6)) 
# data1 = read_data('S003DryLBIni.data', 14, (1, 6)) 
# data2 = read_data('S004DryLBIni.data', 14, (1, 6)) 
data0 = read_data('S003DryLBModelTP.data', 14, (1, 6))
data1 = read_data('S003M1LBModelTP.data', 14, (1.5, 6.5))
data2 = read_data('S003M5LBModelTP.data', 14, (1.5, 6.5))
data3 = read_data('S003M10LBModelTP.data', 14, (1.5, 6.5))
data4 = read_data('S003M20LBModelTP.data', 14, (1.5, 6.5))

data_list = [data0, data1, data2, data3, data4]

# Save each data element to a separate pickle file
for i, data in enumerate(data_list):
    file_name = f"dataLBM{3}_{i}.pkl"  # Create file names like data_0.pkl, data_1.pkl, etc.
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {file_name}")


