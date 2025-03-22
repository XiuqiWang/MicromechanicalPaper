# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:35:00 2024

@author: WangX3
"""
# module/read_data.py


import os
import numpy as np

def read_data(filename=None, format=14, time=(-np.inf, np.inf)):
    """
    Reads data from a file with the specified format and time range.

    Parameters:
        filename (str): Name of the file to read. If None, the first .data file in the directory is used.
        format (int): Format specifier. Defaults to 14.
        time (tuple): Time range as (min_time, max_time). Defaults to (-inf, inf).

    Returns:
        list: A list of dictionaries containing the parsed data.
    """
    # If no filename is given, pick the first .data file in the directory
    if filename is None:
        files = sorted([f for f in os.listdir('.') if f.endswith('.data')], key=os.path.getmtime)
        if not files:
            raise FileNotFoundError("No .data files found in the directory.")
        filename = files[0]
        print(f"Using file: {filename}")

    # Initialize data list
    data = []

    # Open the file
    with open(filename, 'r') as fid:
        while True:
            if format == 14:
                # Read the header line
                header = fid.readline().strip().split()
                if not header:
                    break  # End of file

                # Parse the header
                header = [float(x) for x in header]
                if len(header) < 8:
                    break  # Malformed header
                
                # Read the raw data block
                n = int(header[0])
                rawdata = []
                for _ in range(n):
                    line = fid.readline().strip().split()
                    if len(line) != 14:
                        raise ValueError("Malformed data row.")
                    rawdata.append([float(x) for x in line])

                rawdata = np.array(rawdata)

                # Write it into a dictionary
                data1 = {
                    'N': header[0],
                    't': header[1],
                    'xmin': header[2],
                    'ymin': header[3],
                    'zmin': header[4],
                    'xmax': header[5],
                    'ymax': header[6],
                    'zmax': header[7],
                    'Position': rawdata[:, 0:3],
                    'Velocity': rawdata[:, 3:6],
                    'Radius': rawdata[:, 6],
                    'Angle': rawdata[:, 7:10],
                    'AngularVelocity': rawdata[:, 10:13],
                    'info': rawdata[:, 13]
                }

                # Filter based on time range
                if time[0] <= data1['t'] <= time[1]:
                    data.append(data1)
                    print(f"Time: {data1['t']}")

    return data
