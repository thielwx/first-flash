#!/usr/bin/env python
# coding: utf-8

# ====================================================
# This script takes in the manual analysis dataset and captures the data in a gridded format with
# ABI, MRMS, GLM, and mesoA (eventaully) added
# Author: Kevin Thiel
# Created: November 2024
# ====================================================

# Process:
# Create a lat/lon grid based on the manual analysis domains (0.2 degrees)
#   Output should be two 1-D arrays
# Get a list of the ABI/MRMS/mesoA/GLM files that correspond to the glm first flashes
#   Get the unique listing of the files so you only have to load the data once
# Loop through the unique times and extract the data from each time/point in the grid
# Save all data into an accumualted dataframe and save out by case


import pandas as pd
import numpy as np
import netCDF4 as nc
from glob import glob
from datetime import datetime
import multiprocessing as mp
import yaml
import ma_gridded_output_functions as mgr

#Constants
source_file = 'ff_v00.csv'
yaml_file = '../manual-analysis-v1/case-settings-manual-analysis.yaml'
output_loc = '/localdata/first-flash/data/ml-manual-analysis/gridded/'

abi_variables = ['CMIP','ACHA']
abi_variables_output = ['CMIP_min', 'CMIP_05', 'ACHA_max', 'ACHA_95']
mrms_variables = ['MergedReflectivityQCComposite','Reflectivity_-10C','ReflectivityAtLowestAltitude']
mrms_variables_output = ['MergedReflectivityQCComposite_max','MergedReflectivityQCComposite_95','Reflectivity_-10C_max','Reflectivity_-10C_95','ReflectivityAtLowestAltitude_max','ReflectivityAtLowestAltitude_95']
ma_variables = ['ma_category', 'ma_convective_core']
glm_variables = ['num_glm16_flashes']

#Getting the file save time
y, m, d, doy, hr, mi = mgr.datetime_converter(datetime.now())
fsave_str = 'ma-grids-v1-ABI-MRMS-GLM-'+y+m+d+hr+mi+'.csv'

#Loading in the yaml file with the case settings
with open(yaml_file, 'r') as f:
    sfile = yaml.safe_load(f)

#Getting the list of cases
cases = sfile['cases']

#Reading in the manually analyzed first flash dataset 
df = pd.read_csv(source_file)
df = df.loc[df['ma_category']!='Miss'] #Subsetting for ONLY the hits/lates (ID'd as first flashes)
df['file_datetime'] = [datetime.strptime(row['fstart'][0:15],'s%Y%j%H%M%S0') for index, row in df.iterrows()]

#The driver function that controls the output for each case
def driver_function(case):
    global sfile
    global df
    dx = 0.2 #Grid resolution (degrees)
    #Subsetting the data for the case
    case_df = df.loc[df['case']==case]

    #Creating a list of times per first flash to use for a dataframe
    file_timestamp, file_times_abi, file_times_mrms = mgr.time_list_creator(df['file_datetime'].to_list())
    
    #Creating the lat/lon grid for the case
    grid_lats, grid_lons = mgr.grid_maker(case, sfile, 0.2)

    #Creating the gridded output dataframe that we'll fill with the ABI and MRMS data
    grid_df = mgr.df_creator(grid_lats, grid_lons, file_timestamp, case_df)
    mgr.df_saver(grid_df, output_loc, case, fsave_str)

    #Placing the first flashes on the grid
    grid_df = mgr.ff_driver(grid_df, case_df, file_timestamp) 
    mgr.df_saver(grid_df, output_loc, case, fsave_str)
    
    #Calling the functions that process the data for each step
    #case_df = mgr.abi_driver(case, case_df, grid_lats, grid_lons, file_times_abi)
    

    #mrms_driver()

    #glm_driver()


#Starting the multiprocessing by calling the driver function for each case
if __name__ == "__main__":
    with mp.Pool(12) as p:
        p.starmap(driver_function,zip(cases[0:1]))
        p.close()
        p.join()


