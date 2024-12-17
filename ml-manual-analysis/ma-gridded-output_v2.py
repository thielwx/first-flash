#!/usr/bin/env python
# coding: utf-8

# ====================================================
# This script takes in the manual analysis dataset and captures the data in a gridded format with
# ABI, MRMS, GLM, and mesoA (eventaully) added
# Author: Kevin Thiel
# Created: November 2024
# 20241216: Update made to include next twenty minutes of GLM flashes
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
from datetime import datetime, timedelta
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
dt_now = datetime.now()
y, m, d, doy, hr, mi = mgr.datetime_converter(dt_now)
fsave_str = 'ma-grids-v3-ABI-MRMS-GLM-'+y+m+d+hr+mi+'.csv'

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
    dt = 15 #Temporal range that the hit is applied (minutes)
    #Subsetting the data for the case
    case_df = df.loc[df['case']==case]

    #Getting a list of times using the case start (-15) and end times
    start_dt = datetime.strptime(sfile[case]['start_time'],'%Y%m%d-%H%M')
    end_dt = datetime.strptime(sfile[case]['end_time'],'%Y%m%d-%H%M')
    times_dt = pd.date_range(start=start_dt-timedelta(minutes=dt), end=end_dt, freq='5min').to_list()

    #Creating a list of times per first flash to use for a dataframe
    file_timestamp, file_times_abi, file_times_mrms = mgr.time_list_creator(times_dt)
    file_timestamp_ff, file_times_abi_ff, file_times_mrms_ff = mgr.time_list_creator(case_df['file_datetime'].to_list())

    #Assigning the timestamps and their datetime objects to the first flashes
    case_df['file_timestamp'] = file_timestamp_ff
    case_df['file_timestamp_datetime'] = [datetime.strptime(t, '%Y%m%d-%H%M') for t in file_timestamp_ff]

    #Creating the lat/lon grid for the case
    grid_lats, grid_lons = mgr.grid_maker(case, sfile, dx)

    #Creating the gridded output dataframe that we'll fill with the ABI and MRMS data
    grid_df = mgr.df_creator(grid_lats, grid_lons, file_timestamp, case)
    mgr.df_saver(grid_df, output_loc, case, case+'-'+fsave_str)

    #Placing the first flashes on the grid
    grid_df = mgr.ff_driver_v2(grid_df, case_df, file_timestamp) 
    mgr.df_saver(grid_df, output_loc, case, case+'-'+fsave_str)
    print (str(case)+': First Flashes Collected')

    #Calling the functions that process the data for each step
    #abi_driver
    grid_df = mgr.abi_driver(grid_df, file_timestamp, file_times_abi, grid_lats, grid_lons)
    mgr.df_saver(grid_df, output_loc, case, case+'-'+fsave_str)
    print (str(case)+': ABI Data Collected')

    #mrms_driver()
    grid_df = mgr.mrms_driver(grid_df, file_timestamp, file_times_mrms, grid_lats, grid_lons, mrms_variables)
    mgr.df_saver(grid_df, output_loc, case, case+'-'+fsave_str)
    print (str(case)+': MRMS Data Collected')

    #glm_driver()
    grid_df = mgr.glm_driver(grid_df, file_timestamp, grid_lats, grid_lons, sfile[case]['glm16_all'])
    mgr.df_saver(grid_df, output_loc, case, case+'-'+fsave_str)
    print (str(case)+': GLM Data Collected')


#Starting the multiprocessing by calling the driver function for each case
if __name__ == "__main__":
    with mp.Pool(12) as p:
        p.starmap(driver_function,zip(cases[:]))
        p.close()
        p.join()

print ('Total Runtime: '+ str(datetime.now()-dt_now))

