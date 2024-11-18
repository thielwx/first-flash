#!/usr/bin/env python
# coding: utf-8

# ====================================================
# This script contains the functions used by ma-gridded-output.py
# Author: Kevin Thiel
# Created: November 2024
# ====================================================

import yaml
import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import sys
import multiprocessing as mp
import os
from glob import glob

#====================================================================
# GENERAL USE FUNCTIONS
#====================================================================

def grid_maker(case, sfile, dx):
    '''
    This function creates the grid that we're projecting the data to
    PARAMS:
        case: Name of the current case
        sfile: The settings file
        dx: Resolution of the dataset (should be 0.2 degrees)
    RETURNS:
        lats_output: 1-D array of lats in grid (all combinations)
        lons_output: 1-D array of lons in grid (all combinations)
    '''
    lat_max = 50
    lat_min = 24
    lon_max = -66
    lon_min = -125

    #Creating the list of lats and lons. Putting them into a grid
    lats = np.arange(lat_min, lat_max+dx, dx)
    lons = np.arange(lon_min, lon_max+dx, dx)
    lat_grid, lon_grid = np.meshgrid(lats,lons)

    #Flattening the grid back down to 1D arrays
    lats_flat = lat_grid.flatten()
    lons_flat = lon_grid.flatten()

    #Finding the grid points within the case domains and subsetting the flattened arrays
    idx = np.where((lats_flat>=sfile[case]['lr_lat'])&(lats_flat<=sfile[case]['ul_lat'])&
                   (lons_flat>=sfile[case]['lr_lon'])&(lons_flat<=sfile[case]['ul_lon']))[0]
    lats_output = lats_flat[idx]
    lons_output = lons_flat[idx]

    return lats_output, lons_output


def datetime_converter(time):
    '''
    This function takes in a datetime object and returns strings of time features
    PARAMS:
        time: input time (datetime object)
    RETURNS:
        y: year (4 digit number as str)
        doy: day of year (3 digit number as str)
        hr: hour of day (2 digit number as str)
        mi: minute of hour (2 digit number as str)
        d: day of month (2 digit number as str)
        m: month of year (2 digit number as str)
    '''
    y = datetime.strftime(time,'%Y') #Year
    doy = datetime.strftime(time,'%j') #Day of year
    hr = datetime.strftime(time,'%H') #Hour
    mi = datetime.strftime(time,'%M') #Minute
    d = datetime.strftime(time,'%d') #Day of month
    m = datetime.strftime(time,'%m') #Month
    
    return y, m, d, doy, hr, mi


def time_list_creator(f_datetime):
    '''
    Creates the list of times that we'll use to pull the abi/mrms/mesoA data from
    PARAMS:
        f_datetime: DateTime subset to the case as list
    RETURNS:
        file_times_abi: List of file times as strings for abi
        file_times_mrms: List of file times as strings for mrms
    '''
    #==== Timestamp ====
    #Getting the time difference from the 0 and 5 ones place
    dt_int = np.array([int(t.strftime('%M'))%5 for t in f_datetime])
    #Getting the target file times from the most recent ABI file
    file_timestamp = [(f_datetime[i] - timedelta(minutes=int(dt_int[i]))).strftime('s%Y%j%H%M') for i in range(len(f_datetime))]

    #==== ABI ====
    #Getting the time difference from the 0 and 5 ones place
    dt_int = np.array([int(t.strftime('%M'))%5 for t in f_datetime])
    #Changing those minutes that are on the 0 and 5 ones place so they're five minutes away
    dt_int[dt_int==0] = 5
    #Getting the target file times from the most recent ABI file
    file_times_abi = [(f_datetime[i] - timedelta(minutes=int(dt_int[i])-1)).strftime('s%Y%j%H%M') for i in range(len(f_datetime))]

    #==== MRMS ====
    #Getting the time difference from the 10s place
    dt_int = [int(t.strftime('%M'))%10 for t in f_datetime]
    #Changing those minutes greater than 5 so they default to 6 rather than 0
    dt_int[dt_int>=6] = dt_int[dt_int>=6] % 6
    #Getting the target file times from the most recent MRMS file
    file_times_mrms = [(f_datetime[i] - timedelta(minutes=int(dt_int[i]))).strftime('s%Y%j%H%M') for i in range(len(f_datetime))]

    return file_timestamp, file_times_abi, file_times_mrms

def df_creator(grid_lats, grid_lons, file_timestamp, case_df):
    '''
    Creates a dataframe for all unique timestamps with the available grid lats and lons
    PARAMS:
        grid_lats: Array of lats from all grid lat/lon combinations (1D)
        grid_lons: Array of lons from all grid lat/lon combinations (1D)
        file_timestamp: Array of file timestamps (0s/5s) for each GLM first flash (strings)
        case_df: Dataframe of all first flashes in the case
    RETURNS:
        grid_df
    '''
    #Getting the number of unique timestamps to pull from
    ts_unique = np.unique(file_timestamp)

    #Creating looped versions of the lats/lons and case names for each unique timestamp
    grid_lats_looped = [grid_lats for i in range(len(ts_unique))]
    grid_lons_looped = [grid_lons for i in range(len(ts_unique))]
    case_looped = [case_df['case'].to_list() for i in range(len(ts_unique))]

    #Creating looped versions of the timestamps that match each lat/lon combination
    ts_looped = []
    for ts in ts_unique:
        ts_new_loop = [ts for i in range(len(grid_lats))]
        ts_looped.append(ts_new_loop)

    #Creating the dataframe which will hold the gridded data
    d = {
        'lat': grid_lats_looped,
        'lon': grid_lons_looped,
        'timestamp': ts_looped,
        'case': case_looped
    }
    grid_df = pd.DataFrame(data=d)
    
    return grid_df

def df_saver(df, output_loc, case, fsave_str):
    '''
    Function saves the dataframe as a csv
    '''
    save_loc = output_loc + '/' + case
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    df.to_csv(save_loc+fsave_str)

#====================================================================
# FIRST FLASH FUNCTIONS
#====================================================================

def ff_driver(grid_df, case_df, file_timestamp):
    '''
    Takes all of the first flashes and places them on the target grid
    PARAMS:
        grid_df
        case_df
        file_timestamp
    RETURNS:
        grid_df
    '''
    dx = 0.1
    #Adding new column to grid dataframe
    grid_df['ff_point'] = pd.Series(data=(np.ones(grid_df.shape[0]) * 0), dtype=int)
    grid_df['ff_fistart_flid'] = pd.Series(dtype=str)

    #Looping through the first flashes in the case and placing them into the dataframe
    for index, row in case_df.iterrows():
        cur_tstamp = file_timestamp[index]
        cur_lat = row['lat']
        cur_lon = row['lon']
        cur_fistart_flid = row['fistart_flid']

        #Finding which index has a matching timestamp and is closest to the current point. 
        idx = np.where(grid_df['timestamp']==cur_tstamp,
                 grid_df['lat']>=cur_lat-dx,
                 grid_df['lat']<cur_lat+dx,
                 grid_df['lon']>=cur_lon-dx,
                 grid_df['lon']<cur_lon+dx)[0]
        
        if len(idx) == 0:
            print ('ERROR: FIRST FLASH NOT PLACED ON GRID')
            print ('---'+cur_fistart_flid+'---')
            continue
        elif len(idx) == 1:
            grid_df[idx[0],'ff_point'] = 1
            grid_df[idx[0],'ff_fistart_flid'] = cur_fistart_flid
        else:
            print ('ERROR: MORE THAN ONE POINT COINCIDES WITH FIRST FLASH')
            print ('---'+cur_fistart_flid+'---')
            continue
    
    return grid_df


#====================================================================
# ABI FUNCTIONS
#====================================================================


#====================================================================
# MRMS FUNCTIONS
#====================================================================


#====================================================================
# GLM FUNCTIONS
#====================================================================