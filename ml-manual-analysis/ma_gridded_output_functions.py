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
from sklearn.neighbors import BallTree
import satpy.modifiers.parallax as plax
from pyproj import Proj
from pyresample import SwathDefinition, kd_tree
import gzip

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
                   (lons_flat<=sfile[case]['lr_lon'])&(lons_flat>=sfile[case]['ul_lon']))[0]

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
    file_timestamp = [(f_datetime[i] - timedelta(minutes=int(dt_int[i]))).strftime('%Y%m%d-%H%M') for i in range(len(f_datetime))]

    #==== ABI ====
    #Getting the time difference from the 0 and 5 ones place
    dt_int = np.array([int(t.strftime('%M'))%5 for t in f_datetime])
    #Changing those minutes that are on the 0 and 5 ones place so they're five minutes away
    dt_int[dt_int==0] = 5
    #Getting the target file times from the most recent ABI file
    file_times_abi = [(f_datetime[i] - timedelta(minutes=int(dt_int[i])-1)).strftime('s%Y%j%H%M') for i in range(len(f_datetime))]

    #==== MRMS ====
    #Getting the time difference from the 10s place
    dt_int = np.array([int(t.strftime('%M'))%10 for t in f_datetime])
    #Changing those minutes greater than 5 so they default to 6 rather than 0
    idx = np.where(dt_int>=6)[0]
    dt_int_new = dt_int[idx] % 6
    dt_int[dt_int>=6] = dt_int_new
    #Getting the target file times from the most recent MRMS file
    file_times_mrms = [(f_datetime[i] - timedelta(minutes=int(dt_int[i]))).strftime('%Y%m%d-%H%M') for i in range(len(f_datetime))]

    return file_timestamp, file_times_abi, file_times_mrms


def df_creator(grid_lats, grid_lons, file_timestamp, case):
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
    grid_lats_looped = [grid_lats[i] for j in ts_unique for i in range(len(grid_lats))]
    grid_lons_looped = [grid_lons[i] for j in ts_unique for i in range(len(grid_lons))]
    case_looped = [case for j in ts_unique for i in range(len(grid_lons))]

    #Creating looped versions of the timestamps that match each lat/lon combination
    ts_looped = [ts for ts in ts_unique for i in grid_lats]

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
    save_loc = output_loc + '/' + case +'/'
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    df.to_csv(save_loc+fsave_str)


def idx_finder(t_lat, t_lon, d_lats, d_lons):
    dx = 0.1
    idx = np.where((d_lats>=t_lat-dx)&
             (d_lats<t_lat+dx)&
             (d_lons>=t_lon-dx)&
             (d_lons<t_lon+dx))[0]
    return idx

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
        cur_tstamp = row['file_timestamp']
        cur_lat = row['lat']
        cur_lon = row['lon']
        cur_fistart_flid = row['fistart_flid']

        #Finding which index has a matching timestamp and is closest to the current point. 
        idx = np.where((grid_df['timestamp']==cur_tstamp) &
                 (grid_df['lat']>=cur_lat-dx) &
                 (grid_df['lat']<cur_lat+dx) &
                 (grid_df['lon']>=cur_lon-dx) &
                 (grid_df['lon']<cur_lon+dx))[0]
        
        if len(idx) == 0:
            print ('GLM FF ERROR: FIRST FLASH NOT PLACED ON GRID')
            print ('---'+cur_fistart_flid+'---')
            continue
        elif len(idx) == 1:
            grid_df.loc[idx[0],'ff_point'] = 1
            grid_df.loc[idx[0],'ff_fistart_flid'] = cur_fistart_flid
        else:
            print ('GLM FF ERROR: MORE THAN ONE POINT COINCIDES WITH FIRST FLASH')
            print ('---'+cur_fistart_flid+'---')
            continue
    
    return grid_df

def tstamp_converter(cur_tstamp):
    '''
    Taking the current tstamp (str) and getting the previous 15 mintues of timestamps
    PARAMS:
        cur_tstamp (str)
    RETURNS:
        tstamps (list of strings)
    '''
    #List of ints to interate through as a timedelta
    dt_int = np.arange(5,20,5)

    #Creating the list we'll append to
    tstamps = [cur_tstamp]

    for t in dt_int:
        dt = timedelta(minutes=int(t))
        cur_dt = datetime.strptime(cur_tstamp, '%Y%m%d-%H%M')
        new_tstamp = datetime.strftime(cur_dt-dt, '%Y%m%d-%H%M')
        tstamps = np.append(tstamps,new_tstamp)

    return tstamps

def ff_driver_v2(grid_df, case_df, file_timestamp):
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
        cur_tstamp = row['file_timestamp']
        cur_lat = row['lat']
        cur_lon = row['lon']
        cur_fistart_flid = row['fistart_flid']

        #Taking the current timestamp and getting the timestamps 5, 10, and 15 minutes before
        tstamps = tstamp_converter(cur_tstamp)

        #Finding which index has a matching timestamp and is closest to the current point. 
        idx = np.where(((grid_df['timestamp']==tstamps[0])|(grid_df['timestamp']==tstamps[1])|(grid_df['timestamp']==tstamps[2])|(grid_df['timestamp']==tstamps[3])) &
                 (grid_df['lat']>=cur_lat-dx) &
                 (grid_df['lat']<cur_lat+dx) &
                 (grid_df['lon']>=cur_lon-dx) &
                 (grid_df['lon']<cur_lon+dx))[0]

        if len(idx) < 4:
            print ('GLM FF ERROR: FIRST FLASHES NOT PLACED ON GRID')
            print ('---'+cur_fistart_flid+'---')
            continue
        elif len(idx) == 4:
            for i in idx:
                grid_df.loc[i,'ff_point'] = 1
                grid_df.loc[i,'ff_fistart_flid'] = cur_fistart_flid
        else:
            print ('GLM FF ERROR: MORE THAN THREE POINTS COINCIDES WITH FIRST FLASH')
            print ('---'+cur_fistart_flid+'---')
            continue
    
    return grid_df



#====================================================================
# ABI FUNCTIONS
#====================================================================

def abi_driver(grid_df, file_timestamp, file_times_abi, grid_lats, grid_lons):
    '''
    Main function that transforms the data
    PARAMS:
        grid_df
        file_timestamp
        file_times_abi
        grid_lats
        grid_lons
    RETURNS:
        grid_df
    '''
    #Adding the columns to the dataframe
    grid_df['abi_cmip_min'] = pd.Series(data=(np.ones(grid_df.shape[0]) * -999.), dtype=float)
    grid_df['abi_cmip_p05'] = pd.Series(data=(np.ones(grid_df.shape[0]) * -999.), dtype=float)
    grid_df['abi_acha_max'] = pd.Series(data=(np.ones(grid_df.shape[0]) * -999.), dtype=float)
    grid_df['abi_acha_p95'] = pd.Series(data=(np.ones(grid_df.shape[0]) * -999.), dtype=float)

    #Getting the unique timestamps for all first flashes
    ts_unique = np.unique(file_timestamp)

    #Looping through each available timestep
    for ts in ts_unique[:]:
        #Subsetting the abi file times to get the current one
        idx = np.where(np.array(file_timestamp)==ts)[0]
        idx = idx[0]
        cur_abi_ftime = file_times_abi[idx]
        
        #Getting the file strings from the file time
        acha_file, cmip_file = abi_file_hunter(cur_abi_ftime)

        #If the CMIP and ACHA data are available then grab it! If not we'll skip it
        if (acha_file != 'MISSING') and (cmip_file != 'MISSING'):
            #Loading the files
            cmip_lats, cmip_lons, acha_var, cmip_var = abi_file_loader_v2(acha_file, cmip_file)
            #Taking the data and placing it into the grid
            grid_df = abi_sampler(grid_df, ts, cmip_lats, cmip_lons, acha_var, cmip_var, grid_lats, grid_lons)
    
    return grid_df

# A function that finds the abi file based on its time string
def abi_file_hunter(abi_time_str):
    #Turning the time string into a datetime object so we can get the date of the file
    abi_file_datetime = datetime.strptime(abi_time_str, 's%Y%j%H%M')
    y, m, d, doy, hr, mi = datetime_converter(abi_file_datetime)
    
    #Creating the file strings we'll put through glob
    acha_loc = '/localdata/first-flash/data/ABI16-ACHAC/'+y+m+d+ '/*' + abi_time_str + '*'
    cmip_loc = '/localdata/first-flash/data/ABI16-CMIPC13/'+y+m+d+ '/*' + abi_time_str + '*'
    
    #Finding the product files via glob
    acha_file = glob(acha_loc)
    cmip_file = glob(cmip_loc)

    #Tagging the times where data is missing for later
    if len(acha_file)<1:
        print ('ACHA DATA MISSING')
        print (acha_loc)
        acha_file = ['MISSING']
    if len(cmip_file)<1:
        print ('CMIP DATA MISSING')
        print (cmip_loc)
        cmip_file = ['MISSING']     
        
    return acha_file[0], cmip_file[0]


def abi_file_loader_v2(acha_file,cmip_file):

    #loading the cmip13 data
    cmip_x, cmip_y, cmip_var, cmip_lons, cmip_lats = abi_importer(cmip_file, 'CMI', np.nan)
    
    #loading the acha data
    acha_x, acha_y, acha_var, acha_lons, acha_lats = abi_importer(acha_file, 'HT', np.nan)        
    
    #If the CMIP and ACHA data are there, resampling the ACHA data to the CMIP 2km grid and use as a clear sky mask
    #Resampling the ACHA the CMIP grid
    acha_var = resample(acha_var, acha_lats, acha_lons, cmip_lats, cmip_lons)
    #Appling a mask to the cmip data based on the acha data
    cmip_var[np.isnan(acha_var)] = np.nan
    #Flattening the arrays for the output
    cmip_var = cmip_var[acha_var>0]
    cmip_lats = cmip_lats[acha_var>0]
    cmip_lons = cmip_lons[acha_var>0]
    acha_var = acha_var[acha_var>0]
        
    return cmip_lats, cmip_lons, acha_var, cmip_var


#Short function to shorten abi_file_loader
def abi_importer(file, var, fill_val):
    dset = nc.Dataset(file, 'r')
    x = dset.variables['x'][:]
    y = dset.variables['y'][:]
    var = np.ma.filled(dset.variables[var][:,:], fill_value=fill_val)
    lons, lats = latlon(dset)
    dset.close()
    return x, y, var, lons, lats


def resample(field, orig_lats, orig_lons, target_lats, target_lons):

    #Creating the swath definitions
    target_swath = SwathDefinition(lons=target_lons,lats=target_lats)
    orig_swath = SwathDefinition(lons=orig_lons,lats=orig_lats)

    #Resampling using a KD-tree to fit the data to a grid
    output = kd_tree.resample_nearest(source_geo_def=orig_swath,
                            data=field,
                            target_geo_def=target_swath,
                            radius_of_influence=4e4)
    
    output[output==0.] = np.nan
    
    return output


#A function that takes in a netCDF Dataset and returns its x/y coordinates as as lon/lat
def latlon(data):    
    #Getting our variables from the netcdf file
    sat_h = data.variables['goes_imager_projection'].perspective_point_height
    sat_lon = data.variables['goes_imager_projection'].longitude_of_projection_origin
    sat_sweep = data.variables['goes_imager_projection'].sweep_angle_axis
    
    X = data.variables['x'][:] * sat_h
    Y = data.variables['y'][:] * sat_h

    #Setting up our projections
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
    YY, XX = np.meshgrid(Y, X)
    lons, lats = p(XX, YY, inverse=True)
    
    #Cleaning up the data
    lons[lons > 10000] = np.nan
    lats[lats > 10000] = np.nan
    lons[lons>180] = lons[lons>180] - 360
    
    return lons.T, lats.T

def abi_sampler(grid_df, ts, cmip_lats, cmip_lons, acha_var, cmip_var, grid_lats, grid_lons):
    #Applying parallax correction
    cmip_lons, cmip_lats = plax.get_parallax_corrected_lonlats(sat_lon=-75.0, sat_lat=0.0, sat_alt=35786023.0,
                                            lon=cmip_lons, lat=cmip_lats, height=acha_var)
    #Looping through each lat/lon on the target grid
    for t_lat, t_lon in zip(grid_lats, grid_lons):

        #Getting the index
        idx = idx_finder(t_lat, t_lon, cmip_lats, cmip_lons)
        
        #Fiding the index in the gridded dataset
        idx_grid =  np.where((grid_df['lat']==t_lat) & (grid_df['lon']==t_lon) & (grid_df['timestamp']==ts))[0]
        if len(idx_grid) == 0:
            print('ABI ERROR: NO GRID POINT FOUND')
            print (t_lat, t_lon)
            continue
        elif len(idx_grid) >1:
            print('ABI ERROR: MILTIPLE GRID POINTS FOUND')
            print (t_lat, t_lon)
            continue
        else:
            idx_grid = idx_grid[0]

        #If there's data sample it    
        if len(idx)>0:
            #Getting the appropriate samples and placing them into the dataset
            grid_df.loc[idx_grid, 'abi_acha_max'] = np.nanmax(acha_var[idx])
            grid_df.loc[idx_grid, 'abi_acha_p95'] = np.nanpercentile(a=acha_var[idx], q=95)
            grid_df.loc[idx_grid, 'abi_cmip_min'] = np.nanmax(cmip_var[idx])
            grid_df.loc[idx_grid, 'abi_cmip_p05'] = np.nanpercentile(a=cmip_var[idx], q=5)
        #If there's no clouds in the scene, we'll return np.nan for now...
        else:
            grid_df.loc[idx_grid, 'abi_acha_max'] = np.nan
            grid_df.loc[idx_grid, 'abi_acha_p95'] = np.nan
            grid_df.loc[idx_grid, 'abi_cmip_min'] = np.nan
            grid_df.loc[idx_grid, 'abi_cmip_p05'] = np.nan
            
    return grid_df




#====================================================================
# MRMS FUNCTIONS
#====================================================================
def mrms_driver(grid_df, file_timestamp, file_times_mrms, grid_lats, grid_lons, mrms_vars):
    #Looping through each mrms variable
    for var in mrms_vars:
        #Creating the columns that we'll fill with data
        grid_df[var+'_max'] = pd.Series(data=(np.ones(grid_df.shape[0]) * -999.), dtype=float)
        grid_df[var+'_p95'] = pd.Series(data=(np.ones(grid_df.shape[0]) * -999.), dtype=float)
        
        #Getting the unique timestamps for all first flashes
        ts_unique = np.unique(file_timestamp)

        #Looping through each available timestep
        for ts in ts_unique[:]:
            #Subsetting the mrms file times to get the current one
            idx = np.where(np.array(file_timestamp)==ts)[0]
            idx = idx[0]
            cur_mrms_ftime = file_times_mrms[idx]

            #Loading the MRMS data
            lat_data, lon_data, data = mrms_data_loader(cur_mrms_ftime, var)

            #If the file isn't found skip it
            if data[0]==-999:
                continue
            #If not then fit to the grid
            else:
                grid_df = mrms_sampler(grid_df, ts, lat_data, lon_data, data, grid_lats, grid_lons, var)

    return grid_df

def mrms_data_loader(cur_mrms_ftime ,var):
    
    #Using the current time and variable to get the MRMS file we need
    fstring_start = '/raid/swat_archive/vmrms/CONUS/'+cur_mrms_ftime[:8]+'/multi/'
    file_str = fstring_start+var+'/00.50/'+cur_mrms_ftime+'*.netcdf.gz'
    
    #Searching for the file
    file_locs = glob(file_str)
    if len(file_locs)==0:
        print ('MRMS ERROR: File not found '+var)
        print (file_str)
        lat_data = [-999]
        lon_data = [-999]
        data = [-999]
    elif len(file_locs)>1:
        print ('MRMS ERROR: More than one file found '+var)
        lat_data = [-999]
        lon_data = [-999]
        data = [-999]
    else:
        #This is what I call a 'pro-gamer' move...loading the netcdfs while zipped on another machine
        with gzip.open(file_locs[0]) as gz:
            with nc.Dataset('dummy', mode='r', memory=gz.read()) as dset:
                #loading in the data from the MRMS netcdf file
                x_pix = dset.variables['pixel_x'][:] #Pixel locations (indicies) for LATITUDE
                y_pix = dset.variables['pixel_y'][:] #Pixel locations (indicies) for LONGITUDE
                data = dset.variables[var][:]

                u_lat = dset.Latitude #Upper-most latitude
                l_lon = dset.Longitude #Left-most longitude

                #Creating the arrays for the lat and lon coordinates
                y = dset.dimensions['Lat'].size #3500
                x = dset.dimensions['Lon'].size #7000
                lat = np.arange(u_lat, u_lat-(y*0.01),-0.01) #Going from upper to lower
                lon = np.arange(l_lon, l_lon+(x*0.01),0.01) #Going from left to right

                #Using the pixel indicides to get the pixel latitudes and longitudes
                lat_data = lat[x_pix] #Remember x_pixel represents LATITUDE
                lon_data = lon[y_pix] #Remember y_pixel represent LONGITUDE
                
                #Removing any false data (less data to process)
                locs = np.where((data>0))[0]
                lon_data = lon_data[locs]
                lat_data = lat_data[locs]
                data = data[locs]

    return lat_data, lon_data, data       

def mrms_sampler(grid_df, ts, lat_data, lon_data, data, grid_lats, grid_lons, var):
    #Looping through each lat/lon on the target grid
    for t_lat, t_lon in zip(grid_lats, grid_lons):
        #Getting the index
        idx = idx_finder(t_lat, t_lon, lat_data, lon_data)
        
        #Fiding the index in the gridded dataset
        idx_grid =  np.where((grid_df['lat']==t_lat) & (grid_df['lon']==t_lon) & (grid_df['timestamp']==ts))[0]
        if len(idx_grid) == 0:
            print('MRMS ERROR: NO GRID POINT FOUND')
            print (t_lat, t_lon)
            continue
        elif len(idx_grid) >1:
            print('MRMS ERROR: MILTIPLE GRID POINTS FOUND')
            print (t_lat, t_lon)
            continue
        else:
            idx_grid = idx_grid[0]

        #If there's data sample it    
        if len(idx)>0:
            #Getting the appropriate samples and placing them into the dataset
            grid_df.loc[idx_grid, var+'_max'] = np.nanmax(data[idx])
            grid_df.loc[idx_grid, var+'_p95'] = np.nanpercentile(a=data[idx], q=95)

        #If there's no data in the scene, we'll return 0. for now...
        else:
            grid_df.loc[idx_grid, var+'_max'] = 0.
            grid_df.loc[idx_grid, var+'_p95'] = 0.


    return grid_df
        

#====================================================================
# GLM FUNCTIONS
#====================================================================

def glm_driver(grid_df, file_timestamp, grid_lats, grid_lons, all_flash_file):
    '''
    Getting the number of GLM flashes that occurred within the grid cell in the previous 20 minutes
    PARAMS:
        grid_df
        file_timestamp
        grid_lats
        grid_lons
        all_flash_file
    RETURNS:
        grid_df
    '''
    #Reading in the cases all flash file
    all_flash_df = read_all_flash_file(all_flash_file)

    #Adding the number of GLM flashes as a variable
    grid_df['glm_number_flashes_pre20'] = pd.Series(data=(np.ones(grid_df.shape[0]) * -999.), dtype=float)
    grid_df['glm_number_flashes_pre05'] = pd.Series(data=(np.ones(grid_df.shape[0]) * -999.), dtype=float)

    #Getting the unique timestamps for all first flashes
    ts_unique = np.unique(file_timestamp)

    #Looping through each available timestep
    for ts in ts_unique[:]:
        #Turning the current timestep into a datetime object
        ts_datetime = datetime.strptime(ts,'%Y%m%d-%H%M')

        #Putting the glm flash counts into the grid
        grid_df = glm_sampler(grid_df, all_flash_df, grid_lats, grid_lons, ts_datetime, ts)

    return grid_df


def read_all_flash_file(all_flash_file):
    #Reading in the first flash data
    file_loc = '/localdata/first-flash/data/GLM16-cases-allflash/'
    all_flash_df = pd.read_csv(file_loc+all_flash_file, index_col=0)
    #Converting the start time to a datetime object
    all_flash_df['start_time'] = pd.to_datetime(all_flash_df['start_time'])

    return all_flash_df


def glm_sampler(grid_df, all_flash_df, grid_lats, grid_lons, ts_datetime, ts):
    #Setting the timedeltas
    dt_20 = timedelta(minutes=20)
    dt_05 = timedelta(minutes=5)

    #Subsetting the all flash dataframes for the previous 20 and 5 minutes
    all_flash_df_pre05 = all_flash_df.loc[(all_flash_df['start_time']>=ts_datetime-dt_05)&(all_flash_df['start_time']<ts_datetime)]
    all_flash_df_pre20 = all_flash_df.loc[(all_flash_df['start_time']>=ts_datetime-dt_20)&(all_flash_df['start_time']<ts_datetime)]

    #Looping through each lat/lon on the target grid
    for t_lat, t_lon in zip(grid_lats, grid_lons):

        #Fiding the index in the gridded dataset
        idx_grid =  np.where((grid_df['lat']==t_lat) & (grid_df['lon']==t_lon) & (grid_df['timestamp']==ts))[0]
        if len(idx_grid) == 0:
            print('GLM ERROR: NO GRID POINT FOUND')
            print (t_lat, t_lon)
            continue
        elif len(idx_grid) >1:
            print('GLM ERROR: MILTIPLE GRID POINTS FOUND')
            print (t_lat, t_lon)
            continue
        else:
            idx_grid = idx_grid[0]

        #Getting the index for the values in the last 5 and 20 minutes
        idx_pre20 = idx_finder(t_lat, t_lon, all_flash_df_pre20['lat'].to_list(), all_flash_df_pre20['lon'].to_list())
        idx_pre05 = idx_finder(t_lat, t_lon, all_flash_df_pre05['lat'].to_list(), all_flash_df_pre05['lon'].to_list())

        #Getting the number of GLM flashes and placing them into the grid
        grid_df.loc[idx_grid,'glm_number_flashes_pre20'] = len(idx_pre20)
        grid_df.loc[idx_grid,'glm_number_flashes_pre05'] = len(idx_pre05)

    return grid_df