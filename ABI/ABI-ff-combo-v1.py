#!/usr/bin/env python
# coding: utf-8

# In[1]:


#===================================================
# This script is designed to find ABI CMIP13 and ACHA from a GLM-ff-processed file. Inspired by MRMS-ff-combo-v1.py
#
# Created: September 2024
# Author: Kevin Thiel (kevin.thiel@ou.edu)
#
# Run like this: python ABI-ff-combo-v1.py 20220101 20220201 /localdata/first-flash/data/GLM16-ff-processed/*FILENAMEHERE*.nc
#===================================================


# In[2]:


import netCDF4 as nc
import pandas as pd
import sys
from glob import glob
import os
from datetime import datetime, timedelta
import numpy as np
import multiprocessing as mp
from pyproj import Proj
from pyresample import SwathDefinition, kd_tree
from sklearn.neighbors import BallTree
import satpy.modifiers.parallax as plax


# In[3]:
dt_start_job = datetime.now()

# Constants
version = 1
abi_variables = ['CMIP','ACHA']
abi_variables_output = ['CMIP_min', 'CMIP_min_pre10', 'CMIP_05', 'CMIP_05_pre10', 'ACHA_max', 'ACHA_max_pre10', 'ACHA_95', 'ACHA_95_pre10']


# # Function Land

# In[4]:


#This function takes in a file start time and the first/last event times to create a list of datetime objects
def GLM_LCFA_times_postprocess(file_times, times):
    '''
    Creates a list of datetime objects from the LCFA L2 file times, and the start time of the file
    PARAMS:
        file_times: listed times on the LCFA L2 file (str)
        times: times (seconds) from the LCFA L2 file (float)
    RETURNS
        flash_datetime: a list of datetimes based on the flash/group/event times in the LCFA L2 file down to ns (datetime)
    '''
    
    #Converting to nanoseconds to use for timedelta
    nanosecond_times = times*(10**9)
    
    #Creating datetime object for the file time
    flash_file_datetime = [np.datetime64(datetime.strptime(file_times[i], 's%Y%j%H%M%S0')) for i in range(len(file_times))]
    
    #Creating timedetla objects from our array
    flash_timedelta = [np.timedelta64(int(val), 'ns') for val in nanosecond_times]
    
    #Creating an array of datetime objects with the (more) exact times down to the microsecond
    #flash_datetime = [flash_file_datetime+dt for dt in flash_timedelta]
    flash_datetime = [flash_file_datetime[i] + flash_timedelta[i] for i in range(len(flash_timedelta))]

    return (flash_datetime)


# In[5]:


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


# In[93]:


#====================================================
#The driver function that puts out the data in two hour chunks
#====================================================

def abi_driver(t_start, t_end):
    global abi_variables
    global abi_variables_output
    global version
    global fstring_start
    global f_time
    global fistart_flid
    global f_lat
    global f_lon
    
    f_time = np.array(f_time)

    #Getting the 2-hour segment 
    df_locs = np.where((f_time>=np.datetime64(t_start)) & (f_time<np.datetime64(t_end)))[0]

    #Creating an empty dataframe to fill
    fistart_flid_cutdown = fistart_flid[df_locs]
    df = pd.DataFrame(index=fistart_flid_cutdown, columns=abi_variables_output)
    
    #Getting the list of ABI file times that are required from the GLM first flashes
    abi_file_time_pre0, abi_file_time_pre10 = abi_file_times_ff(f_time[df_locs])
    
    #Placing the file strings in the dataframe in case we need them later (ya never know...)
    df['abi_file_stime_pre0'] = abi_file_time_pre0
    df['abi_file_stime_pre10'] = abi_file_time_pre10
    
    #Getting the list of ABI file times for the time period
    abi_times = abi_file_times(pd.date_range(start=t_start-timedelta(minutes=15), end=t_end, freq='5min'))
    
    #Getting an updated list of first flash lat-lon points that correspond with the dataframe
    f_lat_cut = f_lat[df_locs]
    f_lon_cut = f_lon[df_locs]

    #Looping through all of the abi files so I only have to open/process them once (I am speed)
    for abi_time_str in abi_times:
        #Finding the files that we'll need to load from
        pre0_locs = np.where(np.array(abi_file_time_pre0)==abi_time_str)[0]
        pre10_locs = np.where(np.array(abi_file_time_pre10)==abi_time_str)[0]
        
        #Getting the files that we'll load in
        acha_file, cmip_file = abi_file_hunter(abi_time_str)
        
        #If there's first flashes that correspond with the current abi time, then we'll load the data
        if (len(pre0_locs)>0 or len(pre10_locs)>0):
            #Loading the acha and cmip file
            abi_lats, abi_lons, acha_vals, cmip_vals = abi_file_loader(acha_file,cmip_file)
            
            #If there's first flashes at the ff time, get the max/min values of CMIP13/ACHA within 20 km
            if len(pre0_locs)>0:
                #print (pre0_locs)
                #Looping through each flash.
                for loc in pre0_locs:
                    cur_fi_fl = fistart_flid_cutdown[loc]
                    cur_fl_lat = f_lat_cut[loc]
                    cur_fl_lon = f_lon_cut[loc]
                    
                    #Sampling the data using a 20 km BallTree to get what we want out of the file
                    cmip_min, cmip_05, acha_max, acha_95 = abi_data_sampler(abi_lats, abi_lons, acha_vals, cmip_vals, cur_fl_lat, cur_fl_lon)
                    #Placing the sampled values in the dataframe
                    df.loc[cur_fi_fl,'CMIP_min'] = cmip_min
                    df.loc[cur_fi_fl,'CMIP_05'] = cmip_05
                    df.loc[cur_fi_fl,'ACHA_max'] = acha_max
                    df.loc[cur_fi_fl,'ACHA_95'] = acha_95
        
            #If there's first flashes 20 min before the ff time, get the max/min values of CMIP13/ACHA within 20 km
            if len(pre10_locs)>0:
                #Looping through each flash.
                for loc in pre10_locs:
                    cur_fi_fl = fistart_flid_cutdown[loc]
                    cur_fl_lat = f_lat_cut[loc]
                    cur_fl_lon = f_lon_cut[loc]

                    #Sampling the data using a 20 km BallTree to get what we want out of the file
                    cmip_min, cmip_05, acha_max, acha_95 = abi_data_sampler(abi_lats, abi_lons, acha_vals, cmip_vals, cur_fl_lat, cur_fl_lon)
                    #Placing the sampled values in the dataframe
                    df.loc[cur_fi_fl,'CMIP_min_pre10'] = cmip_min
                    df.loc[cur_fi_fl,'CMIP_05_pre10'] = cmip_05
                    df.loc[cur_fi_fl,'ACHA_max_pre10'] = acha_max
                    df.loc[cur_fi_fl,'ACHA_95_pre10'] = acha_95
                    
    abi_data_saver(df, t_start, t_end, version)
            
            


# In[7]:


# Takes in the glm ff times and gets the corresponding ABI file start times (pre0 and pre10)
def abi_file_times_ff(f_time):
    f_time2 = [pd.to_datetime(t) for t in f_time]
    
    #Getting the time difference from the 0 and 5 ones place
    dt_int = np.array([int(t.strftime('%M'))%5 for t in f_time2])
    
    #Changing those minutes that are on the 0 and 5 ones place so they're five minutes away
    dt_int[dt_int==0] = 5

    #Getting the target file times from the most recent file and the file ten minutes before that
    file_time_pre0 = [(f_time2[i] - timedelta(minutes=int(dt_int[i])-1)).strftime('s%Y%j%H%M') for i in range(len(f_time2))]
    file_time_pre10 = [(f_time2[i] - timedelta(minutes=int(dt_int[i])+9)).strftime('s%Y%j%H%M') for i in range(len(f_time2))]
    
    return file_time_pre0, file_time_pre10


# In[8]:


# Takes in the glm ff times and gets the corresponding ABI file start times (pre0 and pre10)
def abi_file_times(time_list):
    
    #Getting the time difference from the 0 and 5 ones place
    dt_int = np.array([int(t.strftime('%M'))%5 for t in time_list])
    
    #Changing those minutes that are on the 0 and 5 ones place so they're five minutes away
    dt_int[dt_int==0] = 5
    
    #Getting the target file times from the most recent file
    abi_file_time = [(time_list[i] - timedelta(minutes=int(dt_int[i])-1)).strftime('s%Y%j%H%M') for i in range(len(time_list))]
    
    return abi_file_time


# In[9]:


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


# In[10]:


# This function takes in the file names and retrieves the data from the file (x,y,val)
def abi_file_loader(acha_file,cmip_file):
    
    #loading the cmip13 data
    if cmip_file != 'MISSING':
        cmip_x, cmip_y, cmip_var, cmip_lons, cmip_lats = abi_importer(cmip_file, 'CMI', np.nan)
    else:
        cmip_lats = np.array([-999])
        cmip_lons = np.array([-999])
        cmip_var = np.array([-999])
    
    #loading the acha data
    if acha_file != 'MISSING':
        acha_x, acha_y, acha_var, acha_lons, acha_lats = abi_importer(acha_file, 'HT', np.nan)
    else:
        acha_lats = np.array([-999])
        acha_lons = np.array([-999])
        acha_var = np.array([-999])
    
    #If no acha data are available, we'll put in the artificial bounds of 280 based on Thiel et al 2020    
    if  (acha_file == 'MISSING') and (cmip_file != 'MISSING'):
        cmip_var[cmip_var<280] = np.nan
    
    #If the CMIP and ACHA data are there, resampling the ACHA data to the CMIP 2km grid and use as a clear sky mask
    if (cmip_file != 'MISSING') and (acha_file != 'MISSING'):
        #Resampling the ACHA the CMIP grid
        acha_var = resample(acha_var, acha_lats, acha_lons, cmip_lats, cmip_lons)
        #Appling a mask to the cmip data based on the acha data
        cmip_var[np.isnan(acha_var)] = np.nan
        #Flattening the arrays for the output
        cmip_var = cmip_var[acha_var>0]
        cmip_lats = cmip_lats[acha_var>0]
        cmip_lons = cmip_lons[acha_var>0]
        acha_var = acha_var[acha_var>0]
        
    # If we dont have cmip data but do have acha data, just flatten the data and swap them with the cmip_x/y
    elif (cmip_file == 'MISSING') and (acha_file != 'MISSING'):
        acha_var = acha_var[acha_var>0]
        cmip_lons = acha_lons[acha_var>0]
        cmip_lats = acha_lats[acha_var>0]
        
    return cmip_lats, cmip_lons, acha_var, cmip_var


# In[11]:


#Short function to shorten abi_file_loader
def abi_importer(file, var, fill_val):
    dset = nc.Dataset(file, 'r')
    x = dset.variables['x'][:]
    y = dset.variables['y'][:]
    var = np.ma.filled(dset.variables[var][:,:], fill_value=fill_val)
    lons, lats = latlon(dset)
    return x, y, var, lons, lats

# In[12]:


def resample(field, orig_lats, orig_lons, target_lats, target_lons):

    #Creating 2D grids: From original import
    #orig_lats2d, orig_lons2d = np.meshgrid(orig_lats, orig_lons)

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


# In[13]:


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


# In[ ]:


# A function that finds the necessary ACHA and CMIP13 values within 20 km of a point
def abi_data_sampler(abi_lats, abi_lons, acha_vals, cmip_vals, cur_fl_lat, cur_fl_lon):
    max_range = 20 #Range in km to search
    R = 6371.0087714 #Earths radius in km
    dx = 1.0 #Search range in degrees (to cut down the amount of ABI data we're searching)
    
    #Cutting down the ABI searchable data
    abi_locs = np.where((abi_lons>=cur_fl_lon-dx) & (abi_lons<=cur_fl_lon+dx) & (abi_lats<=cur_fl_lat+dx) & (abi_lats>=cur_fl_lat-dx))[0]
    
    #If no data are found, assign all values to nans
    if len(abi_locs)==0:
        cmip_min = np.nan
        cmip_05 = np.nan
        acha_max = np.nan
        acha_95 = np.nan

    else:
        #Running the parallax correction
        lon_search, lat_search = plax.get_parallax_corrected_lonlats(sat_lon=-75.0, sat_lat=0.0, sat_alt=35786023.0,
                                            lon=abi_lons[abi_locs], lat=abi_lats[abi_locs], height=acha_vals[abi_locs])
        
        #Converting the abi lat/lon to radians
        abi_lats_rad = lat_search * (np.pi/180)
        abi_lons_rad = lon_search * (np.pi/180)
        #Converting first flash lat/lon to radians
        fl_lat_rad = cur_fl_lat * (np.pi/180)
        fl_lon_rad = cur_fl_lon * (np.pi/180)

        #Configuring the lat/lon data for the BallTree
        ff_latlons = np.reshape([fl_lat_rad,fl_lon_rad], (-1, 2))
        abi_latlons = np.vstack((abi_lats_rad, abi_lons_rad)).T
        
        #Implement a Ball Tree to capture the maximum/minimum and 95th/05th percentiles within the range of 20km
        btree = BallTree(abi_latlons, leaf_size=2, metric='haversine')
        indicies = btree.query_radius(ff_latlons, r = max_range/R)
        idx = indicies[0]
            
        #If no data in ball tree range OR the file is empty then set all to nans, if there's data, sample it!
        
        if (acha_vals[0] != -999) and (len(idx)!=0):
            acha_max = np.nanmax(acha_vals[idx])
            acha_95 = np.nanpercentile(a=acha_vals[idx], q=95)
        else: 
            acha_max = -999
            acha_95 = -999
            #print ('ACHA NANs')
            
        if (cmip_vals[0] != -999) and (len(idx)!=0):
            cmip_min = np.nanmin(cmip_vals[idx])
            cmip_05 = np.nanpercentile(a=cmip_vals[idx], q=5)
        else:
            cmip_min = -999
            cmip_05 = -999
            #print ('CMIP NANs')
        
    return cmip_min, cmip_05, acha_max, acha_95


# In[ ]:


def abi_data_saver(df, t_start, t_end, version):
    global glm_number

    y, m, d, doy, hr, mi = datetime_converter(t_start)
    output_folder = y+m+d
    start_time_str = 's'+y+m+d+hr+mi
    y, m, d, doy, hr, mi = datetime_converter(t_end)
    end_time_str = 'e'+y+m+d+hr+mi
    y, m, d, doy, hr, mi = datetime_converter(datetime.now())
    cur_time_str = 'c'+y+m+d+hr+mi
    
    output_loc = '/localdata/first-flash/data/ABI-processed-GLM'+glm_number+'-v'+str(version)+'/'+output_folder +'/'
    output_file = 'ABI-GLM'+glm_number+'-ff-v'+str(version)+'-'+start_time_str+'-'+end_time_str+'-'+cur_time_str+'.csv'
    if not os.path.exists(output_loc):
        os.makedirs(output_loc, exist_ok=True)
    df.to_csv(output_loc+output_file)
    print (output_loc+output_file)


# # Work Zone

# In[14]:


#Importing the user-defined variables
args = sys.argv
#args = ['NULL','20220101','20220201','../../local-data/2022/GLM16_first-flash-data-all_v32_s202201010000_e202212310000_c202409021440.nc'] #DEVMODE

s_time_str = args[1]
e_time_str = args[2]
nc_file_loc = args[3]

#Formatting the time based on the user inputs
start_time = datetime.strptime(s_time_str, '%Y%m%d')
end_time = datetime.strptime(e_time_str, '%Y%m%d')
time_list_days = pd.date_range(start=start_time, end=end_time, freq='1D') #Daily list to loop through

#Getting the necessary information from the netCDF file
nc_dset = nc.Dataset(nc_file_loc,'r')

#Getting the flash ids, lats, and lons for searching later...
fistart_flid = nc_dset.variables['flash_fistart_flid'][:]
f_lat = nc_dset.variables['flash_lat'][:]
f_lon = nc_dset.variables['flash_lon'][:]

#GLM number for later
glm_number = nc_dset.glm_number

#Getting the flash times for seraching later...
fistart_str = [i[0:15] for i in nc_dset.variables['flash_fistart_flid'][:]]
f_time = GLM_LCFA_times_postprocess(fistart_str, nc_dset.variables['flash_time_offset_of_first_event'][:])

# In[10]:


#Outmost loop for a daily basis
for i in range(len(time_list_days)-1):
    t_range_start = time_list_days[i]
    t_range_end = time_list_days[i+1]
    
    #Breaking the day into 12, 2-hour chunks
    tlist_starmap = pd.date_range(start=t_range_start, end=t_range_end, freq='2H')
    
    #Getting the time for the MRMS file string
    y, m, d, doy, hr, mi = datetime_converter(t_range_start)
    print ('---'+y+m+d+'---')
    
    #Sending the file string to the mrms_driver function that takes over from here...
    if __name__ == "__main__":
        with mp.Pool(12) as p:
            p.starmap(abi_driver, zip(tlist_starmap[:-1], tlist_starmap[1:]))
            #p.starmap(abi_driver, zip(tlist_starmap[0:1], tlist_starmap[1:2])) #DEVMODE
            p.close()
            p.join()

print ('Total Run Time:')
print (datetime.now()-dt_start_job)