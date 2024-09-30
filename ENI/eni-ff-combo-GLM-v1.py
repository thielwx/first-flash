#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This script is designed to match GLM flashes with ENI flashes for the purposes 
# of a comparison study between the two networks


# In[44]:


import netCDF4 as nc
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
import multiprocessing as mp
from glob import glob
import gzip
from sklearn.neighbors import BallTree
import os

dt_start_job = datetime.now()


# # Function Land

# In[20]:


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

    return np.array(flash_datetime)


# In[21]:


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


# In[57]:


def eni_loader(start_time, end_time, eni_vars):
    file_loc = '/localdata/first-flash/data/ENI-base-stock/'
    #Getting all of the strings for the days before and after the file
    y, m, d, doy, hr, mi = datetime_converter(start_time-timedelta(seconds=1))
    file_str1 = 'eni_flash_flash'+y+m+d+'.csv'
    y, m, d, doy, hr, mi = datetime_converter(start_time)
    file_str2 = 'eni_flash_flash'+y+m+d+'.csv'
    err_str = y+m+d
    y, m, d, doy, hr, mi = datetime_converter(end_time)
    file_str3 = 'eni_flash_flash'+y+m+d+'.csv'
    
    #Empty dataframe to fill
    df = pd.DataFrame(columns=eni_vars)
    
    #Looping through the data to load the full dataset
    for file_str in [file_str1, file_str2, file_str3]:
        eni_file = glob(file_loc+file_str)
        if len(eni_file>0):
            df_new = pd.read_csv(eni_file[0])
            df = pd.concat((df,df_new),axis=0)
        else:
            print ('FILE NOT FOUND')
            print (file_loc+file_str)
    
    #If we have data, get the times and pair it down
    if df.shape[0]>0:
        #Pairing down by the spatial bounds
        latlon_locs = lat_lon_bounds(df['latitude'].values, df['longitude'].values)
        df = df.iloc[latlon_locs,:]
        
        #Getting the times for the data in the dataframe and pairing down by time
        eni_time = np.array([np.datetime64(df['timestamp'].values[i]) for i in range(df.shape[0])])
        eni_time_locs = np.where((eni_time>=start_time-timedelta(seconds=10)) & (eni_time<=end_time+timedelta(seconds=10)))[0]
        df = df.iloc[eni_time_locs] 
    
    else:
        print('NO DATA LOADED')
        
    return df


# In[ ]:


def latlon_bounds(flash_lats, flash_lons):
    '''
    Takes in the flash latitudes and longitudes determines which ones are within the domain
    PARAMS:
        flash_lats: array of flash latitudes (floats)
        flash_lons: array of flash longitudes (floats) 
    RETURNS:
        latlon_locs: array of indicies from which the input lat/lon values are within the domain
    '''
    #Specifying the lat/lon bounds for the first-flash domain
    lat_max = 50
    lat_min = 24
    lon_max = -66
    lon_min = -125
    
    #Finding the indicies where the lat/lon locs sit within the bounds
    latlon_locs = np.where((flash_lats<=lat_max)&(flash_lats>=lat_min)&(flash_lons<=lon_max)&(flash_lons>=lon_min))[0]
    
    return latlon_locs


# In[59]:


#====================================================
#The driver function that puts out the data in two hour chunks
#====================================================
def eni_driver(t_start, t_end):
    global eni_vars
    global version
    global ff_time_start
    global ff_time_end
    global eni_df
    global f_lat
    global f_lon
    
    #Getting the 2-hour segment 
    df_locs = np.where((ff_time_start>=np.datetime64(t_start)) & (ff_time_end<np.datetime64(t_end)))[0]

    #If there's first flash data lets run stuff. If not then don't!
    if len(df_locs)>0:
        #Creating an empty dataframe to fill
        fistart_flid_cutdown = fistart_flid[df_locs]
        df = pd.DataFrame(index=fistart_flid_cutdown, columns=eni_vars)
        
        #Getting an updated list of first flash lat-lon-time points that correspond with the dataframe
        f_lat_cut = f_lat[df_locs]
        f_lon_cut = f_lon[df_locs]
        f_tstart_cut = ff_time_start[df_locs]
        f_tend_cut = ff_time_end[df_locs]
        
        #Getting the eni times
        eni_stime = np.array([np.datetime64(eni_df['starttime'].values[i]) for i in range(eni_df.shape[0])])
        eni_etime = np.array([np.datetime64(eni_df['endtime'].values[i]) for i in range(eni_df.shape[0])])
        
        #Looping through the individual first flashes
        for i in range(len(f_lat_cut)):
            #Getting the current flash data that we'll be finding in the eni data
            cur_fi_fl = fi_start_flid_cutdown[i]
            cur_f_lat = f_lat_cut[i]
            cur_f_lon = f_lon_cut[i]
            cur_tstart = f_tstart_cut[i]
            cur_tend = f_tend_cut[i]
            
            eni_data_package = eni_flash_grabber(eni_df, eni_stime, eni_etime, eni_vars, cur_f_lat, cur_f_lon, cur_tstart, cur_tend)
            
            for i in range(len(eni_vars)):
                df[cur_fi_fl,eni_vars[i]] = eni_data_package[i]
         
        eni_data_saver(df,t_start,t_end,version)
    


# In[94]:


#A function that finds the appropriate eni flash for the current GLM first flash under investigation
def eni_flash_grabber(eni_df, eni_stime, eni_etime, eni_vars, f_lat, f_lon, f_start, f_end):
    global search_ms_dt #Search time bounds defined previously
    global search_km #Search radius defined previously
    R = 6371.0087714 #Earths radius in km
    dx = 1.0 #Search range in degrees (to cut down the amount of ABI data we're searching)
    eni_data_package = np.empty(len(eni_vars)) #Empty 'package for our eni data
    
    eni_lons = eni_df['longitude'].values
    eni_lats = eni_df['latitude'].values
    
    #Cutting down the ABI searchable data by space and time
    eni_locs = np.where((eni_lons>=f_lon-dx) & (eni_lons<=f_lon+dx) & (eni_lats<=f_lat+dx) & (eni_lats>=f_lat-dx) &
                       ((eni_etime>=f_start-search_ms_dt) | (eni_stime<=f_end+search_ms_dt)))[0]
    eni_df_cut = eni_df.iloc[eni_locs,:]
    
    #If there's eni data, go through the trouble of running the ball tree
    if eni_df_cut.shape[0]>0:
        eni_lons_rad = eni_df_cut['longitude'].values * (np.pi/180)
        eni_lats_rad = eni_df_cut['latitude'].values * (np.pi/180)
        eni_latlons = np.vstack((eni_lats_rad, eni_lons_rad)).T

        #Setting up the first flash lat lons
        f_lat_rad = f_lat * (np.pi/180)
        f_lon_rad = f_lon * (np.pi/180)
        ff_latlons = np.reshape([f_lat_rad,f_lon_rad], (-1, 2))

        #Implement a Ball Tree to capture the flashes within range
        btree = BallTree(eni_latlons, leaf_size=2, metric='haversine')
        indicies = btree.query_radius(ff_latlons, r = search_km/R)
        idx = indicies[0]


        
        #If there's only one data point in range, we're done and grab the data!
        if len(idx)==1:
            #Looping through the package to fill it with the corresponding data from each variable in the dataframe
            for i in range(len(eni_vars)):
                eni_data_package[i] = eni_df_cut.iloc[idx[0],i]
        #If there's more than one eni flash, then we'll grab the one closest temporally
        elif len(idx)>1:
            #Getting the mean times of the first flashes and eni flashes
            eni_mean_times = [np.average(np.array(eni_stime[idx[i]], eni_etime[idx[i]])) for i in range(len(idx))]
            ff_mean_time = np.average(np.array(f_start, f_end))
            time_difference = np.abs(eni_mean_times - ff_mean_time) #Absolute time difference of the two
            t_idx = np.where(time_difference==np.nanmin(time_difference))[0] #Getting where the time difference in the smallest

            for i in range(len(eni_vars)):
                eni_data_package[i] = eni_df_cut.iloc[t_idx,i]


    return eni_data_package


# In[ ]:


def eni_data_saver(df, t_start, t_end, version):
    global glm_number

    y, m, d, doy, hr, mi = datetime_converter(t_start)
    output_folder = y+m+d
    start_time_str = 's'+y+m+d+hr+mi
    y, m, d, doy, hr, mi = datetime_converter(t_end)
    end_time_str = 'e'+y+m+d+hr+mi
    y, m, d, doy, hr, mi = datetime_converter(datetime.now())
    cur_time_str = 'c'+y+m+d+hr+mi
    
    output_loc = '/localdata/first-flash/data/ENI-processed-GLM'+glm_number+'-v'+str(version)+'/'+output_folder +'/'
    output_file = 'ENI-GLM'+glm_number+'-ff-v'+str(version)+'-'+start_time_str+'-'+end_time_str+'-'+cur_time_str+'.csv'
    if not os.path.exists(output_loc):
        os.makedirs(output_loc, exist_ok=True)
    df.to_csv(output_loc+output_file)
    print (output_loc+output_file)


# # Work Zone

# In[22]:


version = 1
search_km = 32 #Search radius in km based on Rudlosky and Virts 2021
search_ms = 200 #Search time in ms based on Rudlosky and Virts 2021
search_ms_dt = np.timedelta64(search_ms, 'ms')
eni_vars = ['type','timestamp','latitude','longitude','peakcurrent','icheight','numbersensors','icmultiplicity','cgmultiplicity','starttime','endtime','duration','ullatitude','ullongitude','lrlatitude','lrlongitude']


# In[23]:


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


# In[86]:


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
ff_time_start = GLM_LCFA_times_postprocess(fistart_str, nc_dset.variables['flash_time_offset_of_first_event'][:])
ff_time_end = GLM_LCFA_times_postprocess(fistart_str, nc_dset.variables['flash_time_offset_of_last_event'][:])


# In[19]:


#Outmost loop for a daily basis
for i in range(len(time_list_days)-1):
    t_range_start = time_list_days[i]
    t_range_end = time_list_days[i+1]
    
    #Breaking the day into 12, 2-hour chunks
    tlist_starmap = pd.date_range(start=t_range_start, end=t_range_end, freq='2H')
    
    #Getting the time for the ENI file string
    y, m, d, doy, hr, mi = datetime_converter(t_range_start)
    print ('---'+y+m+d+'---')
    
    #Downloading the eni data
    eni_df = eni_loader(t_range_start,t_range_end, eni_vars)
    
    #Sending the file string to the mrms_driver function that takes over from here...
    if __name__ == "__main__":
        with mp.Pool(12) as p:
            #p.starmap(eni_driver, zip(tlist_starmap[:-1], tlist_starmap[1:]))
            p.starmap(eni_driver, zip(tlist_starmap[0:1], tlist_starmap[1:2])) #DEVMODE
            p.close()
            p.join()

print ('Total Run Time:')
print (datetime.now()-dt_start_job)

