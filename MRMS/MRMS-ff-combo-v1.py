#!/usr/bin/env python
# coding: utf-8

# In[1]:


#===================================================
# This script is designed to find MRMS -10C dBZ, Comp dBZ, and RALA from a GLM-ff-processed file
#
# Created: September 2024
# Author: Kevin Thiel (kevin.thiel@ou.edu)
#
# Run like this: python MRMS-ff-combo-v1.py /localdata/first-flash/data/GLM16-ff-processed/*FILENAMEHERE*.nc
#===================================================


# In[2]:


import netCDF4 as nc
import pandas as pd
import numpy as np
import sys
from datetime import datetime
import multiprocessing as mp
from glob import glob
import gzip
from sklearn.neighbors import BallTree
import os
import warnings
warnings.filterwarnings('ignore') 

# In[3]:


#Constants
mrms_variables = ['MergedReflectivityQCComposite','Reflectivity_-10C','ReflectivityAtLowestAltitude']
mrms_variables_output = ['MergedReflectivityQCComposite_max','MergedReflectivityQCComposite_95','Reflectivity_-10C_max','Reflectivity_-10C_95','ReflectivityAtLowestAltitude_max','ReflectivityAtLowestAltitude_95']
version = 1


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


# In[22]:


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


# In[5]:


#The driver function for starmap that processes the data in two hour chunks
def mrms_driver(t_start, t_end):
    #Loading in the other data to the function
    global mrms_variables
    global mrms_variables_output
    global version
    global fstring_start
    global f_time
    global fistart_flid
    global f_lat
    global f_lon
    
    #A 2-minute time list to go through each  mrms file
    tlist_2min = pd.date_range(start=t_start, end=t_end, freq='120s')
    
    df_locs = np.where((f_time>=np.datetime64(t_start)) & (f_time<np.datetime64(t_end)))[0]

    #Creating an empty dataframe to fill
    df = pd.DataFrame(index=fistart_flid[df_locs], columns=mrms_variables_output)
    
    #Looping through each 2 minute span in the mrms data
    for i in range(len(tlist_2min)-1):
        #Figuring out if there's any data in this section
        ff_locs = np.where((f_time>=np.datetime64(tlist_2min[i])) & (f_time<np.datetime64(tlist_2min[i+1])))[0]
        
        if len(ff_locs)>0: #If there's first flashes in the 2 minute span, we'll get the MRMS values
            
            #Adding each mrms variable individually
            for var in mrms_variables:
                mrms_lats, mrms_lons, mrms_data = MRMS_data_loader(tlist_2min[i], fstring_start, var)
                #A check that we actually have mrms data, or else skip this variable
                if mrms_lats[0]==-999:
                    continue
                
                #looping through each falsh now...
                for ff in ff_locs:
                    cur_fi_fl = fistart_flid[ff]
                    cur_fl_lat = f_lat[ff]
                    cur_fl_lon = f_lon[ff]
                    
                    #Getting the maximum and 95th percentile values and putting them into the dataframe
                    var_max, var_95 = mrms_max_finder(cur_fl_lat, cur_fl_lon, mrms_lats, mrms_lons, mrms_data)
                    df.loc[cur_fi_fl,var+'_max'] = var_max
                    df.loc[cur_fi_fl,var+'_95'] = var_95
    
    mrms_data_saver(df, t_start, t_end, version)


# In[ ]:


def MRMS_data_loader(time, fstring_start ,var):
    
    #Using the current time and variable to get the MRMS file we need
    y, m, d, doy, hr, mi = datetime_converter(time)
    file_time_str = y+m+d+'-'+hr+mi+'*.netcdf.gz'
    file_str = fstring_start+var+'/00.50/'+file_time_str
    
    #Searching for the file
    file_locs = glob(file_str)
    if len(file_locs)==0:
        print ('ERROR: File not found')
        print (file_str)
        lat_data = [-999]
        lon_data = [-999]
        data = [-999]
    elif len(file_locs)>1:
        print ('ERROR: More than one file found')
        lat_data = [-999]
        lon_data = [-999]
        data = [-999]
    else:
        #This is what I call a pro-'gramer' move...loading the netcdfs while zipped on another machine
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



# In[ ]:


#A function to find the maximum/95th percentile values within a specified range:
def mrms_max_finder(cur_fl_lat, cur_fl_lon, mrms_lats, mrms_lons, mrms_data):
    max_range = 20 #Range in km to search
    R = 6371.0087714 #Earths radius in km
    dx = 0.5 #Search range in degrees (to cut down the amount of MRMS data we're searching)
    
    #Cutting down the mrms searchable data and converting lat/lon to radians
    mrms_locs = np.where((mrms_lons>=cur_fl_lon-dx) & (mrms_lons<=cur_fl_lon+dx) & (mrms_lats<=cur_fl_lat+dx) & (mrms_lats>=cur_fl_lat-dx))[0]
    if len(mrms_locs)==0:
        mrms_data_max = np.nan
        mrms_data_95 = np.nan
    
    else:
        mrms_lats_rad = mrms_lats[mrms_locs] * (np.pi/180)
        mrms_lons_rad = mrms_lons[mrms_locs] * (np.pi/180)
        mrms_data_search = mrms_data[mrms_locs]

        mrms_latlons = np.vstack((mrms_lats_rad, mrms_lons_rad)).T
        if len(mrms_lats_rad)==1:
            mrms_latlons = np.reshape(mrms_latlons, (-1, 2))
        
        #Converting first flash lat/lon to radians
        fl_lat_rad = cur_fl_lat * (np.pi/180)
        fl_lon_rad = cur_fl_lon * (np.pi/180)

        ff_latlons = np.reshape([fl_lat_rad,fl_lon_rad], (-1, 2))
        
        #Implement a Ball Tree to capture the maximum and 95th percentiles within the range of 20km
        btree = BallTree(mrms_latlons, leaf_size=2, metric='haversine')
        indicies = btree.query_radius(ff_latlons, r = max_range/R)
        idx = indicies[0]
        if len(idx)==0:
            mrms_data_max = np.nan
            mrms_data_95 = np.nan
        else:
            mrms_data_max = np.nanmax(mrms_data_search[idx])
            mrms_data_95 = np.nanpercentile(a=mrms_data_search[idx], q=95)
    
    return mrms_data_max, mrms_data_95


# In[ ]:


# Saving the DataFrame out as a CSV
def mrms_data_saver(df, t_start, t_end, version):
    global glm_number


    y, m, d, doy, hr, mi = datetime_converter(t_start)
    output_folder = y+m+d
    start_time_str = 's'+y+m+d+hr+mi
    y, m, d, doy, hr, mi = datetime_converter(t_end)
    end_time_str = 'e'+y+m+d+hr+mi
    y, m, d, doy, hr, mi = datetime_converter(datetime.now())
    cur_time_str = 'c'+y+m+d+hr+mi
    
    output_loc = '/localdata/first-flash/data/MRMS-processed-GLM'+glm_number+'-v'+str(version)+'/'+output_folder +'/'
    output_file = 'MRMS-ff-v'+str(version)+'-'+start_time_str+'-'+end_time_str+'-'+cur_time_str+'.csv'
    if not os.path.exists(output_loc):
        os.makedirs(output_loc, exist_ok=True)
    df.to_csv(output_loc+output_file)
    print (output_loc+output_file)


# # Work Zone

# In[2]:


#Importing the data from the files
args = sys.argv
nc_file_loc = args[1]
# nc_file_loc = '../../local-data/2022/GLM16_first-flash-data-all_v32_s202201010000_e202212310000_c202409021440.nc' #DEVMODE

#Getting the necessary information from the netCDF file
nc_dset = nc.Dataset(nc_file_loc,'r')

#Setting up the time range
start_time = datetime.strptime(nc_dset.time_coverage_start, '%Y-%m-%d %H:%M:%S')
#start_time = datetime.strptime('2022-01-19 00:00:00', '%Y-%m-%d %H:%M:%S') #DEVMODE
end_time = datetime.strptime(nc_dset.time_coverage_end, '%Y-%m-%d %H:%M:%S')
time_list_days = pd.date_range(start=start_time, end=end_time, freq='1D') #Daily list to loop through

#GLM Number for later
glm_number = nc_dset.glm_number

#Getting the flash ids, lats, and lons for searching later...
fistart_flid = nc_dset.variables['flash_fistart_flid'][:]
f_lat = nc_dset.variables['flash_lat'][:]
f_lon = nc_dset.variables['flash_lon'][:]

#Getting the flash times for seraching later...
fistart_str = [i[0:15] for i in nc_dset.variables['flash_fistart_flid'][:]]
f_time = GLM_LCFA_times_postprocess(fistart_str, nc_dset.variables['flash_time_offset_of_first_event'][:])


# In[12]:


#Outmost loop for a daily basis
for i in range(len(time_list_days)-1):
    t_range_start = time_list_days[i]
    t_range_end = time_list_days[i+1]
    
    #Breaking the day into 12, 2-hour chunks
    tlist_starmap = pd.date_range(start=t_range_start, end=t_range_end, freq='2H')

    #Getting the time for the MRMS file string
    y, m, d, doy, hr, mi = datetime_converter(t_range_start)
    print ('---'+y+m+d+'---')
    
    #Getting the location of the MRMS data (in its daily folder)
    fstring_start = '/raid/swat_archive/vmrms/CONUS/'+y+m+d+'/multi/'
    
    #Sending the file string to the mrms_driver function that takes over from here...
    if __name__ == "__main__":
        with mp.Pool(12) as p:
            p.starmap(mrms_driver, zip(tlist_starmap[:-1], tlist_starmap[1:]))
            p.close()
            p.join()
    

