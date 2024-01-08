#!/usr/bin/env python
# coding: utf-8

# Base code to work with ENI data

# In[24]:


import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from glob import glob
import global_land_mask as globe


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


# In[15]:


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


# In[2]:


def dset_land_points(flash_lats,flash_lons):
    '''
    Takes in the flash latitudes and longitudes to find the ones that are on land
    PARAMS:
        flash_lats: array of flash latitudes (floats)
        flash_lons: array of flash longitudes (floats)
    RETURNS:
        land_index: array of indexes that are classified as on land (ints)
    '''
    #Using the is_land method to find the indicies from the arrays that are on land
    land_index = globe.is_land(flash_lats, flash_lons)
    
    return land_index


# In[28]:


def eni_loader(start_time, end_time, input_loc):
    '''
    A function for collecting ENI data from a specified time period and range (CONUS), and putting it into a single dataframe
    PARAMS:
        start_time: Beginning of the time period (datetime object)
        end_time: End of the time period (datetime object)
        input_loc: Location of the ENI data (str)
    RETURNS:
        df: The compiled flash data from the csv files (DataFrame)
    '''
    #Creating a list of times to rotate through
    time_list = pd.date_range(start=start_time, end=end_time,freq='1min').to_list()
    
    #Empty DataFrame that we'll fill
    df = pd.DataFrame(columns=('Lightning_Time_String','Latitude','Longitude','Height','Flash_Type','Amplitude','Flash_Solution','Confidence'))
    
    #Looping through the list of available times
    for cur_time in time_list:
        y,m,d,doy,hr,mi = datetime_converter(cur_time) #Turning the current time into a string
        
        #Specifying the folder by date
        file_loc = input_loc + y + m + d + '/'
        file_loc = input_loc #DEVMODE
        
        #Creating the file string we'll use in the glob function
        file_str = y + m + d + 'T' + hr + mi + '.csv'
        
        #Collecting the files on the given day
        collected_file = sorted(glob(file_loc+file_str))
        
        if len(collected_file)==0:
            print (cur_time)
            print (file_loc+file_str)
            print ('ERROR: NO FILE FOUND')
            continue
        
        #Reading in the collected file
        new_df = pd.read_csv(collected_file[0])
        
        #Removing the data outside of the domain of study
        bound_index = latlon_bounds(new_df['Latitude'].values, new_df['Longitude'].values)
        new_df = new_df.iloc[bound_index,:]
        
        #Appending the new DataFrame to the combined one
        df = pd.concat((df,new_df),axis=0)
        
    return df
        
