#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#=================================================================================================
# This script takes in ENI files (24.5 hrs) and finds the first ENI flashes 
# based on distance and time (30 minutes, 30 km). Data is output as csv files 
# as 'raw' data to be compiled later (so we can capture all the group and event data)
# Note: Script was adapted from GLM-ff-raw-creator-v2.py
#
# Author: Kevin Thiel
# Created: January 2024
# Email: kevin.thiel@ou.edu
# 
#
# Inputs:
#      start time YYYYmmddHHMM
#      end time YYYYmmddHHMM
#================================================================================================


# In[1]:


import eni_function_master_v1 as efm
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
import json
import multiprocessing as mp
import sys
import os


# In[2]:


args = sys.argv
#args = ['BLANK','202203200100','202203200200'] #casename,start date, end date, glm name #DEV MODE

start_time_str = args[1]
end_time_str = args[2]

#Getting the start and end time as datetime objects
start_time = datetime.strptime(start_time_str, '%Y%m%d%H%M')
end_time = datetime.strptime(end_time_str, '%Y%m%d%H%M')

#Search criterion
search_r = 30 #km
search_m = 30 #minutes
ver = 1
delta_t = timedelta(minutes=search_m)

#Data Location String
#data_loc = '../../test-data/ENI-test/' #DEVMODE
data_loc = '/raid/lightning-archive/ENI_CSV/flash/' #NEED TO FILL OUT BEFORE RUNNING ON DEVLAB4


# In[3]:


def df_formatter(df):
    '''
    A funciton that formats the data as we need
    '''
    #Adding radian data and a coutner as the index
    df['lat_rad'] = df['Latitude'].values * np.pi/180
    df['lon_rad'] = df['Longitude'].values * np.pi/180
    df['counter'] = np.arange(0,df.shape[0])
    df = df.set_index('counter')
    
    #meta_keys = ['st', 'et', 'v', 'ns', 'im', 'cm', 'aa', 'ia', 'ao', 'io', 'd', 's']
    #converted_keys = ['start_time','end_time','version','number_stations','ic_mult','cg_mult','max_lat','min_lat','max_lon','min_lon','d','network']
    
    return df
        


# In[4]:


def eni_ff_driver(s_time , e_time):
    '''
    Main function used to control the other functions
    INPUTS:
        s_time: starting time (datetime)
        e_time: ending time (datetime)
    RETUNS:
        None
    '''
    #Grabbing all the constant variables
    global search_r
    global search_m
    global ver
    global eni_df
    global delta_t
    
    #Making sure we actually have data to search for. If not then we skip it!
    #if eni_df.shape[0]>0:
    #Finding the first flash events over the given time period
    ff_df = efm.eni_ff_hunter(eni_df, s_time, e_time, search_r, search_m)
        
    #    if ff_df.shape[0]>0:
    #Saving that file as a csv
    ff_raw_saver(ff_df, s_time, e_time, ver, search_r, search_m)   


# In[5]:


def ff_raw_saver(ff_df, s_time, e_time, version, search_r, search_m):
    '''
    A function for saving out the raw files
    PARAMS:
        ff_df: The pandas DataFrame that has the first-flashes in them
        s_time: Start time (DateTime)
        e_time: End time (DateTime)
        version: Version of the output (str)
        search_r: Search radius in km (int)
        search_m: Search time period in minutes (int)
    RETURNS:
        None
    '''
    #Creating the start time string for the current file
    y, m, d, doy, hr, mi = efm.datetime_converter(s_time)
    stime_str = 's'+y+m+d+hr+mi
    output_date_str = y+m+d
    
    #Creating the end time string for the current file
    y, m, d, doy, hr, mi = efm.datetime_converter(e_time)
    etime_str = 'e'+y+m+d+hr+mi
    
    #Creating the current time string for the current time
    y, m, d, doy, hr, mi = efm.datetime_converter(datetime.now())
    ctime_str = 'c'+y+m+d+hr+mi
    
    #The beginning string for the file name
    front_string = 'ENI_ffRAW_r'+str(search_r)+'_t'+str(search_m)+'_v'+str(version)
    
    #Creating the entire save string
    save_str = front_string+'_'+stime_str+'_'+etime_str+'_'+ctime_str+'.csv'
    
    save_loc = '/localdata/first-flash/data/ENI_ffRAW_v'+str(version)+'/'+output_date_str+'/'
    #save_loc = './' #devmode
    
    #We only need to save out a subset of the data so we can extact it later
    ff_df = ff_df[['Lightning_Time_String','Latitude','Longitude','File_String']]
    
    #Time to save the dataframe!
    print (save_loc+save_str)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    ff_df.to_csv(save_loc+save_str)


# In[6]:


def first_flash_multiprocessor(start_time, end_time):
    '''
    Breaking down the daily GLM data into chunks that can be processed by ff_hunter
    PARAMS:
        start_time: DateTime
        end_time: DateTime
    RETURNS:
        None
    '''
    #Making a list of times in two hour chunks
    time_list = pd.date_range(start=start_time, end=end_time, freq='2H')
    
    if __name__ == "__main__":
        with mp.Pool(12) as p:
            p.starmap(eni_ff_driver,zip(time_list[:-1],time_list[1:]))
            p.close()
            p.join()


# # Driver Section to run the above code

# In[7]:


#Loading in the data
eni_df = efm.eni_loader(start_time-delta_t, end_time, data_loc)
print ('File List Created')

#Formatting the data with some extra data
eni_df = df_formatter(eni_df)


#eni_ff_driver(start_time, end_time) #DEVMODE
first_flash_multiprocessor(start_time, end_time)
print ('First Flashes Found')


# In[ ]:




