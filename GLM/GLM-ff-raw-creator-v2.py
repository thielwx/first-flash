#!/usr/bin/env python
# coding: utf-8

# In[1]:


#=================================================================================================
# This script takes in GLM L2 LCFA files (25 hrs) and finds the first GLM flashes 
# based on distance and time (30 minutes, 30 km). Data is output as csv files 
# as 'raw' data to be compiled later (so we can capture all the group and event data)
# Note: Script was adapted from ff-playground-v2-1dayofglm-local.py
#
# Author: Kevin Thiel
# Created: October 2023
# Email: kevin.thiel@ou.edu
# 
#
# Inputs:
#      start time YYYYmmddHHMM
#      end time YYYYmmddHHMM
#      GLM number
#================================================================================================

# In[9]:


import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import sys
import first_flash_function_master as ff
import multiprocessing as mp
import os
from glob import glob

# # Input variables

# In[3]:


args = sys.argv
#args = ['BLANK','202205040000','202205050000','16'] #casename,start date, end date, glm name #DEV MODE

start_time_str = args[1]
end_time_str = args[2]
glm_sat = args[3]

#Converting the time strings to datetimes
start_time = datetime.strptime(start_time_str, '%Y%m%d%H%M')
end_time = datetime.strptime(end_time_str, '%Y%m%d%H%M')


#Constants
search_r = 30
search_m = 30
dt = timedelta(minutes=search_m)
ver = 2


# # Function Land

# In[4]:


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
            p.starmap(ff_driver,zip(time_list[:-1],time_list[1:]))
            p.close()
            p.join()


# In[5]:


def ff_raw_saver(ff_df, s_time, e_time, version, glm_sat, search_r, search_m):
    '''
    A function for saving out the raw files
    PARAMS:
        ff_df: The pandas DataFrame that has the first-flashes in them
        s_time: Start time (DateTime)
        e_time: End time (DateTime)
        version: Version of the output (str)
        glm_sat: GOES GLM number (16, 17, 18, 19)
        search_r: Search radius in km (int)
        search_m: Search time period in minutes (int)
    RETURNS:
        None
    '''
    #Creating the start time string for the current file
    y, m, d, doy, hr, mi = ff.datetime_converter(s_time)
    stime_str = 's'+y+m+d+hr+mi
    output_date_str = y+m+d
    
    #Creating the current time string for the current time
    y, m, d, doy, hr, mi = ff.datetime_converter(datetime.now())
    ctime_str = 'c'+y+m+d+hr+mi
    
    #The beginning string for the file name
    front_string = 'GLM'+str(glm_sat)+'_ffRAW_r'+str(search_r)+'_t'+str(search_m)+'_v'+str(version)
    
    #Creating the entire save string
    save_str = front_string+'_'+stime_str+'_'+ctime_str+'.csv'
    
    save_loc = '/localdata/first-flash/data/GLM'+str(glm_sat)+'_ffRAW_v'+str(version)+'/'+output_date_str+'/'
    #save_loc = './' #devmode
    
    print (save_loc+save_str)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    ff_df.to_csv(save_loc+save_str)


# In[7]:


def ff_driver(s_time, e_time):
    #Grabbing all the extra variables
    global search_r
    global search_m
    global ver
    dt = timedelta(minutes=search_m)
    global df
    global glm_sat
    
    #Cutting down the initial dataframe to something smaller 
    #df_cutdown = df.loc[(df['start_time']>=s_time-dt)&(df['start_time']<=e_time+dt)] I don't think we need this?
    #Getting the first flashes for the time period of interest, output as a dataframe
    ff_df = ff.ff_hunter(df, s_time, e_time, search_r, search_m)
    
    ff_df = ff.ff_next_flashes(df, ff_df, s_time, e_time, search_r, search_m)
    
    ff_df.index.names['fistart_flid'] #Trying to force the index to take on the correct names before saving it

    ff_raw_saver(ff_df, s_time, e_time, ver, glm_sat, search_r, search_m)


# # Driver section

# In[8]:


file_list = ff.data_loader_list(start_time=start_time-dt, end_time=end_time+dt, glm_sat=glm_sat)
print ('File List Created')

df = ff.data_loader(file_list)
print ('DataFrame Created')

#df.to_pickle('20220504-ALL-GLM16.pkl')

first_flash_multiprocessor(start_time, end_time)

print ('First Flashes Found')
