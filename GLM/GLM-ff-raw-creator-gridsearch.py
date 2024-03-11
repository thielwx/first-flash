#!/usr/bin/env python
# coding: utf-8

# In[1]:


#=================================================================================================
# This script takes in GLM L2 LCFA files (25+ hrs) and finds the first GLM flashes 
# based on distance and time (varaible). Data is output as csv files 
# as 'raw' data to be compiled later (so we can capture all the group and event data)
# Note: Script was adapted from GLM-ff-raw-creator-v2.py
#
# Author: Kevin Thiel
# Created: March 2024
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

# Getting all of the combinations that we'll be searching
search_combos = pd.read_csv('ff_gridsearch_combinations.csv', index_col=0)

#Constants
dt = timedelta(minutes=50) #Set to the max possible serach time period
ver = 1


# # Function Land

# In[4]:


def first_flash_multiprocessor(start_time, end_time):
    '''
    Breaking down the daily GLM data into chunks that can be processed by ff_hunter
    PARAMS:
        start_time: DateTime
        end_time: DateTime
        search_m
        search_r
        search_flash_r
        i
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


def ff_raw_saver(ff_df, s_time, e_time, version, glm_sat, search_r, search_m, search_flash_r, i):
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
        search_flash_r
        i
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
    front_string = 'GLM'+str(glm_sat)+'_ffRAW_r'+str(int(search_r))+'_t'+str(int(search_m))+'_fr'+str(int(search_flash_r))+'_v'+str(i).zfill(2)
    
    #Creating the entire save string
    save_str = front_string+'_'+stime_str+'_'+ctime_str+'.csv'
    
    save_loc = '/localdata/first-flash/data/GLM-gridsearch-'+str(ver)+'/GLM'+str(glm_sat)+'_ffRAW_v'+str(i).zfill(2)+'/'+output_date_str+'/'
    
    print (save_loc+save_str)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    ff_df.to_csv(save_loc+save_str)


# In[7]:


def ff_driver(s_time, e_time):
    #Grabbing all the extra variables
    global ver
    global df
    global glm_sat
    global search_m
    global search_r
    global search_flash_r
    global i

    #Getting the first flashes for the time period of interest, output as a dataframe
    ff_df = ff.ff_hunter_gridsearch(df, s_time, e_time, search_r, search_m, search_flash_r)
    
    ff_df = ff.ff_next_flashes(df, ff_df, s_time, e_time, 30, 30) #Keeping those at 30 min/30 km for consistency
    
    ff_df.index.names['fistart_flid'] #Trying to force the index to take on the correct names before saving it

    ff_raw_saver(ff_df, s_time, e_time, ver, glm_sat, search_r, search_m, search_flash_r, i)


# # Driver section

# In[8]:


file_list = ff.data_loader_list(start_time=start_time-dt, end_time=end_time+dt, glm_sat=glm_sat)
print ('File List Created')

df = ff.data_loader_gridsearch(file_list)
print ('DataFrame Created')

#Looping through each combination of search criterion
for i in search_combos.index:
    search_m = search_combos['minutes'][i]
    search_r = search_combos['simple_radius'][i]
    search_flash_r = search_combos['flash_area_radius'][i]
    first_flash_multiprocessor(start_time, end_time, search_m, search_r, search_flash_r, i)

print ('First Flashes Found')
