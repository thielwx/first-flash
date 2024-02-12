#!/usr/bin/env python
# coding: utf-8

# In[1]:


#=================================================================================================
# This script commpiles all GLM flashes from a specified period of time, and outputs basic data
# (lat,lon,time) as a csv file to use for analysis later.
# Note: Script was adapted from GLM-ff-raw-creator-v2.py
#
# Author: Kevin Thiel
# Created: February 2024
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
dt = timedelta(minutes=search_m)
ver = 1



def ff_raw_saver(df, s_time, e_time, version, glm_sat):
    '''
    A function for saving out the compiled files
    PARAMS:
        df: The pandas DataFrame that has the all the flashes in them
        s_time: Start time (DateTime)
        e_time: End time (DateTime)
        version: Version of the output (str)
        glm_sat: GOES GLM number (16, 17, 18, 19)
    RETURNS:
        None
    '''
    #Creating the start time string for the current file
    y, m, d, doy, hr, mi = ff.datetime_converter(s_time)
    stime_str = 's'+y+m+d+hr+mi

    #Creating the start time string for the current file
    y, m, d, doy, hr, mi = ff.datetime_converter(e_time)
    etime_str = 's'+y+m+d+hr+mi
    
    #Creating the current time string for the current time
    y, m, d, doy, hr, mi = ff.datetime_converter(datetime.now())
    ctime_str = 'c'+y+m+d+hr+mi
    
    #The beginning string for the file name
    front_string = 'GLM'+str(glm_sat)+'allflashes_v'+str(version)
    
    #Creating the entire save string
    save_str = front_string+'_'+stime_str+'_'+etime_str+'_'+ctime_str+'.csv'
    
    save_loc = '/localdata/first-flash/data/GLM'+str(glm_sat)+'-cases-allflash/'
    #save_loc = './' #devmode
    
    print (save_loc+save_str)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    df.to_csv(save_loc+save_str)


# In[7]:

# # Driver section

# In[8]:

#Loading the data
file_list = ff.data_loader_list(start_time=start_time-dt, end_time=end_time+dt, glm_sat=glm_sat)
print ('File List Created')

df = ff.data_loader(file_list)
print ('DataFrame Created')

#Saving the data
ff_raw_saver(s_time, e_time, ver, glm_sat)