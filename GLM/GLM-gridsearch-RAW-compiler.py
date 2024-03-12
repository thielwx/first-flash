#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#======================================
# This script takes in all of the raw gridsearch files and combines them into one csv
# Author: Kevin Thiel (kevin.thiel@ou.edu)
# Created: March 2024
#======================================


# In[19]:


import sys
import os
import pandas as pd
from datetime import datetime
from glob import glob
import numpy as np


# In[10]:


combos = pd.read_csv('ff_gridsearch_combinations.csv', index_col=0)

args = sys.argv
#args = ['BLANK', '20220322-perils', '16'] #devmode

case = args[1]
glm_sat = args[2]

if case == '20220322-perils':
    start_time = datetime(2022, 3, 22, 0, 0)
    end_time = datetime(2022, 3, 24, 0, 0)
    start_time_search = datetime(2022, 3, 22, 0, 0)
    end_time_search = datetime(2022, 3, 23, 6, 0)
    search_bounds = [37, 28, -83, -99]
    
elif case == '20220423-oklma':
    start_time = datetime(2022, 4, 23, 0, 0)
    end_time = datetime(2022, 4, 25, 0, 0)
    start_time_search = datetime(2022, 4, 23, 21, 0)
    end_time_search = datetime(2022, 4, 24, 12, 0)
    search_bounds = [37, 33, -92, -100]
    
t_list = pd.date_range(start=start_time, end=end_time, freq='1D')


# In[21]:


def latlon_bounds_custom(flash_lats, flash_lons, search_bounds):
    '''
    Takes in the flash latitudes and longitudes determines which ones are within the domain
    PARAMS:
        flash_lats: array of flash latitudes (floats)
        flash_lons: array of flash longitudes (floats) 
    RETURNS:
        latlon_locs: array of indicies from which the input lat/lon values are within the domain
        flash_lats: array of flash lats within the domain
        flash_lons: array of flash lons within the domain
    '''
    lat_max = search_bounds[0]
    lat_min = search_bounds[1]
    lon_max = search_bounds[2]
    lon_min = search_bounds[3]
    
    latlon_locs = np.where((flash_lats<=lat_max)&(flash_lats>=lat_min)&(flash_lons<=lon_max)&(flash_lons>=lon_min))[0]
    
    return latlon_locs


# In[ ]:


def file_list_creator(glm_sat, t_list, combo_num):
    file_loc = '/localdata/first-flash/data/GLM-gridsearch-1/GLM'+glm_sat+'_ffRAW_v'+str(combo_num).zfill(2)+'/'
    
    file_list = [] #Empty file list that we'll fill
    
    for t in t_list[:-1]:
        t_str = t.strftime('%Y%m%d')
        
        collected_files = sorted(glob[file_loc+t_str+'/*.csv'])
        
        if len(collected_files)==0:
            print ('ERROR: NO FILES FOUND')
            print (file_loc+t_str+'/*.csv')
        
        file_list = np.append(file_list,collected_files)


# In[ ]:


def file_loader(file_list, search_bounds, start_time_search, end_time_search, search_m, search_r, search_flash_r, ver):
    #Loading in all of the data
    df = pd.DataFrame()
    for f in file_list:
        new_df = pd.read_csv(f)
        df = pd.concat((df,new_df))
        
    #Cutting down the dataframe by time bounds
    flash_time = [np.datetime64(time) for time in df['start_time']]
    df = df.loc[(flash_time>=start_time_search)&(flash_time<=end_time_search)]
    #Cutting down the dataframe by spatial bounds
    df = df.iloc[latlon_bounds_custom(df['lat'].values, df['lon'].values ,search_bounds)]
    
    #Adding search information to the dataframe
    df['search_m'] = np.full(df.shape[0], search_m)
    df['search_r'] = np.full(df.shape[0], search_r)
    df['search_flash_r'] = np.full(df.shape[0], search_flash_r)
    df['search_version'] = np.full(df.shape[0], ver)
    
    return df


# In[6]:


df = pd.DataFrame() #Empty dataframe that will be filled with the first flash events
#Outer loop is for the ff serach combinations
for i in combos.index:
    #Grabbing the current serach criterion
    search_m = combos['minutes'][i]
    search_r = combos['simple_radius'][i]
    search_flash_r = combos['flash_area_radius'][i]
    
    #Get the list of files
    file_list = file_list_creator(glm_sat, t_list, i)
    #Loading the csv files
    new_df = file_loader(file_list, search_bounds, start_time_search, end_time_search, search_m, search_r, search_flash_r, i)
    
    #Adding the data to the dataframe
    df = pd.concat((df,new_df))

#Saving the finished dataframe
df.to_csv('/localdata/first-flash/data/GLM-gridsearch-1/'+case+'-flashes-combos.csv')

