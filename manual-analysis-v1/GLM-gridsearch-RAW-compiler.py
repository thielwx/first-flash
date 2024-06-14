#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#======================================
# This script takes in all of the raw gridsearch files and combines them into one csv per case 
# (adapted from GLM-gridsearch-RAW-compiler.py in GLM/)
#
# Author: Kevin Thiel (kevin.thiel@ou.edu)
# Created: June 2024
#======================================


# In[19]:


import pandas as pd
from datetime import datetime
from glob import glob
import numpy as np
import yaml


# In[10]:
#Importing the appropriate yaml file
with open('case-settings-manual-analysis.yaml', 'r') as f:
    sfile = yaml.safe_load(f)

cases = sfile['cases']

combos = pd.read_csv('ff_gridsearch_combinations.csv', index_col=0)

glm_sat = 16


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


def file_list_creator(glm_sat, combo_num, case):
    file_loc = '/localdata/first-flash/data/manual-analysis-v1/'+case+'/GLM'+str(glm_sat)+'_ffRAW_v'+str(combo_num).zfill(2)+'/'
    
    file_list = [] #Empty file list that we'll fill
        
    collected_files = sorted(glob(file_loc+'/*.csv'))
    print (file_loc+'/*.csv')
    
    if len(collected_files)==0:
        print ('ERROR: NO FILES FOUND')
    
    file_list = np.append(file_list,collected_files)

    return file_list


# In[ ]:


def file_loader(file_list, search_m, search_r, search_flash_r, ver):
    #Loading in all of the data
    df = pd.DataFrame()
    for f in file_list:
        new_df = pd.read_csv(f, index_col=0)
        df = pd.concat((df,new_df))
    
    
    #Adding search information to the dataframe
    df['search_m'] = np.full(df.shape[0], search_m)
    df['search_r'] = np.full(df.shape[0], search_r)
    df['search_flash_r'] = np.full(df.shape[0], search_flash_r)
    df['search_version'] = np.full(df.shape[0], ver)
    
    return df


# In[6]:
#Outer loop for a case-by-case basis
for case in cases[:1]:
    
    #Getting the case times
    start_time_str = sfile[case]['start_time']
    end_time_str = sfile[case]['end_time']
    #Converting the time strings to datetimes
    start_time = datetime.strptime(start_time_str, '%Y%m%d-%H%M')
    end_time = datetime.strptime(end_time_str, '%Y%m%d-%H%M')

    df = pd.DataFrame() #Empty dataframe that will be filled with the first flash events

    #Inner loop is for the ff serach combinations
    for i in combos.index:
        #Grabbing the current serach criterion
        search_m = combos['minutes'][i]
        search_r = combos['simple_radius'][i]
        search_flash_r = combos['flash_area_radius'][i]
        
        #Get the list of files
        file_list = file_list_creator(glm_sat, i, case)
        
        if len(file_list)>0:
            #Loading the csv files
            new_df = file_loader(file_list, search_m, search_r, search_flash_r, i)
            
            #Adding the data to the dataframe
            df = pd.concat((df,new_df))

    #Saving the finished 
    save_loc = '/localdata/first-flash/data/manual-analysis-v1/'+case+'/'
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    df.to_csv(save_loc+ case+'-ffRAW-ALLcombos-v1.csv')

