#!/usr/bin/env python
# coding: utf-8

# In[1]:

#=======================================================================================
# This script is designed to read in LMA data from a variety of sources
# and output daily csv files for the flashes and sources. The LMA data
# are flash sorted from lmatools.
#
# Author: Kevin Thiel (kevin.thiel@ou.edu)
# Creation date: December 2023
#
# INPUTS:
#   start date: YYYYMMDD
#   end date: YYYYMMDD (not inclusive)
#   lma source: OK-LMA / perils-LMA / LEE-LMA / NAL-LMA / NGA-LMA (list as of Feb 2024)
#
# OUTPUTS:
#   DataFrames as csv files containing all of the LMA sources and flashes (min 10 sources)
#   Note: You can change the end of the script if you only want to output just flashes or sources
#=======================================================================================


import sys
import os
from glob import glob
import pandas as pd
import h5py
from datetime import datetime
import lma_function_master_v1 as lma_kit
import numpy as np




def lma_file_finder(time, data_loc, lma_string):
    '''
    Finding the lma files for a given date
    PARAMS:
        time: The date of interest (DateTime)
        data_loc: The base file string that contains the data (str)
        lma_string: The specified lma
    RETURNS:
        collected_files: A list of collected strings from the glob function with the data (str)
    '''
    y, m, d, doy, hr, mi = lma_kit.datetime_converter(time) #Get current date as string
    if (lma_string == 'OK-LMA') or (lma_string == 'LEE-LMA'):
        file_loc_end = y + '/' + m + '/' + d +'/'
    elif (lma_string == 'NGA-LMA') or (lma_string == 'NAL-LMA'):
        file_loc_end = y + m + d + '/'
    elif (lma_string == 'perils-LMA'):
        file_loc_end = y[-2:] + m + d + '/'
    else:
        print ('ERROR: Undefined file_loc_end')
        exit()


    file_loc_str = data_loc+file_loc_end + '*.h5'
    #print (file_loc_str)
    collected_files = glob(file_loc_str)
    
    return collected_files


# In[8]:


def lma_file_reader(file_str, t_start_dt, flash_df, event_df, n_source_min):
    '''
    A function that opens an h5 file using h5py, and extracting the flashes and events (10 source min)
    PARAMS:
        file_str: String that leads to the file location
        t_start_dt: DateTime object for the start of the file
        flash_df: Current dataframe with LMA flashes
        event_df: Current dataframe with LMA events
        n_source_min: Estabilished number of minimum sources (10)
    RETURNS:
        flash_df: Output of the LMA flashes
        event_df: Output of the LMA 
    '''
        
    y, m, d, doy, hr, mi = lma_kit.datetime_converter(t_start_dt) #Get current date as string
    
    
    print (file_str)
    #Loading in the file
    f = h5py.File(file_str,'r')
    
    #Getting the lma flashes as a pandas dataframe
    new_flash_df = lma_kit.lma_data_puller(f,'flashes')
    
    #Reducing our search to only the flashes with at least 10 sources
    new_flash_df = new_flash_df.loc[new_flash_df['n_points'] >= n_source_min]
    
    #If there's no more flashes left (assuming all noise) going to the next file
    # Not now we'll just skip all this and return the original dataframes
    if new_flash_df.shape[0] > 0:
        
        #Getting an array of flash ids
        flash_ids = new_flash_df['flash_id'].values
        
        #Getting the lma events (sources) as a pandas dataframe
        new_event_df = lma_kit.lma_data_puller(f,'events')
        
        #Reducing the event dataframe to only the flashes left in that dataframe (>=10 sources)
        new_event_df = new_event_df.loc[new_event_df['flash_id'].isin(flash_ids)]
        
        #Getting a time that is unique to the file (since the flash ids reset)
        file_time_str = '20'+file_str[-31:-18]
        
        #Creating a list of file strings and adding them to the dataframes
        event_file_str = [file_time_str for i in range(new_event_df.shape[0])]
        flash_file_str = [file_time_str for i in range(new_flash_df.shape[0])]
        new_event_df['file_time'] = event_file_str
        new_flash_df['file_time'] = flash_file_str
        
        #Adding the dataframes to the larger dataframe
        flash_df = pd.concat((flash_df,new_flash_df),axis=0)
        event_df = pd.concat((event_df,new_event_df),axis=0)
        
    return flash_df, event_df


# In[9]:


def file_output_creator(t_start_dt, n_source_min, output_data_loc, lma_string, data, df):
    '''
    A function that saves the LMA data (flashes and events) as csv files
    PARAMS:
        t_start_dt: DateTime object for the start of the file (datetime)
        n_source_min: Estabilished number of minimum sources (int, 10)
        output_data_loc: Defined location that the csv files should go (str)
        lma_string: Beginning of the lma string for the output file name (str)
        data: Data type (events or flashes) (str)
        df: The dataframe being saved (DataFrame)
    RETURNS:
        None
    '''
    
    #Get current date/date and times as strings
    y, m, d, doy, hr, mi = lma_kit.datetime_converter(t_start_dt) 
    file_time_str = y+m+d+hr+mi
    file_date_str = y+m+d
    
    #Getting the current time as a string
    y, m, d, doy, hr, mi = lma_kit.datetime_converter(datetime.now()) #Get current date as string
    cur_time_str = 'c'+y+m+d+hr+mi
    
    #Creating the strings for the file location and the csv file name
    file_save_str = lma_string+'_RAW-'+data+'_'+file_time_str+'_'+cur_time_str+'_source-min-'+str(n_source_min)+'.csv'
    file_loc_str = output_data_loc+file_date_str+'/'
    
    #Showing what is being saved and ensuring a path exists
    print (file_loc_str+file_save_str)
    if not os.path.exists(file_loc_str):
        os.makedirs(file_loc_str)
        
    #Outputting the dataframe as a string
    df.to_csv(file_loc_str+file_save_str)
    
    return 0


# In[ ]:


datetime_start = datetime.now()

#==========================
# Run like this python lma-sorted-flash-loader_v2.py 20220501 20220502 OK-LMA
#==========================

#args = ['','20220504', '20220505', 'OK-LMA']
args = sys.argv

t_start = args[1] #Start time
t_end = args[2] #End time (does not include data from that data, just up to it)
lma_string = args[3]

#Getting the list of dates we need
t_start_dt = datetime.strptime(t_start, '%Y%m%d') #Converting into datetimes
t_end_dt = datetime.strptime(t_end, '%Y%m%d') #Converting into datetimes
time_list = pd.date_range(start=t_start_dt, end=t_end_dt, freq='D').to_list()

#Variables 
n_source_min = 10 # Minimum number of sources
output_data_loc = '/localdata/first-flash/data/'+lma_string+'-RAW/'

#Source of the flash sorted LMA files
#If not running on a local machine you will have to change these
#to point to your local directory
if lma_string == 'OK-LMA':
    data_loc = '/raid/lng1/flashsort/h5_files/'
elif lma_string == 'perils-LMA':
    data_loc = '/raid/lng1/analyzed_data_v10/deployments/flashsort/h5_files/'
elif lma_string == 'LEE-LMA':
    data_loc = '/raid/lng1/analyzed_data_v10/deployments/flashsort/h5_files/'
elif lma_string == 'NAL-LMA':
    data_loc = '/localdata/first-flash/data/NAL-LMA-h5_files/'
elif lma_string == 'NGA-LMA':
    data_loc =  '/localdata/first-flash/data/NGA-LMA-h5_files/'
else:
    print('ERROR: Invalid lma network entered, pick from the following:/n OK-LMA / perils-LMA / LEE-LMA / NAL-LMA / NGA-LMA')
    exit()


# In[ ]:

flash_df = pd.DataFrame() #Empty dataframe that we're filling with the flashes
event_df = pd.DataFrame() #Empty dataframe that we're filling with the sources (events)

#Outer loop so we're running the data on a daily basis
for i in range(len(time_list)-1): #Looping through the files on a daily basis
    print (time_list[i])

    lma_files = lma_file_finder(time_list[i], data_loc, lma_string) #Getting the list of file strings to then read them in
    
    if len(lma_files) == 0: #A check that we actually have data, if not then skip this date
        print ('NO FILES FOUND: '+str(time_list[i]))
        continue
    
    #Inner loop so we're appending the data from each file to the dataframes
    for file_str in lma_files:
        flash_df, event_df = lma_file_reader(file_str, time_list[i], flash_df, event_df, n_source_min)
    
#Outputting the dataframes as csv's for all days
file_output_creator(t_start_dt, n_source_min, output_data_loc, lma_string, 'flash', flash_df)    
file_output_creator(t_start_dt, n_source_min, output_data_loc, lma_string, 'event', event_df) 

print('RAW Files Loaded')
print (datetime.now()-datetime_start)

