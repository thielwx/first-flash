#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#=================================================================================================
# This script checks for ENI files in the NSSL ENI_CSV archive and outputs the missing files as a dataframe
# Script based on eni_ff_raw_creator_v1.py
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
import sys
import os
from glob import glob


# In[2]:


args = sys.argv
#args = ['BLANK','202203200100','202203200200'] #casename,start date, end date #DEV MODE

start_time_str = args[1]
end_time_str = args[2]

#Getting the start and end time as datetime objects
start_time = datetime.strptime(start_time_str, '%Y%m%d%H%M')
end_time = datetime.strptime(end_time_str, '%Y%m%d%H%M')

#Data Location String
#data_loc = '../../test-data/ENI-test/' #DEVMODE
data_loc = '/raid/lightning-archive/ENI_CSV/flash/' #NEED TO FILL OUT BEFORE RUNNING ON DEVLAB4


# In[5]:


def eni_missing_saver(ff_df, s_time, e_time):
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
    
    #Creating the end time string for the current file
    y, m, d, doy, hr, mi = efm.datetime_converter(e_time)
    etime_str = 'e'+y+m+d+hr+mi
    
    #Creating the current time string for the current time
    y, m, d, doy, hr, mi = efm.datetime_converter(datetime.now())
    ctime_str = 'c'+y+m+d+hr+mi
    
    #The beginning string for the file name
    front_string = 'ENI_CSV_missing'
    
    #Creating the entire save string
    save_str = front_string+'_'+stime_str+'_'+etime_str+'_'+ctime_str+'.csv'
    
    save_loc = '/localdata/first-flash/data/'
    #save_loc = './' #devmode
    

    #Time to save the dataframe!
    print (save_loc+save_str)
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    ff_df.to_csv(save_loc+save_str)



def eni_file_list_loader(start_time, end_time, input_loc):
    '''
    A function for collecting ENI data availability in the NSSL archive
    PARAMS:
        start_time: Beginning of the time period (datetime object)
        end_time: End of the time period (datetime object)
        input_loc: Location of the ENI data (str)
    RETURNS:
        df: The compiled missing files within the range (DataFrame)
    '''

    #Creating a list of times to rotate through
    time_list = pd.date_range(start=start_time, end=end_time,freq='1min').to_list()
    
    #Empty DataFrame that we'll fill
    df = pd.DataFrame(columns=('time','string'))
    
    #Looping through the list of available times
    for cur_time in time_list:
        y,m,d,doy,hr,mi = efm.datetime_converter(cur_time) #Turning the current time into a string
        
        if (d=='01') & (hr=='00') & (mi=='00'):
            print (y+m+d+'-'+hr+mi)

        #Specifying the folder by date
        file_loc = input_loc + y + m + d + '/'
        #file_loc = input_loc #DEVMODE
        
        #Creating the file string we'll use in the glob function
        file_str = y + m + d + 'T' + hr + mi + '.csv'
        
        #Collecting the files on the given day
        collected_file = sorted(glob(file_loc+file_str))
        
        if len(collected_file)==0:
            print (file_loc+file_str)
            #print ('ERROR: NO FILE FOUND')
            #No file found so we'll add that to the dataframe
            new_dictionary = {
                'time': [cur_time],
                'string': [file_str]
            }
            new_df = pd.DataFrame(data=new_dictionary) #dataframe that will be appened to file

            #Appending the new DataFrame to the combined one
            df = pd.concat((df,new_df),axis=0)
        
    return df


# # Driver Section to run the above code

# In[7]:


#Loading in the data
eni_df = eni_file_list_loader(start_time, end_time, data_loc)
print ('Missing File List Created')

eni_missing_saver(eni_df, start_time, end_time)
print ('Missing File List Saved')





