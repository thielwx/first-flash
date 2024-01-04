#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
from glob import glob
import numpy as np
from datetime import datetime
from datetime import timedelta
import gzip
import shutil
import lma_function_master_v1 as lma_kit
import subprocess as sp
import sys


# My goal is to create a dataframe that gives an overview of the number of active stations. The process will be to
# -  Create a list of LMA dat files from a specified time range
# 
# IN A LOOP
# -  Use that list to unzip/extract the .dat.gz files into a temporary file/folder
# -  Extract the number of active and available stations
# -  Delete the temporary file

# In[77]:


def file_str_finder(time_list, zip_file_loc):
    file_list = [] #Creating an empty list that you'll fill with the file names
    new_time_list = [] #Creating an empty list that you'll fill with times
    
    #Looping through the time list
    for cur_time in time_list[:-1]:
        y, m, d, doy, hr, mi = lma_kit.datetime_converter(cur_time) #Get current date as string
        
        #putting the file string together
        file_str_start = 'LYLOUT_'
        file_str_end = '_0600.dat.gz'
        file_str_mid = str(y[2:]) + str(m) + str(d) + '_' + str(hr) + str(mi) +'00'
        file_str = file_str_start + file_str_mid + file_str_end
        
        #Putting the file location string together
        file_loc_end = str(y)+'/'+str(m)+'/'+str(d)+'/'
        file_loc = zip_file_loc + file_loc_end
#         file_loc = '' #DEVMODE
        
        #Using glob to find the files
        collected_files = glob(file_loc+file_str)
        
        #Checking that we actually got the right number of files
        if len(collected_files) == 0:
            print ('FILE NOT FOUND: '+file_loc+file_str)
            continue
        elif len(collected_files) == 1:
            file_list = np.append(file_list, collected_files[0])
            new_time_list = np.append(new_time_list, cur_time)
        else:
            print ('MORE THAN ONE FILE FOUND:')
            print (file_list)
            continue
    
    #Returning the complete file list
    return file_list, new_time_list
            


# In[ ]:


#==========================
# Run like this python LMA-data-availability-df-creator-v1.py 202205010000 202205020000 OK-LMA
#==========================

args = sys.argv

t_start = args[1] #Start datetime
t_end = args[2] #End time (does not include data from that datetime, just up to it)
lma_str = args[3] #LMA type

#Getting the list of dates we need
start_time = datetime.strptime(t_start, '%Y%m%d%H%M') #Converting into datetimes
end_time = datetime.strptime(t_end, '%Y%m%d%H%M') #Converting into datetimes


# In[78]:


#Important file locations
zip_file_loc = '/raid/lng1/analyzed_data_v10/'
temp_file_loc = '/localdata/first-flash/data/LMA-temp/'
final_df_loc = '/localdata/first-flash/data/LMA-availability/'
#DEVMODE
# zip_file_loc = ''
# temp_file_loc = ''
# final_df_loc = ''

#Getting a list of times between our start and end times that run every 10 minutes
time_list = pd.date_range(start=start_time, end=end_time,freq='600S').to_list()

#Gettnig the list of files that coincide with the time list
file_list, new_time_list = file_str_finder(time_list, zip_file_loc)

#Setting up the DataFrame that we'll be filling
col_names = ['time','active_stations','total_stations']
df = pd.DataFrame(columns=col_names)

#Looping through the list of files
for i in range(len(file_list)):
    cur_file = file_list[i]
    cur_time = new_time_list[i]
    temp_file = temp_file_loc + cur_file[-32:-3] #Temporary file name with the .gz removed
    
    #Step 1: Upzip  the file and place its contents in the temporary file location
    with gzip.open(cur_file,'rb') as f_in:
        with open(temp_file, 'wb') as f_out:
            print (temp_file)
            shutil.copyfileobj(f_in, f_out)
    
    #Step 2: Open the temporary file and extract its contents to put into a dataframe
    file = open(temp_file, 'r') #Opening the file
    lines = file.readlines() #Reading the text file line by line
    num_total_stations = int(lines[11][20:]) #The number of total stations
    num_active_stations = int(lines[12][27:]) #The number of active stations
    
    #Puting the data into the dataframe
    d = {
        col_names[0]:[cur_time],
        col_names[1]: [num_active_stations],
        col_names[2]: [num_total_stations],
    }
    new_df = pd.DataFrame(data=d,index=[i])
    df = pd.concat((df,new_df),axis=0)
    
    #Step 3: Close and remove the temporary file
    file.close()
    #Using the rm command to remove the temporary file
    cmd = 'rm '+temp_file
    print (cmd)
    p = sp.Popen(cmd,shell=True)
    p.wait()   


# In[79]:


#Saving out the dataframe as a csv file

#All of this garbage is just getting file strings
y, m, d, doy, hr, mi = lma_kit.datetime_converter(time_list[0]) #Get first date as string
start_str = 's'+y+m+d+hr+mi
y, m, d, doy, hr, mi = lma_kit.datetime_converter(time_list[-1]) #Get last date as string
end_str = 'e'+y+m+d+hr+mi
y, m, d, doy, hr, mi = lma_kit.datetime_converter(datetime.now()) #Get current date as string
cur_str = 'c'+y+m+d+hr+mi

#Creating the file string
df_str = final_df_loc+lma_str+'_station-counts_'+start_str+'_'+end_str+'_'+cur_str+'.csv'
df.to_csv(df_str)

