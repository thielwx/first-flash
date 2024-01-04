#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from datetime import datetime
from glob import glob
import first_flash_function_master as ff

# In[ ]:


def raw_finder(start_time, glm_sat, data_loc, search_r, search_m, ver):
    #data_loc = '../../test-data/' #Dev Mode
    
    #converting the datetime objects to string pieces
    y, m, d, doy, hr, mi = ff.datetime_converter(start_time)
    #Making the file string
    file_str_start = 'GLM' + str(glm_sat) + '_ffRAW_r' + str(search_r) + '_t' + str(search_m) + '_v' + str(ver) + '_'
    file_str_end = 's' + y + m + d + '*.csv'
    #Getting the file string
    file_loc = data_loc + 'GLM' + str(glm_sat) + '_ffRAW_v' + str(ver) + '/' + y + m + d + '/'
    total_file_str = file_loc + file_str_start + file_str_end 
    
    #Searching for the files using glob
    glob_files = glob(total_file_str)
    
    ff_df = pd.DataFrame()
    
    for files in glob_files:
        new_df = pd.read_csv(files)

        #Included for when the first flash file is empty
        if new_df.shape[0]>0:
            new_df['start_time'] = pd.to_datetime(new_df['start_time'])
            ff_df = pd.concat((ff_df,new_df))
    
    return ff_df


# In[ ]:


#User inputs
start_time_str = '20220501'
end_time_str = '20220601'
glm_sat = '17'

#Converting into datetimes
start_time = datetime.strptime(start_time_str, '%Y%m%d') 
end_time = datetime.strptime(end_time_str, '%Y%m%d') 
#Creating list of dates to pull from when compiling the dataset
date_list = pd.date_range(start=start_time,end=end_time,freq='1D',inclusive='left').to_list()

#Defined constants
search_r = 30
search_m = 30
ver = 2
data_loc = '/localdata/first-flash/data/'
#data_loc = '../../test-data/GLM16_ffRAW_v2/20220504/' #Dev Mode


# In[ ]:


#Loading in the raw files
ff_df = pd.DataFrame()

#Looping through each date requested to find the glm first flashes and incldue them in the dataset
for stime in date_list:
    print (stime)
    #Compiling the dataframe from the entire day of first flash events
    new_df = raw_finder(stime, glm_sat, data_loc, search_r, search_m, ver)
    ff_df = pd.concat((new_df,ff_df))
    
ff_df.to_pickle('202205-GLM17-ff_RAW-compiled.pkl')

