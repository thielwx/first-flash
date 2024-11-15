#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ====================================================
# This script takes in the manual analysis first flash dataset and find the corresponding abi and mrms datasets
# Author: Kevin Thiel
# Created: November 2024
# ====================================================


# In[2]:


import pandas as pd
import numpy as np
import netCDF4 as nc
from glob import glob
from datetime import datetime
import ma_abi_mrms_functions as mamf


# In[3]:


# Constants
version = 1
abi_variables = ['CMIP','ACHA']
abi_variables_output = ['CMIP_min', 'CMIP_05', 'ACHA_max', 'ACHA_95']
abi_variables_output_random_sample = ['CMIP_min', 'CMIP_05', 'ACHA_max', 'ACHA_95', 'random_lat', 'random_lon', 'fistart_flid'] 
mrms_variables = ['MergedReflectivityQCComposite','Reflectivity_-10C','ReflectivityAtLowestAltitude']
mrms_variables_output = ['MergedReflectivityQCComposite_max','MergedReflectivityQCComposite_95','Reflectivity_-10C_max','Reflectivity_-10C_95','ReflectivityAtLowestAltitude_max','ReflectivityAtLowestAltitude_95']

   
# In[4]:


#Reading in the manually analyzed first flash dataset 
df = pd.read_csv('ff_v00.csv', index_col=0)
df['file_datetime'] = [datetime.strptime(row['fstart'][0:15],'s%Y%j%H%M%S0') for index, row in df.iterrows()]
#df = df.loc[df['case']=='20220423-ok'] #DEVMODE

# In[5]:


#Setting up the ABI and MRMS dataframes
abi_df = pd.DataFrame(index=df.index, columns=abi_variables_output)
abi_df['lat'] = df['lat']
abi_df['lon'] = df['lon']
abi_df['fstart'] = df['fstart']
abi_df['fistart_flid'] = df['fistart_flid']
abi_df['ma_category'] = df['ma_category']

mrms_df = pd.DataFrame(index=df.index, columns=mrms_variables_output)
mrms_df['lat'] = df['lat']
mrms_df['lon'] = df['lon']
mrms_df['fstart'] = df['fstart']
mrms_df['fistart_flid'] = df['fistart_flid']
mrms_df['ma_category'] = df['ma_category']


# In[16]:


#Function that collects the abi data
def abi_driver(abi, ff, row):
    #Getting the flash datetime (approx) from the file start time
    f_time = datetime.strptime(row['fstart'][0:15], 's%Y%j%H%M%S0')
    
    #Getting the ABI file times from the GLM flash time
    abi_time_pre0, abi_time_pre10 =  mamf.abi_file_times_ff(f_time)

    #Finding the ACHA and CMIP files
    acha_file, cmip_file = mamf.abi_file_hunter(abi_time_pre0)

    #If the CMIP and ACHA data are available then grab it! If not we'll skip it for the nulls since that dataset will be bigger
    if (acha_file != 'MISSING') and (cmip_file != 'MISSING'):
        #Loading the acha and cmip file
        abi_lats, abi_lons, acha_vals, cmip_vals = mamf.abi_file_loader_v2(acha_file,cmip_file)

        #Getting the variables for the ABI data sampler
        cur_fi_fl = row['fstart']
        cur_fl_lat = row['lat']
        cur_fl_lon = row['lon']

        #Sampling the data using a 20 km BallTree to get what we want out of the file
        cmip_min, cmip_05, acha_max, acha_95 = mamf.abi_data_sampler(abi_lats, abi_lons, acha_vals, cmip_vals, cur_fl_lat, cur_fl_lon)

        #Placing the sampled values in the dataframe
        abi.loc[row['fistart_flid'], 'CMIP_min'] = cmip_min
        abi.loc[row['fistart_flid'], 'CMIP_05'] = cmip_05
        abi.loc[row['fistart_flid'], 'ACHA_max'] = acha_max
        abi.loc[row['fistart_flid'], 'ACHA_95'] = acha_95
        
        #If the ma_category is a hit/late, then use it to randomly sample the ABI scene
        if row['ma_category'] != 'Miss':
            # Getting the lat/lons of all the hits from the manual analysis within 500 km and the prev/next 20 minutes of
            # the first falsh point
            ff_lats, ff_lons = mamf.glm_ff_pts(ff.loc[ff['ma_category']=='Hit'], f_time, cur_fl_lat, cur_fl_lon)
        
            #Getting all the ABI points >20 km, <1000 km, and over the CONUS from the first flash points
            random_df = mamf.abi_subset(abi_lats, abi_lons, acha_vals, cmip_vals, ff_lats, ff_lons, cur_fl_lat, cur_fl_lon)
            
            if random_df.shape[0]>0:
                #Putting in the original flash id for reference later
                fi_fl = [row['fistart_flid'] for i in range(random_df.shape[0])]
                random_df['fistart_flid'] = fi_fl
        
        #Making a dummy array to send back if it's not a hit
        else:
            random_df = pd.DataFrame()
            ff_lats = [-999]
            ff_lons = [-999]

    #If theres no data, then pass in the dummy array for the random sampling too
    else:
        random_df = pd.DataFrame()
        ff_lats = [-999]
        ff_lons = [-999] 
        
    return abi, random_df, ff_lats, ff_lons


# In[7]:


# Function that collects the MRMS data
def mrms_driver(mrms, ff, row, random_df):
    global mrms_variables
    
    #Getting the variables for the ABI data sampler
    cur_fi_fl = row['fstart']
    cur_fl_lat = row['lat']
    cur_fl_lon = row['lon']
    
    #Getting the flash datetime (approx) from the file start time
    f_time = datetime.strptime(cur_fi_fl[0:15], 's%Y%j%H%M%S0')
    
    #Doing some setup with the GLM file time to feed into the MRMS data loader
    fstring_start, m_time = mamf.mrms_setup(f_time)
    
    #Looping through each MRMS variable
    for var in mrms_variables:
        #Getting the mrms data for the specified variable
        mrms_lats, mrms_lons, mrms_data = mamf.MRMS_data_loader(m_time, fstring_start, var)
        
        #A check that we actually have mrms data, or else skip this variable
        if mrms_lats[0]==-999:
            mrms_data_max = np.nan
            mrms_data_95 = np.nan
        
        else:
            #If theres data, get the points within 20 km and sample them!
            mrms_data_max, mrms_data_95 = mamf.mrms_max_finder(cur_fl_lat, cur_fl_lon, mrms_lats, mrms_lons, mrms_data)
            #Putting the data into the mrms dataframe
            mrms.loc[row['fistart_flid'],var+'_max'] = mrms_data_max
            mrms.loc[row['fistart_flid'],var+'_95'] = mrms_data_95
    
        # If the ma_category is a hit/late, we have random points from the ABI data, and we have ABI data
        # then use it to randomly sample the ABI scene
        if (row['ma_category'] != 'Miss') & (random_df.shape[0]>0) & (mrms_lats[0]!=-999):
            #Sampling the MRMS points >20 km, <1000 km, and over the CONUS from the first flash points
            random_df = mamf.mrms_data_subset(mrms_lats, mrms_lons, mrms_data, random_df, var)
        
    return mrms, random_df


# In[21]:


full_rdf = pd.DataFrame(columns=abi_variables_output_random_sample+mrms_variables_output)
full_madf = pd.DataFrame()

counter = 1
#Looping through each row
for index, row in abi_df.iterrows():
    print (index + ': ' + str(counter)+'/'+str(abi_df.shape[0]))
    counter+=1
    
    #Getting the ABI samples from the manually analyzed points, and the randomly sampled ABI points
    abi_df, random_df, ff_lats, ff_lons = abi_driver(abi_df, df, row)
    
    #Getting the MRMS samples from the manually analyzed points, and the randomly sampled MRMS points
    #NOTE: These are based on the abi points for consistency
    mrms_df, random_df = mrms_driver(mrms_df, df, row, random_df)
    
    #Combining new randomly sampled points with the full dataframe
    full_rdf = pd.concat((full_rdf,random_df),axis=0) 
    
    #Combining and the abi and mrms dataframes
    abi_df = abi_df.iloc[:,0:4] #Cutting down the dataframe for redundant information
    combo_df = pd.concat((abi_df,mrms_df),axis=1)

    #Saving out the dataframes
    save_loc = '/localdata/first-flash/data/ml-manual-analysis/'
    combo_df.to_csv(save_loc+'ma-abi-mrms-v00.csv')
    full_rdf.to_csv(save_loc+'ma-random-samples-abi-mrms-v00.csv')