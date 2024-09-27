#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#===================================================
# This script combines all of the files of a specific GLM and version from ABI-ff-combo-v1.py
#
# Created: September 2024
# Author: Kevin Thiel (kevin.thiel@ou.edu)
#
# Run like this: python ABI-compiler-v1.py 16 1
#===================================================


# In[2]:


import sys
import pandas as pd
import numpy as np
import os
from glob import glob


# In[ ]:


#User inputs
args = sys.argv
GLM_number = args[1]
version = args[2]

#Constants
abi_variables_output = ['CMIP_min', 'CMIP_min_pre10', 'CMIP_05', 'CMIP_05_pre10', 'ACHA_max', 'ACHA_max_pre10', 'ACHA_95', 'ACHA_95_pre10', 'abi_file_stime_pre0','abi_file_stime_pre10']
loc = '/localdata/first-flash/data/ABI-processed-GLM'+str(GLM_number)+'-v'+str(version)+'/'


# In[5]:


#Getting a list of the file dates to pull from
file_list = sorted(glob(loc+'2*'))

#Creating an empty DataFrame that all of the data will go into
df = pd.DataFrame(columns=abi_variables_output)


# In[10]:


#Looping through the daily files
for file in file_list:
    print (file)
    #Getting the CSV names to imoport from the daily file
    csv_names = sorted(glob(file+'/*.csv'))
    
    #To avoid double counting files (multiple iterations in one daily file)
    if len(csv_names)>12:
        print('ERROR: TOO MANY CSV FILES IN A FOLDER')
        exit()
    
    #Looping through the individual csv files to load them in
    for csv_file in csv_names:
        #Reading in the CSV file
        new_df = pd.read_csv(csv_file, index_col=0)
        #Concatenating the new dataFrame and the compiled dataFrame
        df = pd.concat((df,new_df),axis=0)

#Adding the GLM number into the file for when we combine the observations later
df['GLM_number'] = np.ones(df.shape[0]) * int(GLM_number)

df.to_csv(loc+'ABI-compiled-GLM'+str(GLM_number)+'-v'+str(version)+'.csv')
