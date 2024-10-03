#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#===================================================
# This script combines all of the files of match Earth Networks flashes from a specific version from eni-ff-combo-GLM-v1.py
#
# Created: October 2024
# Author: Kevin Thiel (kevin.thiel@ou.edu)
#
# Run like this: python eni-ff-combo-GLM-compiler-v1.py 16 1
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
eni_variables_output = ['type','timestamp','latitude','longitude','peakcurrent','icheight','numbersensors','icmultiplicity','cgmultiplicity','starttime','endtime','duration','ullatitude','ullongitude','lrlatitude','lrlongitude']
loc = '/localdata/first-flash/data/ENI-processed-GLM'+str(GLM_number)+'-v'+str(version)+'/'


# In[5]:


#Getting a list of the file dates to pull from
file_list = sorted(glob(loc+'2*'))

#Creating an empty DataFrame that all of the data will go into
df = pd.DataFrame(columns=eni_variables_output)


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

df.to_csv(loc+'ENI-compiled-GLM'+str(GLM_number)+'-v'+str(version)+'.csv')
