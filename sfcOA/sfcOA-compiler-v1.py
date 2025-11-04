#!/usr/bin/env python
# coding: utf-8


#===================================================
# This script combines all of the raw files from the sfcOA processing
# and puts them into a single file
#
# Created: November 2025
# Author: Kevin Thiel (kevin.thiel@ou.edu)
#
# Run like this: python sfcOA-compiler-v1.py
#===================================================


import pandas as pd
import numpy as np
import os
from glob import glob


#Constants
version = 1

oa_vars_df = pd.read_csv('ff-sfcoa-vars.csv') #A csv of the variables we'll be pulling from in the sfcoaruc files
oa_vars_input = oa_vars_df['FIELD'].values
oa_vars_output = np.array([[var+'_T0', var+'_T1', var+'_T2', var+'_T3'] for var in oa_vars_input], dtype='str').flatten() #Adding the time lag subsets(T0,T1,T2,T3)

loc = '/localdata/first-flash/data/sfcOA-processed-v'+str(version)+'/'


#Getting a list of the file dates to pull from
file_list = sorted(glob(loc+'2*'))

#Creating an empty DataFrame that all of the data will go into
df = pd.DataFrame(columns=oa_vars_output)


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


df.to_csv(loc+'sfcOA-compiled-v'+str(version)+'.csv')
