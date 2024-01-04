#!/usr/bin/env python
# coding: utf-8
# ==========================================================================================
# This script pulls GOES data from the Google Cloud storage on an hourly basis
# based on user input related to:
#   - GOES-16/17/18 satellites
#   - GLM/ABI sensors
#   - CONUS/Full/Meso secene
#   - A handfull of ABI L2 products to boot!
# Data is stored locally on a daily basis (YYYYMMMDD)
#
# Created by: Kevin Thiel
# Date: October 2023
# Notes: Requires Google Cloud clinet to be installed: https://cloud.google.com/sdk/docs/install
# =========================================================================================

# In[1]:


import subprocess as sp
import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
from datetime import datetime


#===========================================================================================
# Funciton Land
#===========================================================================================

# In[2]:

def input_checker(text, options):
    '''
    Used to get the desired user input from the text, and check that it's one of the avilable options
    '''
    while True:
        var = input(text).upper()
        if var in options:
            break
        print ('Incorrect entry. Please try again.')
        
    return var


# In[3]:


def input_time(text):
    '''
    Used to input time from the user, and check that it's able to be converted into DateTime
    '''
    while True:
        var = input(text).upper()
        if len(var) == 12:
            break
        print ('Incorrect entry. Please try again.')
        
    return var


# In[4]:


def input_save(text, sensor, satellite, product, scene, meso_num, channel_num):
    '''
    Used to get the input save location from the user, and then create a file path for saving the data locally
    '''
    #Default to make the code more simple
    #var_out = '/localdata/first-flash/data/'+ sensor + satellite 
    
    #If you ever want to extend this and make it more generalizable...update I did!
    while True:
        var_out = input(text)
        if len(var_out) > 0:
            break
        print ('Incorrect entry. Please try again.')
    
    save_string = 'None'
    
    
    #Going through the dirty process of creating the save locations based on the combination of user inputs
    
    #ABI CMIP Products
    if sensor=='ABI' and product=='CMIP' and meso_num=='None':
        save_string = var_out + '-' + product + scene + str(channel_num).zfill(2)+ '/'
    elif sensor=='ABI' and product=='CMIP' and meso_num!='None':
        save_string = var_out + '-' + product + scene + str(channel_num).zfill(2)+ '-' + meso_num +'/'
    
    #ABI L2 products
    elif sensor=='ABI' and product !='CMIP' and meso_num=='None':
        save_string = var_out + '-' + product + scene + '/'
    elif sensor=='ABI' and product !='CMIP' and meso_num!='None':
        save_string = var_out + '-' + product + scene + '-' + meso_num +'/'
    
    #GLM
    elif sensor=='GLM':
        save_string = var_out + sensor + satellite + '-LCFA/'
    
    #Making sure we actually got a string we can use
    if save_string == 'None':
        print('ERROR IN THE SAVE STRING')
        exit()
    
    return save_string


# In[5]:


def command_maker_v2(t, sensor, satellite, product, scene, meso_num, channel_num, save_string_full):
    '''
    Used to create the string that contains the command that will be used to download data from Google Cloud
    '''
    year = t.strftime('%Y')
    doy = t.strftime("%j")
    hr = t.strftime("%H")
    
    #Padding with zeros correctly
    channel_num = str(channel_num).zfill(2)
    doy = str(doy).zfill(3)
    hr = str(hr).zfill(2)
    
    start = 'gsutil -m cp gs://gcp-public-data-goes-'+satellite+'/'+sensor+'-L2-'
    
    
    #ABI CMIP Products
    if sensor=='ABI' and product=='CMIP' and meso_num=='None': #CONUS and Full Disk
        middle = product+scene + '/' + year + '/' + doy + '/' + hr + '/'
        end = 'OR_'+sensor+'-L2-'+product+scene+'-M?C'+channel_num+'_G'+satellite+'_s'+year+doy+hr+'* '
    elif sensor=='ABI' and product=='CMIP' and meso_num!='None': #Meso (1/2)
        middle = product+scene + '/' + year + '/' + doy + '/' + hr + '/'
        end = 'OR_'+sensor+'-L2-'+product+scene+meso_num+'-M?C'+channel_num+'_G'+satellite+'_s'+year+doy+hr+'* '
    
    #ABI L2 products
    elif sensor=='ABI' and product !='CMIP' and meso_num=='None': #CONUS and Full Disk
        middle = product+scene + '/' + year + '/' + doy + '/' + hr + '/'
        end = 'OR_'+sensor+'-L2-'+product+scene+'-M?_G'+satellite+'_s'+year+doy+hr+'* '
    elif sensor=='ABI' and product !='CMIP' and meso_num!='None': #Meso (1/2)
        middle = product+scene + '/' + year + '/' + doy + '/' + hr + '/'
        end = 'OR_'+sensor+'-L2-'+product+scene+meso_num+'-M?_G'+satellite+'_s'+year+doy+hr+'* '
        
    
    #GLM
    elif sensor=='GLM':
        middle = 'LCFA/' + year + '/' + doy + '/' + hr + '/'
        end = 'OR_GLM-L2-LCFA_G'+satellite+'_s'+year+doy+hr+'* '
        
    command = start + middle + end + save_string_full
    print (command)
    return command


# In[6]:


#===========================================================================================
# The section that drives the functions to download Google Cloud GOES data
#===========================================================================================

# Dummy variables that we may fill later
product = 'None'
channel_num = 'None'
meso_num = 'None'
scene = 'None'


#Start of user input section
print ('Welcome to the GOES Data Puller!')

satellite = input_checker('Satellite (16,17,18):', ['16','17','18'])
sensor = input_checker('Sensor (GLM or ABI):', ['GLM','ABI'])

#Getting the necessary information from the user based on their inputs
if sensor == 'ABI':
    abi_prods = ['CMIP','ACHA','ACM','ACT'] #This list can be expanded for other ABI L2 products
    product = input_checker('ABI Product (e.g. CMIP, ACHA, ACM, ACT, etc.):', abi_prods)
    
    if product == 'CMIP':
        ch_prods = [str(x) for x in np.arange(1,17,1)]
        channel_num = input_checker('CMIP Band (1-16):', ch_prods)
        
    scene = input_checker('Scene (F,C,M):', ['F','C','M'])
    if scene == 'M':
        meso_num = input_checker('Meso Number (1,2):', ['1','2'])
        
t_start = input_time('Start Time (YYYYMMDDHHmm):')
t_end = input_time('End Time (YYYYMMDDHHmm):')


# In[7]:


#Soliciting the save path and creating an output spring
save_string = input_save('Save Location Path:', sensor, satellite, product, scene, meso_num, channel_num) 


# In[8]:


t_start_dt = datetime.strptime(t_start, '%Y%m%d%H%M%S') #Converting into datetimes
t_end_dt = datetime.strptime(t_end, '%Y%m%d%H%M%S') #Converting into datetimes

#Creating a list of dates and times that we can pull our hourly analysis from
time_list = pd.date_range(start=t_start_dt, end=t_end_dt, freq='H').to_list()


# In[9]:

# Creating a day checker to use in the if statement below
day_checker = (time_list[0] - timedelta(hours=24)).strftime('%d')


#Looping through the data on an hourly basis
for t in time_list:
    yr_current = t.strftime('%Y')
    day_current = t.strftime('%d')
    mo_current = t.strftime('%m')
    
    #Checking the current day in case we need to create a new save path (data is organized into daily files)
    if day_current != day_checker:
        save_string_full = save_string + yr_current + mo_current + day_current +'/'
        print (save_string_full)
        if not os.path.exists(save_string_full):
            os.makedirs(save_string_full)
        
        day_checker = day_current
    
    # Running the built command
    command = command_maker_v2(t, sensor, satellite, product, scene, meso_num, channel_num, save_string_full)
    p = sp.Popen(command, shell=True)
    p.wait()