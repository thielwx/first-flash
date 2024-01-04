#!/usr/bin/env python
# coding: utf-8

# This script pulls GOES data from the Google Cloud storage on an hourly basis
# 
# GOES-16/17/18 satellites
# GLM/ABI sensors
# CONUS/Full/Meso secene
# A handfull of ABI L2 products to boot!

# In[2]:


import subprocess as sp
import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta
from datetime import datetime


# In[3]:


def input_checker(text, options):
    
    while True:
        var = input(text).upper()
        if var in options:
            break
        print ('Incorrect entry. Please try again.')
        
    return var


# In[4]:


def input_time(text):
    
    while True:
        var = input(text).upper()
        if len(var) == 12:
            break
        print ('Incorrect entry. Please try again.')
        
    return var


# In[5]:


def input_save(text, sensor, satellite, product, scene, meso_num, channel_num):
    
    #Default to make the code more simple
    var_out = '/localdata/first-flash/data/'+ sensor + satellite 
    
    #If you ever want to extend this and make it more generalizable...
    #var = input(text)
    #var out = '/localdata/' + var + sensor + satellite + '-' + product
    
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
        save_string = var_out + '-LCFA/'
    
    #Making sure we actually got a string we can use
    if save_string == 'None':
        print('ERROR IN THE SAVE STRING')
        exit()
    
    return save_string


# In[33]:


def command_maker_v2(t, sensor, satellite, product, scene, meso_num, channel_num, save_string_full):
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


# In[26]:


#Trying to create user input for the script so it's more flexible:

# Dummy variables that we may fill later
product = 'None'
channel_num = 'None'
meso_num = 'None'
scene = 'None'


#Start of user input section
print ('Welcome to the GOES Data Puller!')

satellite = input_checker('Satellite (16,17,18):', ['16','17','18'])
sensor = input_checker('Sensor (GLM or ABI):', ['GLM','ABI'])

if sensor == 'ABI':
    abi_prods = ['CMIP','ACHA','ACM','ACT']
    product = input_checker('ABI Product (initials only):', abi_prods)
    
    if product == 'CMIP':
        ch_prods = [str(x) for x in np.arange(1,17,1)]
        channel_num = input_checker('CMIP Band (1-16):', ch_prods)
        
    scene = input_checker('Scene (F,C,M):', ['F','C','M'])
    if scene == 'M':
        meso_num = input_checker('Meso Number (1,2):', ['1','2'])
        
t_start = input_time('Start Time (YYYYMMDDHHmm):')
t_end = input_time('End Time (YYYYMMDDHHmm):')


# In[30]:


#should go 'first-flash/data/' then product and date will follow, for now just defaulting to the current project
save_string = input_save('Save Location (in /localdata/)', sensor, satellite, product, scene, meso_num, channel_num) 


# In[31]:


t_start_dt = datetime.strptime(t_start, '%Y%m%d%H%M%S') #Converting into datetimes
t_end_dt = datetime.strptime(t_end, '%Y%m%d%H%M%S') #Converting into datetimes

#Creating a list of dates and times that we can pull our hourly analysis from
time_list = pd.date_range(start=t_start_dt, end=t_end_dt, freq='H').to_list()


# In[32]:


day_checker = (time_list[0] - timedelta(hours=24)).strftime('%d')


for t in time_list:
    yr_current = t.strftime('%Y')
    day_current = t.strftime('%d')
    mo_current = t.strftime('%m')
    
    if day_current != day_checker:
        save_string_full = save_string + yr_current + mo_current + day_current +'/'
        print (save_string_full)
        if not os.path.exists(save_string_full):
            os.makedirs(save_string_full)
        
        day_checker = day_current
    
    
    command = command_maker_v2(t, sensor, satellite, product, scene, meso_num, channel_num, save_string_full)
    p = sp.Popen(command, shell=True)
    p.wait()


# In[17]:


#(time_list[0] - timedelta(hours=24)).strftime('%d')


# In[18]:


#save_string


# In[ ]:




