#!/usr/bin/env python
# coding: utf-8

# This script is designed to cycle through first flash candidates and manually select if they are actually first flash events

# In[10]:


import pandas as pd
import sys
import matplotlib.pyplot as pt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import yaml



#Loading in the settings file
with open('case-settings-manual-analysis.yaml', 'r') as f:
    sfile = yaml.safe_load(f)

cases = sfile['cases']

#Variables, input options, and display text

ff_cat_text = 'Hit (h), miss (m), or late (l)? '
ff_cat_options = ['h','m','l']
ff_cat_full = ['Hit','Miss','Late']

confidence_text = 'Confidence from 1 to 5: '
confidence_range = [1,5]

scenario_options = ['c','s','a','o']
scenario_text = 'Convective core (c), stratiform (s), anvil (a), or other (o): '
scenario_full = ['Convective Core', 'Stratiform', 'Anvil', 'Other']

t_diff_text_eni = 'ENI late time difference (-1 to -600s): '
t_diff_range = [-1,600]

# In[12]:


def input_checker(text, options):
    
    while True:
        var = input(text).lower()
        if var in options:
            break
        print ('Incorrect entry. Please try again.')
        
    return var


# In[13]:


def input_checker_range(text, options):
    
    while True:
        var = input(text)
        if (int(var)>=options[0])and(int(var)<=options[1]):
            break
        print ('Incorrect entry. Please try again.')
        
    return var


# In[14]:


print ('Welcome to the manual analysis processor!')

# Getting the case
print ('Which case would you like to use?')
for case in cases:
    print (case)

case = input_checker('Case: ', cases)

ff_loc_folder = '/localdata/first-flash/data/manual-analysis-v1/'+case+'/'
ff_loc = ff_loc_folder + case+'-ffRAW-v00combos-v1-manual-analysis.csv'

# Reading in the first flash data
ff = pd.read_csv(ff_loc, index_col=0)

#Deciding where to start if picking up from a previous spot
start_text = 'Which event would you like to start with? (Index:0-'+str(ff.shape[0]-1)+') '
start_range = [0,ff.shape[0]-1]

start_index = input_checker_range(start_text, start_range)


#Setting up the manual analysis dataframe and filling the categories with the dummy entries
ff_ma = ff.copy()

#If there are any fewer columns, then we know it's a new dataset and needs the extra fields
if ff_ma.shape[1]<25: 
    ff_ma['ma_category'] = ['NULL' for i in range(ff.shape[0])]
    ff_ma['ma_confidence'] = [0 for i in range(ff.shape[0])]
    ff_ma['ma_scenario'] = ['NULL' for i in range(ff.shape[0])]
    ff_ma['ma_time_diff_eni'] = [np.nan for i in range(ff.shape[0])]


# In[17]:


#Looping through each row in the dataframe and collecting user input
for i in range(ff.shape[0])[int(start_index):]:
    
    #Categorizing the flash as a 'Hit' or a 'Miss'
    folder_str = str(i) + '-' + ff['fistart_flid'].values[i]+': '
    ff_cat = input_checker(folder_str+ff_cat_text, ff_cat_options)
    
    if ff_cat == 'h': #hit
        ff_ma['ma_category'].values[i] = ff_cat_full[0]
    elif ff_cat == 'm': #miss
        ff_ma['ma_category'].values[i] = ff_cat_full[1]
    elif ff_cat == 'l': #late
        ff_ma['ma_category'].values[i] = ff_cat_full[2]
        
    #Getting a confidence level on the previous categorization
    ff_con = input_checker_range(confidence_text, confidence_range)
    ff_ma['ma_confidence'].values[i] = int(ff_con)
    
    #Getting the convective scenario
    ff_scen = input_checker(scenario_text,scenario_options)
    
    if ff_scen=='c':
        ff_ma['ma_scenario'].values[i] = scenario_full[0]
    elif ff_scen=='s':
        ff_ma['ma_scenario'].values[i] = scenario_full[1]
    elif ff_scen=='a':
        ff_ma['ma_scenario'].values[i] = scenario_full[2]
    elif ff_scen=='o':
        ff_ma['ma_scenario'].values[i] = scenario_full[3]
        
    #Getting how late was the flash if (it wasn't a miss) for ENI
    if ff_cat == 'h':
        ff_ma['ma_time_diff_eni'].values[i] = 0
    elif ff_cat == 'l':
        ff_late_eni = input_checker_range(t_diff_text_eni, t_diff_range)
        ff_ma['ma_time_diff_eni'].values[i] = ff_late_eni
    
    ff_ma.to_csv(ff_loc)

# In[ ]:





