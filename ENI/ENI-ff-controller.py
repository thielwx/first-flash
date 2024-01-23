#!/usr/bin/env python
# coding: utf-8

# In[13]:


#=====================================
# This is the code that controls GLM-ff-raw-creator-v2.py
# and allows us to run over multiple days
#
# Author: Kevin Thiel
# Date: January 2024
#=====================================


# In[14]:


from datetime import datetime
import pandas as pd
import subprocess as sp
import eni_function_master_v1 as efm
import sys

# In[ ]:


datetime_start = datetime.now()

#==========================
# Run like this python ENI-ff-controller.py 20220501 20220502
#==========================

args = sys.argv

t_start = args[1] #Start time
t_end = args[2] #End time (does not include data from that e, just up to it)

#Getting the list of dates we need
t_start_dt = datetime.strptime(t_start, '%Y%m%d') #Converting into datetimes
t_end_dt = datetime.strptime(t_end, '%Y%m%d') #Converting into datetimes
time_list = pd.date_range(start=t_start_dt, end=t_end_dt, freq='D').to_list()


# In[12]:


#looping through all the dates
for i in range(len(time_list)-1):
    #Printing the current time for the user
    print (time_list[i])
    #Getting the times into a string format
    y, m, d, doy, hr, mi = efm.datetime_converter(time_list[i])
    start_time_str = y+m+d+hr+mi
    y, m, d, doy, hr, mi = efm.datetime_converter(time_list[i+1])
    end_time_str = y+m+d+hr+mi
    
    cmd = 'python /localdata/PyScripts/first-flash/ENI/eni_ff_raw_creator_v1.py '+start_time_str+' '+end_time_str
    print (cmd)
    
    p = sp.Popen(cmd,shell=True)
    p.wait()


# In[ ]:


datetime_end = datetime.now()

print('Total runtime:')
print(datetime_end-datetime_start)

