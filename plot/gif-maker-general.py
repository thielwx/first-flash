#!/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import imageio as img
import os
import numpy as np


# In[2]:


#=========EDIT THIS SECTION==============================
data_loc = '/localdata/first-flash/figures/cases/20220322-perils/full-animation-1/' #This should have '/' on both ends
file_format = '.png'
gif_name = '20220322-perils-all-v2' #DON'T ADD .gif
loop_nums = 0
frame_duration = 100 #Milliseconds per frame
#========================================================


# In[3]:


globber = glob(data_loc+'*'+file_format)


# In[4]:


pics = []
for i in (sorted(globber))[::5]:
    pics.append(img.imread(i))

img.mimsave(data_loc+gif_name+'-'+str(frame_duration)+'.gif', pics , format='gif', loop=loop_nums, duration=frame_duration/1000)


# In[ ]:



