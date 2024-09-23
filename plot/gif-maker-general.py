# !/usr/bin/env python
# coding: utf-8

# In[1]:


from glob import glob
import imageio as img
import os
import numpy as np


# In[2]:


#=========EDIT THIS SECTION==============================
case = '20220423-oklma'
flash_id = 'f2'
data_loc = '/localdata/first-flash/figures/cases/'+case+'/'+case+'-'+flash_id+'-v1/' #This should have '/' on both ends
file_format = '.png'
skip_frames = 5 #Animate every nth frame
gif_name = case+'-'+flash_id+'-every'+str(skip_frames)+'-v2' #DON'T ADD .gif
loop_nums = 0
frame_duration = 100 #Milliseconds per frame
#========================================================


# In[3]:


globber = glob(data_loc+'*'+file_format)


# In[4]:


pics = []
for i in (sorted(globber))[::skip_frames]:
    pics.append(img.imread(i))

img.mimsave(data_loc+gif_name+'-'+str(frame_duration)+'.gif', pics , format='gif', loop=loop_nums, duration=frame_duration/1000)




