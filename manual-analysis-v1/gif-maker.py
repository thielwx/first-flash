#!/usr/bin/env python
# coding: utf-8



from glob import glob
import imageio.v2 as img
import os
import numpy as np
import yaml

#Constants
loop_nums = 0
frame_duration = 500 #Milliseconds per frame


#Loading in the settings file
with open('case-settings-manual-analysis.yaml', 'r') as f:
    sfile = yaml.safe_load(f)

cases = sfile['cases']


fig_loc = '/localdata/first-flash/figures/manual-analysis-v1/'
output_loc = '/localdata/first-flash/figures/manual-analysis-v1-gifs/'

#Looping through each case...
for case in cases[:]:
    #Getting the list of files available
    files = sorted(glob(fig_loc+case+'/*/'))
    event_names = sorted(os.listdir(fig_loc+case))

    #Looping through each event...
    for i in range(len(files))[:]:
        f = files[i]
        ename = event_names[i]
        output_str = output_loc + case + '/' + ename + '.gif'
        pics = []
        pfiles = glob(f+'*.png')
        
        #Looping through each time step in the event...
        for j in sorted(pfiles):
            pics.append(img.imread(j))
        
        #Reading out the saved images as a gif in the specified location
        img.mimsave(output_str, pics, format='gif', loop=loop_nums, duration=frame_duration)
