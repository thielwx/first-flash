#!/usr/bin/env python
# coding: utf-8

# In[1]:


#===============================================================
# Taking the output files from GLM16 and GLM17 and combining them into one dataset
# in the netCDF format. Includes the 103 W line for East/West data, and filtering for over CONUS.
# May be updated in the future to include additional fitlers as needed.
#
# Created: October 2024
# Author: Kevin Thiel (kevin.thiel@ou.edu)
#===============================================================


# In[17]:


import netCDF4 as nc
import numpy as np
import pandas as pd
import sys
from datetime import datetime
import regionmask


# In[3]:


def east_west_line(dset):
    f_lon = dset.variables['flash_lon'][:]
    glm_number = dset.glm_number
    
    #Using the GLM number to decide whether to select values east or west of 103W
    if glm_number == '16':
        locs = np.where(f_lon >=-103)[0]
    elif glm_number == '17':
        locs = np.where(f_lon <-103)[0]
    else:
        print('ERROR: GLM Number Unknown')
        print('NO EAST/WEST BOUNDS APPLIED')
        locs = np.arange(0,len(f_lon),1)
    
    return locs


# In[ ]:


def us_land_bounds(dset):
    
    f_lon = dset.variables['flash_lon'][:]
    f_lat = dset.variables['flash_lat'][:]
    
    mask_output = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50.mask(lons,lats).values[0]
    locs = np.where(mask_output>0)[0]
    
    return locs


# In[13]:


#Function to take in a GLM ff dataset and output the indicides needed for the flashes, groups, and events
def idx_finder(dset):
    
    #Getting the indicies of the flashes east/west of 103W for the GLM16/17 data
    ew_locs = east_west_line(dset)
    
    #Placeholder until I get regionmask figured out...
    #tot_locs = np.arange(0,dset.dimensions['number_of_flashes'].size,1)
    
    #My attempt at regionmask
    us_land_locs = us_land_bounds(dset)
    
    #Finding the intersetction of all occurances
    flash_locs = list(set.intersection(set(ew_locs),set(us_land_locs)))
    
    flash_fistart_flid = dset.variables['flash_fistart_flid'][flash_locs]
    event_fi_fl = dset.variables['event_fistart_flid'][:]
    group_fi_fl = dset.variables['group_fistart_flid'][:]
    
    event_locs = np.empty(0)
    group_locs = np.empty(0)

    for fi_fl in flash_fistart_flid[:100]:
        e_idx = np.where(event_fi_fl == fi_fl)[0]
        event_locs = np.append(event_locs, e_idx)
        g_idx = np.where(group_fi_fl == fi_fl)[0]
        group_locs = np.append(group_locs, g_idx)
        
    return flash_locs, group_locs.astype('int32'), event_locs.astype('int32')


# In[5]:


#Dictionary of meta data that you'll use to set up your data set and fill it
def get_meta_data():
    meta_dict = {
    'flash_id':['1','file-unique lightning flash identifier',('number_of_flashes',),'int32',False],
    'flash_time_offset_of_first_event':['seconds since flash_parent_file time','time of occurrence of first constituent event in flash',('number_of_flashes',),'float',True],
    'flash_time_offset_of_last_event':['seconds since flash_parent_file time','time of occurrence of last constituent event in flash',('number_of_flashes',),'float',True],
    'flash_lat':['degrees_north','flash centroid (mean constituent event latitude weighted by their energies) latitude coordinate',('number_of_flashes',),'float',False],
    'flash_lon':['degrees_east','flash centroid (mean constituent event longitude weighted by their energies) longitude coordinate',('number_of_flashes',),'float',False],
    'flash_area':['m2','flash area coverage (pixels containing at least one constituent event only)',('number_of_flashes',),'float',True],
    'flash_energy':['J','flash radiant energy',('number_of_flashes',),'float',True],
    'flash_quality_flag':['1','flash data quality flags [0 1 3 5]',('number_of_flashes',),'int',True],
    'group_id':['1','file-unique lightning group identifier',('number_of_groups',),'int32',False],
    'group_time_offset':['seconds since group_parent_file time','mean time of group\'s constituent events\' times of occurrence',('number_of_groups',),'float',True],
    'group_lat':['degrees_north','group centroid (mean constituent event latitude weighted by their energies) latitude coordinate',('number_of_groups',),'float',False],
    'group_lon':['degrees_east','group centroid (mean constituent event longitude weighted by their energies) longitude coordinate',('number_of_groups',),'float',False],
    'group_area':['m2','group area coverage (pixels containing at least one constituent event only)',('number_of_groups',),'float',True],
    'group_energy':['J','group radiant energy',('number_of_groups',),'float',True],
    'group_parent_flash_id':['1','product-unique lightning flash identifier for one or more groups',('number_of_groups',),'int32',False],
    'group_quality_flag':['1','group data quality flags [0 1 3 5]',('number_of_groups',),'int',True],
    'event_id':['1','file-unique lightning event identifier',('number_of_events',),'int32',False],
    'event_time_offset':['seconds since event_parent_file time',' event\'s time of occurrence',('number_of_events',),'float',True],
    'event_lat':['degrees_north','event latitude coordinate',('number_of_events',),'float',True],
    'event_lon':['degrees_east','event longitude coordinate',('number_of_events',),'float',True],
    'event_energy':['J','event radiant energy',('number_of_events',),'float',True],
    'event_parent_group_id':['1','product-unique lightning group identifier for one or more events',('number_of_events',),'int32',False],
    'flashes_after_5_minutes':['1','number of flashes within 30 km and 5 minutes after the first flash event',('number_of_flashes',),'int',False],
    'flashes_after_10_minutes':['1','number of flashes within 30 km and 10 minutes after the first flash event',('number_of_flashes',),'int',False],
    'flashes_after_15_minutes':['1','number of flashes within 30 km and 15 minutes after the first flash event',('number_of_flashes',),'int',False],
    'flashes_after_20_minutes':['1','number of flashes within 30 km and 20 minutes after the first flash event',('number_of_flashes',),'int',False],
    'flashes_after_25_minutes':['1','number of flashes within 30 km and 25 minutes after the first flash event',('number_of_flashes',),'int',False],
    'flashes_after_30_minutes':['1','number of flashes within 30 km and 30 minutes after the first flash event',('number_of_flashes',),'int',False],
    'flash_fistart_flid':['1', 'string of the original file start time (fistart) and flash id (flid) to easily connect to the groups/events',('number_of_flashes',),'str',False],
    'group_fistart_flid':['1', 'string of the original file start time (fistart) and parent flash id (flid) to easily connect to the groups/events',('number_of_groups',),'str',False],
    'event_fistart_flid':['1', 'string of the original file start time (fistart) and parent flash id (flid) to easily connect to the groups/events',('number_of_events',),'str',False]
    }

    return meta_dict


# In[6]:


def output_netcdf_setup(out, g16_dset, cur_time, num_flash, num_group, num_event):
    '''
    This function removes all of the points that are not on land from the compiled raw GLM dataframe
    PARAMS:
        out: The output netCDF file in write mode (netCDF)
        meta_dict: A dictionary containing the meta-data that formats the netCDF file (dictionary)
        compress_dict: The dictionary that contains the information for compressing the data (not used but may be useful someday) (dictionary)
    RETURNS:
        out: The output netCDF file in write mode with the correct format (netCDF)
    '''

    #Creating dimensions
    flash_num = out.createDimension('number_of_flashes', num_flash)
    group_num = out.createDimension('number_of_groups', num_group)
    event_num = out.createDimension('number_of_events', num_event)

    #File meta data
    out.author='Kevin Thiel (kevin.thiel@ou.edu)'
    out.time_coverage_start=g16_dset.time_coverage_start
    out.time_coverage_end=g16_dset.time_coverage_end
    out.date_created=str(cur_time)
    out.glm_number = 'Both'
    out.title='Combined and Collected GLM First Flashes'
    out.first_flash_search_time = g16_dset.first_flash_search_time
    out.first_flash_search_radius = g16_dset.first_flash_search_radius
    out.first_flash_search_radius_flash_area = g16_dset.first_flash_search_radius_flash_area
    out.lat_max = str(50)
    out.lat_min = str(24)
    out.lon_max = str(-66)
    out.lon_min = str(-125)
    out.summary = 'This is collected GLM data for the GLM first-flash project. First flash events were identifed using spatial and temporal thresholds. Flash, group, and event data were pulled from the GLM LCFA L2 files, then combined from GOES-East and -West into a single file using the 103W longitude between E/W and selecting only flash on land in the CONUS. Enjoy!'
    
    return out


# In[7]:


#Funciton that creates each variable in the meta_dictionary, extracts the data from each file, and fills it with the concatenated arrays
def file_filler(out, g16_dset, g17_dset, meta_dict, g16_floc, g16_gloc, g16_eloc, g17_floc, g17_gloc, g17_eloc):
    
    #Going through all by the fistart_flids
    for key in list(meta_dict.keys())[:-3]:
        #Ensuring we set up each variable correctly
        if meta_dict[key][3] == 'int':
            var =  out.createVariable(key, np.int16, meta_dict[key][2])
        elif meta_dict[key][3] == 'int32':
            var =  out.createVariable(key, np.int32, meta_dict[key][2])
        elif meta_dict[key][3] == 'float':
            var =  out.createVariable(key, np.float32, meta_dict[key][2])
        elif meta_dict[key][3] == 'float64':
            var =  out.createVariable(key, np.float64, meta_dict[key][2])
        else:
            var =  out.createVariable(key, np.str_, meta_dict[key][2])
        var.units = meta_dict[key][0]
        var.long_name = meta_dict[key][1]
        
        #Parsing the data based on its indicies 
        if key[0] == 'f':
            g16_var = g16_dset.variables[key][g16_floc]
            g17_var = g17_dset.variables[key][g17_floc]
        elif key[0] == 'g':
            g16_var = g16_dset.variables[key][g16_gloc]
            g17_var = g17_dset.variables[key][g17_gloc]
        elif key[0] == 'e':
            g16_var = g16_dset.variables[key][g16_eloc]
            g17_var = g17_dset.variables[key][g17_eloc]
            
        #Combining the arrays and putting them into the variable
        var[:] = np.concatenate((g16_var,g17_var))
            
    #Going through the fistart_flid because I need them to 
    for key in list(meta_dict.keys())[-3:]:
        var =  out.createVariable(key, np.str_, meta_dict[key][2])
        var.units = meta_dict[key][0]
        var.long_name = meta_dict[key][1]
        
        #Loading the data
        g16_var = g16_dset.variables[key][:]
        g17_var = g17_dset.variables[key][:]
        if key[0] == 'f':
            #parsing by indicies
            g16_var = g16_var[g16_floc]
            #Adding original glm location to fistart_flid
            g16_var = np.array([val+'_16' for val in g16_var])
            g17_var = g17_var[g17_floc]
            g17_var = np.array([val+'_17' for val in g17_var])
        elif key[0] == 'g':
            g16_var = g16_var[g16_gloc]
            g16_var = np.array([val+'_16' for val in g16_var])
            g17_var = g17_var[g17_gloc]
            g17_var = np.array([val+'_17' for val in g17_var])
        elif key[0] == 'e':
            g16_var = g16_var[g16_eloc]
            g16_var = np.array([val+'_16' for val in g16_var])
            g17_var = g17_var[g17_eloc]
            g17_var = np.array([val+'_17' for val in g17_var])
        
        #Combining the arrays and putting them into the variable    
        var[:] = np.concatenate((g16_var,g17_var))
    return out
    


# In[9]:


# User defined inputs
args = sys.argv
#args = ['test', '../../local-data/2022/GLM16_first-flash-data-land_v32_s202201010000_e202301010000_c202409052049.nc', '../../local-data/2022/GLM17_first-flash-data-land_v32_s202201010000_e202301010000_c202409052049.nc']
glm16_file_str = args[1]
glm17_file_str = args[2]

#Loading the original datasets
g16_dset = nc.Dataset(glm16_file_str)
g17_dset = nc.Dataset(glm17_file_str)

#Safety checks with the file meta data
if ((g16_dset.time_coverage_start != g17_dset.time_coverage_start) 
    | (g16_dset.time_coverage_end != g17_dset.time_coverage_end)):
    print ('WARNING: TIME BOUNDS DO NOT MATCH')
if ((g16_dset.first_flash_search_time != g17_dset.first_flash_search_time) 
    | (g16_dset.first_flash_search_radius != g17_dset.first_flash_search_radius)
    | (g16_dset.first_flash_search_radius_flash_area != g17_dset.first_flash_search_radius_flash_area)):
    print ('WARNING: FIRST FLASH DEFINITIONS DO NOT MATCH')
if (g16_dset.land_flashes_only != g17_dset.land_flashes_only):
    print ('WARNING: LAND CONSTRATINTS DO NOT MATCH')

#Getting the strings to put into the output file
s_time_str = datetime.strptime(g16_dset.time_coverage_start, '%Y-%m-%d %H:%M:%S').strftime('s%Y%m%d%H%M')
e_time_str = datetime.strptime(g16_dset.time_coverage_end, '%Y-%m-%d %H:%M:%S').strftime('e%Y%m%d%H%M')
cur_time = datetime.now()
c_time_str = cur_time.strftime('c%Y%m%d%H%M')

#Constants
file_loc = '/localdata/first-flash/data/GLM-ff-east-west-combined/'
#file_loc = '' #DEVMODE
file_str = 'GLM-East-West_first-flash-data_v'+glm16_file_str[-47:-45]+'_'+s_time_str+'_'+e_time_str+'_'+c_time_str+'.nc'


# In[14]:


#Getting the flash, group, and event indicies in each dataset based on our filters
g16_floc, g16_gloc, g16_eloc = idx_finder(g16_dset)
g17_floc, g17_gloc, g17_eloc = idx_finder(g17_dset)

num_flash = len(g16_floc) + len(g17_floc)
num_group = len(g16_gloc) + len(g17_gloc)
num_event = len(g16_eloc) + len(g17_eloc)


# In[15]:


#Getting the dictionary of meta-data to loop through
meta_dict = get_meta_data()


# In[16]:


#Creating and setting up the netCDF file in write mode
out = nc.Dataset(file_loc+file_str, mode='w', format='NETCDF4')
out = output_netcdf_setup(out, g16_dset, cur_time, num_flash, num_group, num_event)
out = file_filler(out, g16_dset, g17_dset, meta_dict, g16_floc, g16_gloc, g16_eloc, g17_floc, g17_gloc, g17_eloc)


# In[ ]:


out.close()

