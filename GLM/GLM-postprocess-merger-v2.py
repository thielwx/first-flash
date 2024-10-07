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
import geopandas as gpd




# A function that removes the GLM16/17 data west/east of 103W
def ew_combo(df_16,df_17):
    df_16_cut = df_16.loc[df_16['lons']>=-103.0]
    df_17_cut = df_17.loc[df_17['lons']<-103.0]
    return df_16_cut, df_17_cut


def conus_mask(df):
    #Converting the DataFrame to a GeoDataFrame and assigning the coordinate reference system
    gdata = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lons, df.lats))
    gdata.set_crs(epsg=4326, inplace=True)

    #Loading in the shapefile, setting the matching coordinate reference system, and creating a CONUS mask
    conus = gpd.read_file('tl_2024_us_state/')
    conus = conus.to_crs('EPSG:4326')
    notCONUS = ['Alaska', 'Hawaii', 'Puerto Rico', 'Commonwealth of the Northern Mariana Islands', 'Guam', 'United States Virgin Islands', 'American Samoa']
    mask = conus['NAME'].isin(notCONUS)
    conus = conus[~mask]

    #Clipping the first flash data to the CONUS mask
    df_clipped = gpd.clip(gdata, conus)

    return df_clipped



#Takes in the two datasets and filters out 
def combined_idx_finder(g16_dset,g17_dset):

    #Adding the GLM number to the fistart_flid's
    fi_fl16 = g16_dset.variables['flash_fistart_flid'][:]
    fi_fl16 = [id_str+'_16' for id_str in fi_fl16]
    fi_fl17 = g17_dset.variables['flash_fistart_flid'][:]
    fi_fl17 = [id_str+'_17' for id_str in fi_fl17]
    
    #Putting the data into the dataframes
    d16 = {
        'lats':g16_dset.variables['flash_lat'][:],
        'lons':g16_dset.variables['flash_lon'][:],
        'idx':fi_fl16
    }

    d17 = {
        'lats':g17_dset.variables['flash_lat'][:],
        'lons':g17_dset.variables['flash_lon'][:],
        'idx':fi_fl17
    }

    df_16 = pd.DataFrame(data=d16)
    df_16.set_index('idx', inplace=True)
    df_17 = pd.DataFrame(data=d17)
    df_17.set_index('idx', inplace=True)

    #Cutting down the data by longitude and combining into one dataframe
    df_16_cut, df_17_cut = ew_combo(df_16,df_17)
    df_combo = pd.concat((df_16_cut,df_17_cut),axis=0)

    #Getting the data over CONUS only
    df_clipped = conus_mask(df_combo)

    return list(df_clipped.index)


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

    f_keys = ['flash_id','flash_time_offset_of_first_event','flash_time_offset_of_last_event','flash_lat','flash_lon','flash_area','flash_energy','flash_quality_flag','flashes_after_5_minutes','flashes_after_10_minutes','flashes_after_15_minutes','flashes_after_20_minutes','flashes_after_25_minutes','flashes_after_30_minutes','flash_fistart_flid']
    g_keys = ['group_id','group_time_offset','group_lat','group_lon','group_area','group_energy','group_parent_flash_id','group_quality_flag','group_fistart_flid']
    e_keys = ['event_id','event_time_offset','event_lat','event_lon','event_energy','event_parent_group_id','event_fistart_flid']    

    return meta_dict, f_keys, g_keys, e_keys


# In[6]:


def df_creator(g16_dset,g17_dset,keys, fistart_flid_list):
    #Creating an empty dictionary to fill
    dict = {}

    #Filling the dictionary with all the variables from the first GLM16/17 files for flash, groups, or events
    for key in keys[:-1]:
        var1 = g16_dset.variables[key][:]
        var2 = g17_dset.variables[key][:]
        var = np.concat((var1,var2))
        dict[key] = var

    #Extra step to add the _16 or _17 to fistart_flid in the dictionary
    fi_fl16 = g16_dset.variables[keys[-1]][:]
    fi_fl16 = [id_str+'_16' for id_str in fi_fl16]
    fi_fl17 = g17_dset.variables[keys[-1]][:]
    fi_fl17 = [id_str+'_17' for id_str in fi_fl17]

    fi_fl_combo = np.concat((fi_fl16,fi_fl17))
    dict[keys[-1]] = fi_fl_combo

    #Creating the DataFrame
    df = pd.DataFrame(data=dict)
    #Setting the index to the flash/group/event_fistart_flid
    df.set_index(keys[-1],inplace=True)
    #Slicing the DataFrame by the list of fistart_flids that you need from the filter
    df = df.loc[fistart_flid_list,:]

    return df





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


#Funciton that creates each variable in the meta_dictionary, extracts the data from each file, and fills it with the concatenated arrays
def file_filler_v2(out, g16_dset, g17_dset, meta_dict, df, keys):
    
    #Going through all by the fistart_flids
    for key in keys:
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
        
        #Putting the data into the variable
        var[:] = df[key].values
            
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

# CUT OUT IN V2
# #Getting the flash, group, and event indicies in each dataset based on our filters
# g16_floc, g16_gloc, g16_eloc = idx_finder(g16_dset)
# g17_floc, g17_gloc, g17_eloc = idx_finder(g17_dset)

# num_flash = len(g16_floc) + len(g17_floc)
# num_group = len(g16_gloc) + len(g17_gloc)
# num_event = len(g16_eloc) + len(g17_eloc)


# In[15]:
#Getting the list of necessary files to match with
fistart_flid_list = combined_idx_finder(g16_dset,g17_dset)
print ('---------------')
print ('Flashes filtered and IDs found, length='+str(fistart_flid_list))
print (datetime.now()-cur_time())
print ('---------------')

#Getting the dictionary of meta-data to loop through
meta_dict, f_keys, g_keys, e_keys = get_meta_data()

#Creating the dataframes we need
f_df = df_creator(g16_dset,g17_dset,f_keys, fistart_flid_list)
g_df = df_creator(g16_dset,g17_dset,g_keys, fistart_flid_list)
e_df = df_creator(g16_dset,g17_dset,e_keys, fistart_flid_list)
print ('---------------')
print ('Flash, group, event DataFrames created')
print (datetime.now()-cur_time())
print ('---------------')

#Creating and setting up the netCDF file in write mode
out = nc.Dataset(file_loc+file_str, mode='w', format='NETCDF4')
out = output_netcdf_setup(out, g16_dset, cur_time, f_df.shape[0], g_df.shape[0], e_df.shape[0])

#out = file_filler(out, g16_dset, g17_dset, meta_dict, g16_floc, g16_gloc, g16_eloc, g17_floc, g17_gloc, g17_eloc)

#Filling the netCDF file with the appropriate datasets
out = file_filler_v2(out, g16_dset, g17_dset, meta_dict, f_df, f_keys)
print ('---------------')
print ('Flashes written into netCDF file')
print (datetime.now()-cur_time())
print ('---------------')
out = file_filler_v2(out, g16_dset, g17_dset, meta_dict, g_df, g_keys)
print ('---------------')
print ('Groups written into netCDF file')
print (datetime.now()-cur_time())
print ('---------------')
out = file_filler_v2(out, g16_dset, g17_dset, meta_dict, e_df, e_keys)
print ('---------------')
print ('Events written into netCDF file')
print (datetime.now()-cur_time())
print ('---------------')

#Closing the file
out.close()

