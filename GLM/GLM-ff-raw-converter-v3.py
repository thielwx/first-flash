#!/usr/bin/env python
# coding: utf-8

#==================================================================================
# Writing out the GLM first flash raw files into a netCDF format containing all of the necessary flash, group, and event data.
# This is updated from GLM-ff-raw-converter-v2_2.py to handle the output from the gridsearh (flash area radius dependent)
# Author: Kevin Thiel (kevin.thiel@ou.edu)
# Created: September 2024
#
# Inputs:
#      start time YYYYmmdd
#      end time YYYYmmdd
#      GLM number
#      'land' or 'all'
#=================================================================================

# In[1]:


import pandas as pd
from glob import glob
import netCDF4 as nc
import sys
from datetime import datetime
import first_flash_function_master as ff
import numpy as np
import global_land_mask as globe


# # Function Land

# In[2]:


def raw_finder(start_time, glm_sat, data_loc, search_r, search_m, search_fr, ver):
    '''
    This function takes in the data-specific info, finds the associated RAW GLM files, and compiles them into a dataframe
    PARAMS:
        time: start date and time (datetime object)
        glm_sat: glm number (e.g. 16) (int)
        data_loc: file location of the raw files (str)
        search_r: radius of search of the raw files (int)
        search_m: time span of serach of the raw files (int)
        search_fr: radius of search on top of the flash area radius (int)
        ver: version of the raw files (int)
    RETURNS:
        ff_df: A dataframe of the compiled raw GLM data from the csv files (DataFrame)
    '''
    #data_loc = '../../test-data/' #Dev Mode
    
    #converting the datetime objects to string pieces
    y, m, d, doy, hr, mi = ff.datetime_converter(start_time)
    #Making the file string
    file_str_start = 'GLM' + str(glm_sat) +'_ffRAW_r'+str(search_r).zfill(2)+'_t'+str(search_m).zfill(2)+'_fr'+str(search_fr).zfill(2)+'_v'+str(ver).zfill(2) + '_'
    file_str_end = 's' + y + m + d + '*.csv'
    #Getting the file string
    file_loc = data_loc + 'GLM' + str(glm_sat) + '_ffRAW_v' + str(ver) + '/' + y + m + d + '/'
    total_file_str = file_loc + file_str_start + file_str_end 
    
    #Searching for the files using glob
    glob_files = sorted(glob(total_file_str))
    
    ff_df = pd.DataFrame()
    
    for files in glob_files:
        new_df = pd.read_csv(files)

        #Included for when the first flash file is empty
        if new_df.shape[0]>0:
            new_df['start_time'] = pd.to_datetime(new_df['start_time'])
            ff_df = pd.concat((ff_df,new_df))
    
    return ff_df


# In[ ]:


def dset_land_points(df):
    '''
    This function removes all of the points that are not on land from the compiled raw GLM dataframe
    PARAMS:
        df: A dataframe of the compiled raw GLM data (DataFrame)
    RETURNS:
        ff_land: A dataframe of the compiled raw GLM data on land (DataFrame)
    '''
    #Getting the lat and lon values from the dataframe
    pre_lat = df['lat'].values
    pre_lon = df['lon'].values
    
    #Subject our lats and lons to some boolean tests
    land_truther = globe.is_land(pre_lat, pre_lon)
    
    #Using the boolean array to give a dataframe of lat/lons that are on land only
    df_land = df.loc[land_truther,:]
    
    return df_land


# In[3]:


def output_file_str_create(start_time, end_time, glm_sat, ver, data_loc):
    '''
    This function takes in the data-specific info, finds the associated RAW GLM files, and compiles them into a dataframe
    PARAMS:
        start_time: start date and time (datetime object)
        end_time: end date and time (datetime object)
        glm_sat: glm number (e.g. 16) (int)
        ver: version of the raw files (int)
        data_loc: file location of the output files (str))
    RETURNS:
        file_save_str: output string of netCDF file (string)
        file_loc_str: output string of directory location (string)
        cur_time: current datetime used in the netCDF file (datetime object)
    '''
    #Formatting the file strings
    y, m, d, doy, hr, mi = ff.datetime_converter(start_time)
    file_start_str = 's'+y+m+d+hr+mi
    y, m, d, doy, hr, mi = ff.datetime_converter(end_time)
    file_end_str = 'e'+y+m+d+hr+mi
    cur_time = datetime.now()
    y, m, d, doy, hr, mi = ff.datetime_converter(cur_time)
    file_cur_str = 'c'+y+m+d+hr+mi

    if land_only == True:
        subset_type = '-land'
    else:
        subset_type = '-all'

    file_save_str = 'GLM'+glm_sat+'_first-flash-data'+subset_type+'_v'+str(ver)+'_'+file_start_str+'_'+file_end_str+'_'+file_cur_str+'.nc'
    file_loc_str = data_loc+'GLM'+glm_sat+'-ff-processed/'
    #file_loc_str = '' #Dev mode
    
    return file_save_str, file_loc_str, cur_time


# In[4]:


def meta_create_v2():
    '''
    This function creates the meta data that gets used when formatting the netCDF file.
    Yes it's a mess, but it's my mess
    PARAMS:
        None
    RETURNS:
        meta_dict: A dictionary containing the meta-data that formats the netCDF file (dictionary)
        f_keys: The flash-level keys used in the meta-dict dictionary (list of strings)
        g_keys: The group-level keys used in the meta-dict dictionary (list of strings)
        e_keys: The event-level keys used in the meta-dict dictionary (list of strings)
        aux_keys1: The first set of auxilary keys used in the meta-dict dictionary (list of strings)
        aux_keys2: The second set of auxilary keys used in the meta-dict dictionary (list of strings)
        compress_dict: The dictionary that contains the information for compressing the data (not used but may be useful someday) (dictionary)
    '''

    f_keys = ['flash_id','flash_time_offset_of_first_event','flash_time_offset_of_last_event','flash_lat','flash_lon','flash_area','flash_energy','flash_quality_flag']
    g_keys = ['group_id','group_time_offset','group_lat','group_lon','group_area','group_energy','group_parent_flash_id','group_quality_flag']
    e_keys = ['event_id','event_time_offset','event_lat','event_lon','event_energy','event_parent_group_id']
    aux_keys1 = ['flashes_after_5_minutes','flashes_after_10_minutes','flashes_after_15_minutes','flashes_after_20_minutes','flashes_after_25_minutes','flashes_after_30_minutes']
    aux_keys2 = ['flash_fistart_flid','group_fistart_flid','event_fistart_flid']
    
    #Creating the meta data that we'll need in the files
    # 0 = units
    # 1 = long_name
    # 2 = dimension_name
    # 3 = data type identifier
    # 4 = compression needed (True/False)

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
    
    # Creating the data for compressing the values back into the form needed in the netCDF4 file (USING GOES-R standard values)
    # 0 = scale_factor (-999 if unnecessary)
    # 1 = add_offset (-998 if unnecessary)
    # 2 = valid_range (-997 if unnecessary)
    # 

    compress_dict = {
        'event_time_offset':[0.0003814756, -5.0, [-997]],
        'event_lat':[0.00203128, -66.56, [-997]],
        'event_lon':[0.00203128, -141.56, [-997]],
        'event_energy':[1.9024e-17, 2.8515e-16, [-997]],
        'group_time_offset':[0.0003814756, -5.0, [-997]],
        'group_area':[152601.86, 0.0, [0, -6]],
        'group_energy':[9.9988e-17, 2.8515e-16, [0, -6]],
        'group_quality_flag': [-999, -998, [0, 5]],
        'flash_time_offset_of_first_event':[0.0003814756, -5.0, [-997]],
        'flash_time_offset_of_last_event':[0.0003814756, -5.0, [-997]],
        'flash_area':[152601.86, 0.0, [0, -6]],
        'flash_energy':[9.99996e-16, 2.8515e-16, [0, -6]],
        'flash_quality_flag':[-999, -998, [0, 5]]
        }

    var_names = list(meta_dict.keys())

    return meta_dict, f_keys, g_keys, e_keys, aux_keys1, aux_keys2, compress_dict


# In[5]:


def output_netcdf_setup(out,meta_dict,compress_dict):
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
    flash_num = out.createDimension('number_of_flashes', None)
    group_num = out.createDimension('number_of_groups', None)
    event_num = out.createDimension('number_of_events', None)

    #File meta data
    out.author='Kevin Thiel (kevin.thiel@ou.edu)'
    out.time_coverage_start=str(start_time)
    out.time_coverage_end=str(end_time)
    out.date_created=str(create_time)
    out.glm_number = str(glm_sat)
    out.title='Collected GLM'+glm_sat+' First Flashes'
    out.first_flash_search_time = str(search_m)+' minutes'
    out.first_flash_search_radius = str(search_r)+ ' km'
    out.first_flash_search_radius_flash_area = str(search_fr)+ 'km'
    out.lat_max = str(50)
    out.lat_min = str(24)
    out.lon_max = str(-66)
    out.lon_min = str(-125)
    out.land_flashes_only = str(land_only)
    out.summary='This is collected GLM data for the GLM first-flash project. First flash events were identifed using spatial and temporal thresholds. Flash, group, and event data were pulled from the GLM LCFA L2 files for analysis and further calculation. Enjoy!'
    
    #Creating the variables that will go in the test
    for key in list(meta_dict.keys()):
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

#         if meta_dict[key][4] == True:
#             if compress_dict[key][0]!=-999:
#                 var.scale_factor = compress_dict[key][0]
#             if compress_dict[key][1]!=-998:
#                 var.add_offset = compress_dict[key][1]
#             if compress_dict[key][2][0]!=-997:
#                 var.valid_range = compress_dict[key][2]


    
    return out


# In[6]:


def netcdf_filler(out, f_keys, g_keys, e_keys, aux_keys1, aux_keys2, meta_dict, dset, flash_id, ff_df, file):
    '''
    This function removes all of the points that are not on land from the compiled raw GLM dataframe
    PARAMS:
        out: The output netCDF file in append mode (netCDF file)
        f_keys: The flash-level keys used in the meta-dict dictionary (dictionary)
        g_keys: The group-level keys used in the meta-dict dictionary (dictionary)
        e_keys: The event-level keys used in the meta-dict dictionary (dictionary)
        aux_keys1: The first set of auxilary keys used in the meta-dict dictionary (list of strings)
        aux_keys2: The second set of auxilary keys used in the meta-dict dictionary (list of strings)
        meta_dict: A dictionary containing the meta-data that formats the netCDF file (dictionary)
        dset: The netCDF file we are currently extracting data from (netCDF file)
        flash_id: The flash_id of the current falsh being extracted from the file (int)
        ff_df: A dataframe of the compiled raw GLM data on land (DataFrame)
        file: The file string of the netCDF file we're extracting from (string)
    RETURNS:
        out: The output netCDF file in write mode with the correct format (netCDF)
    '''
    
    #Getting the flash, group, and event locations within the GLM L2 file
    flash_locs = np.where(dset.variables['flash_id'][:]==flash_id)[0]
    group_locs = np.where(dset.variables['group_parent_flash_id'][:]==flash_id)[0]
    group_ids = np.unique(dset.variables['group_id'][group_locs])#Getting all of the group ids within a flash
    event_locs = np.array([],dtype=int)
    
    #Need to loop through each group within the flash to find the events
    for g_id in group_ids:
        cur_event_loc = np.where(dset.variables['event_parent_group_id']==g_id)[0]
        event_locs = np.append(event_locs,cur_event_loc)
    
    #aux data2 (file names for the flashes, groups, and events so they are all traceable)
    new_f_files = [file+str(flash_id) for i in range(len(flash_locs))] #Creating an array of file start and flash id names based on the number of flashes (1)
    new_g_files = [file+str(flash_id) for i in range(len(group_locs))] #Creating an array of file start and flash id names based on the number of groups
    new_e_files = [file+str(flash_id) for i in range(len(event_locs))] #Creating an array of file start and flash id names based on the number of events

    
    #Flash: Placing the variables for each flash together        
    for a in range(len(f_keys)): #Single loop for all flash level variables
        cur_var = f_keys[a]
        dim_len = out.dimensions[meta_dict[cur_var][2][0]].size #Getting the current dimension length
        if a == 0: #IF the first flash variable, need to add a new index for it to go in
            out.variables[cur_var][dim_len] = dset.variables[cur_var][:].data[flash_locs[0]] #Using the dimension length as the new INDEX for the new data
        else:
            out.variables[cur_var][dim_len-1] = dset.variables[cur_var][:].data[flash_locs[0]] #Using the dimension length as the new INDEX for the new data
    
    #Adding in the fistart_flid name for the flash
    out.variables[aux_keys2[0]][dim_len-1] = new_f_files[0] 

    
    #Groups: Looping through each group, and inside its variables in the GLML2 file and putting them into the netCDF
    for a in range(len(group_locs)): #Outer loop so we're going through each group
        for b in range(len(g_keys)): #Inner loop so we're going throgh each variable of the group
            cur_var = g_keys[b] #Getting the current variable name that we'll reference for in GLM L2 and the output netCDF files
            dim_len = out.dimensions[meta_dict[cur_var][2][0]].size #Getting the current dimension length
            if b == 0:
                out.variables[cur_var][dim_len] = dset.variables[cur_var][:].data[group_locs[a]] #Using the dimension length as the new INDEX for the new data
            else:
                out.variables[cur_var][dim_len-1] = dset.variables[cur_var][:].data[group_locs[a]] #Using the dimension length (-1) as the current  INDEX for the new data
    #Putting all of the fistart_flid names for the groups together 
    for i in range(len(new_g_files)):
        rev_index = i*-1
        out.variables[aux_keys2[1]][rev_index-1] = new_g_files[rev_index-1]

    #Events: Looping through each event, and inside its variables in the GLML2 file and putting them into the netCDF
    for a in range(len(event_locs)): #Outer loop so we're going through each event
        for b in range(len(e_keys)): #Inner loop so we're going throgh each variable of the event
            cur_var = e_keys[b] #Getting the current variable name that we'll reference for in GLM L2 and the output netCDF files
            dim_len = out.dimensions[meta_dict[cur_var][2][0]].size #Getting the current dimension length
            if b == 0:
                out.variables[cur_var][dim_len] = dset.variables[cur_var][:].data[event_locs[a]] #Using the dimension length as the new INDEX for the new data
            else:
                out.variables[cur_var][dim_len-1] = dset.variables[cur_var][:].data[event_locs[a]] #Using the dimension length (-1) as the current  INDEX for the new data
   
    #Putting all of the fistart_flid names for the groups together 
    for i in range(len(new_e_files)):
        rev_index = i*-1
        out.variables[aux_keys2[2]][rev_index-1] = new_e_files[rev_index-1]
            
    df_keys = ['5_after','10_after','15_after','20_after','25_after','30_after']

    #Aux data1 (flash counts after the first flash)
    for i in range(len(df_keys)):
        df_key = df_keys[i]
        aux_key1 = aux_keys1[i]
        dim_len = out.dimensions[meta_dict[aux_key1][2][0]].size #getting the current dimension length
        out.variables[aux_key1][dim_len-1] = ff_df.loc[(ff_df['fstart']==file)&(ff_df['flash_id']==flash_id),df_key] #Using the dimension length as the new INDEX for the new data


    return out


# # Processing Land

# In[7]:


#This chunk of code gets things set up to then pull the raw files and create the netCDF file

datetime_start = datetime.now()

#==========================
# Run like this python GLM-ff-raw-converter-v3.py 20220501 20220502 16 land
#==========================

args = sys.argv
#args = ['','20220504','20220505','16','land] 
#User inputs
start_time_str = args[1]
end_time_str = args[2]
glm_sat = args[3]
land_only = (args[4] == 'land')

#Converting into datetimes
start_time = datetime.strptime(start_time_str, '%Y%m%d') 
end_time = datetime.strptime(end_time_str, '%Y%m%d') 
#Creating list of dates to pull from when compiling the dataset
date_list = pd.date_range(start=start_time,end=end_time,freq='1D',inclusive='left').to_list()

#Defined constants
search_r = 0
search_m = 20
search_fr = 20
ver = 32
data_loc = '/localdata/first-flash/data/'
#data_loc = '../../test-data/GLM16_ffRAW_v2/20220504/' #Dev Mode


# In[8]:


#Now we load in the raw files (IN ORDER SO HELP ME GOD)

ff_df = pd.DataFrame()

#Looping through each date requested to find the glm first flashes and incldue them in the dataset
for stime in date_list:
    print (stime)
    #Compiling the dataframe from the entire day of first flash events
    new_df = raw_finder(stime, glm_sat, data_loc, search_r, search_m, search_fr, ver)
    ff_df = pd.concat((new_df,ff_df))


#Setting the index as unique flash ids to hopefully put them in order
if ff_df.columns[0] != 'fistart_flid':
    ff_df.rename(columns={ff_df.columns[0]: 'fistart_flid'}, inplace=True) #A workaround because not all files have 'fistart_flid' as a column name (whoops!)   

ff_df = ff_df.set_index('fistart_flid')#.sort_index()

#Removing the points over land if determined by the land_only variable
if land_only == True:
    ff_df = dset_land_points(ff_df)
    
print ('First Flashes Loaded')
print (datetime.now()-datetime_start)
print ('Number of flashes: '+str(ff_df.shape[0]))


# In[9]:


#A list of all the unique file-start names from the first flashes over the 24 hour increment
df_fstart = ff_df['fstart'].unique()


# In[10]:


# Creating a netCDF file containing all of the collected data
file_save_str, file_loc_str, create_time = output_file_str_create(start_time, end_time, glm_sat, ver, data_loc)


# In[11]:


#Creating the dictionary of keys and metadata for the netcdf file to use
meta_dict, f_keys, g_keys, e_keys, aux_keys1, aux_keys2, compress_dict = meta_create_v2()


# In[12]:


#Creating the netcdf file, formatting it, and closing it so it can be in append mode
out = nc.Dataset(file_loc_str+file_save_str, mode='w', format='NETCDF4')
#out = nc.Dataset('test-file.nc', mode='w', format='NETCDF4') #Devmode
out = output_netcdf_setup(out,meta_dict,compress_dict)
out.close() #Closing the file so we can open it again and fill it with data


# In[13]:


#Loading in the data, but in append mode now
out = nc.Dataset(file_loc_str+file_save_str, mode='a', format='NETCDF4')
#out = nc.Dataset('test-file.nc', mode='a', format='NETCDF4') #Devmode


# In[14]:


# This section of code takes the unique file names and goes through them,
# so we only have to read each file once. Then we can go down to the flash
# level and extract that flash, group, and event data
counter = 0

for file in df_fstart:
    #Locating all the first-flashes with the current filename
    df_file = ff_df.loc[ff_df['fstart']==file]
    #Getting the flash ids from the files
    df_file_ids = df_file['flash_id'].values

    #Getting the date that the LCFA files will be in
    file_start_datetime = datetime.strptime(file[1:-2],'%Y%j%H%M%S')
    y, m, d, doy, hr, mi = ff.datetime_converter(file_start_datetime)

    #Finding and opening the file as dset
    collected_file = glob(data_loc+'GLM'+glm_sat+'-LCFA/'+y+m+d+'/*'+file+'*.nc')
    #collected_file = glob(data_loc+'*'+file+'*.nc')
    #print (collected_file)
    if len(collected_file)==1:
        dset = nc.Dataset(collected_file[0])
    else:
        print ('File not found!')
        continue
    
    #Inner loop on a per flash basis to add them to the data dictionaries
    for flash_id in df_file_ids:
        if counter % 1000 == 0:
            print (counter)
        counter+=1
        out = netcdf_filler(out, f_keys, g_keys, e_keys, aux_keys1, aux_keys2, meta_dict, dset, flash_id, ff_df, file)
    
    #Closing the current netCDF file
    dset.close()
    
    
print ('Output File Saved')
print (file_loc_str+file_save_str)
print (datetime.now()-datetime_start)


# In[15]:


out.close()

