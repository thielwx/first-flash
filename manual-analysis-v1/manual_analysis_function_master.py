#!/usr/bin/env python
# coding: utf-8

# In[1]:

#=================================================================================
#This script is a hub for all major GLM functions used during the first-flash project
# Author: Kevin Thiel (kevin.thiel@ou.edu)
# Created: October 2023
#=================================================================================


# In[2]:


import numpy as np
import netCDF4 as nc
import pandas as pd
from datetime import datetime
from datetime import timedelta
from sklearn.neighbors import BallTree
from glob import glob


# In[ ]:


def datetime_converter(time):
    '''
    This function takes in a datetime object and returns strings of time features
    PARAMS:
        time: input time (datetime object)
    RETURNS:
        y: year (4 digit number as str)
        doy: day of year (3 digit number as str)
        hr: hour of day (2 digit number as str)
        mi: minute of hour (2 digit number as str)
        d: day of month (2 digit number as str)
        m: month of year (2 digit number as str)
    '''
    y = datetime.strftime(time,'%Y') #Year
    doy = datetime.strftime(time,'%j') #Day of year
    hr = datetime.strftime(time,'%H') #Hour
    mi = datetime.strftime(time,'%M') #Minute
    d = datetime.strftime(time,'%d') #Day of month
    m = datetime.strftime(time,'%m') #Month
    
    return y, m, d, doy, hr, mi


# In[ ]:


#This function is used to load up a bunch of data into a pandas dataframe
def data_loader_list(start_time, end_time, glm_sat):
    '''
    This function is used to create a list of files within a specified time range to load
    PARAMS:
        start_time: start time of desired file list (datetime object)
        end_time: end time of desired file list (datetime object)
        glm_sat: the satelite number of the glm satellite (int)
    RETURNS:
        chunk files: a list of file names and locations all as str
    '''
    #Making a list of times that we can pull from
    chunk_list = pd.date_range(start=start_time, end=end_time, freq='1min')[:-1]
    chunk_files = []
    
    
    #Loop that collect the file names in 60 minute swaths
    for i in range(len(chunk_list)):
        y, m, d, doy, hr, mi = datetime_converter(chunk_list[i])
        date1 = y+m+d
        date2 = y+doy+hr+mi
        
        file_loc = '/localdata/first-flash/data/GLM'+str(glm_sat)+'-LCFA/'+str(date1)+'/OR_GLM-L2-LCFA_G'+str(glm_sat)+'_s'+str(date2)+'*.nc' #Production Mode
        #file_loc = '../../test-data/GLM'+str(glm_sat)+'-LCFA/'+str(date1)+'/OR_GLM-L2-LCFA_G'+str(glm_sat)+'_s'+str(date2)+'*.nc' #Dev mode
        
        files = sorted(glob(file_loc))
        print (files)
        chunk_files = np.append(chunk_files,files)
       
    return chunk_files


# In[ ]:


def latlon_bounds(flash_lats, flash_lons):
    '''
    Takes in the flash latitudes and longitudes determines which ones are within the domain
    PARAMS:
        flash_lats: array of flash latitudes (floats)
        flash_lons: array of flash longitudes (floats) 
    RETURNS:
        latlon_locs: array of indicies from which the input lat/lon values are within the domain
        flash_lats: array of flash lats within the domain
        flash_lons: array of flash lons within the domain
    '''
    lat_max = 50
    lat_min = 24
    lon_max = -66
    lon_min = -125
    
    latlon_locs = np.where((flash_lats<=lat_max)&(flash_lats>=lat_min)&(flash_lons<=lon_max)&(flash_lons>=lon_min))[0]
    
    return latlon_locs, flash_lats[latlon_locs], flash_lons[latlon_locs]

def latlon_bounds_custom(flash_lats, flash_lons, bounds, dx):
    '''
    Takes in the flash latitudes and longitudes determines which ones are within the domain
    PARAMS:
        flash_lats: array of flash latitudes (floats)
        flash_lons: array of flash longitudes (floats) 
        bounds: String of values containing custom spatial bounds in the format [ul_lat,ul_lon,lr_lat,lr_lon]
        dx: padding (if needed)
    RETURNS:
        latlon_locs: array of indicies from which the input lat/lon values are within the domain
        flash_lats: array of flash lats within the domain
        flash_lons: array of flash lons within the domain
    '''
    lat_max = bounds[0] + dx
    lat_min = bounds[2] - dx
    lon_max = bounds[1] + dx
    lon_min = bounds[3] - dx
    
    latlon_locs = np.where((flash_lats<=lat_max)&(flash_lats>=lat_min)&(flash_lons<=lon_max)&(flash_lons>=lon_min))[0]
    
    return latlon_locs, flash_lats[latlon_locs], flash_lons[latlon_locs]


# In[ ]:

#This function takes in a file start time and the first/last event times to create a list of datetime objects
def GLM_LCFA_times(file_time, times):
    '''
    Creates a list of datetime objects from the LCFA L2 file times, and the start time of the file
    PARAMS:
        file_time: listed time on the LCFA L2 file (str)
        times: times (seconds) from the LCFA L2 file (float)
    RETURNS
        flash_datetime: a list of datetimes based on the flash/group/event times in the LCFA L2 file down to ns (datetime)
    '''
    
    #Converting to nanoseconds to use for timedelta
    nanosecond_times = times*(10**9)
    
    #Creating datetime object for the file time
    flash_file_datetime = np.datetime64(datetime.strptime(file_time, '%Y-%m-%dT%H:%M:%S.0Z'))
    
    #Creating timedetla objects from our array
    flash_timedelta = [np.timedelta64(int(val), 'ns') for val in nanosecond_times]
    
    #Creating an array of datetime objects with the (more) exact times down to the microsecond
    flash_datetime = [flash_file_datetime+dt for dt in flash_timedelta]
    
    return (flash_datetime)


#This function takes in a file start time and the first/last event times to create a list of datetime objects
def GLM_LCFA_times_postprocess(file_times, times):
    '''
    Creates a list of datetime objects from the LCFA L2 file times, and the start time of the file
    PARAMS:
        file_times: listed times on the LCFA L2 file (str)
        times: times (seconds) from the LCFA L2 file (float)
    RETURNS
        flash_datetime: a list of datetimes based on the flash/group/event times in the LCFA L2 file down to ns (datetime)
    '''
    
    #Converting to nanoseconds to use for timedelta
    nanosecond_times = times*(10**9)
    
    #Creating datetime object for the file time
    flash_file_datetime = [np.datetime64(datetime.strptime(file_times[i], 's%Y%j%H%M%S0')) for i in range(len(file_times))]
    
    #Creating timedetla objects from our array
    flash_timedelta = [np.timedelta64(int(val), 'ns') for val in nanosecond_times]
    
    #Creating an array of datetime objects with the (more) exact times down to the microsecond
    #flash_datetime = [flash_file_datetime+dt for dt in flash_timedelta]
    flash_datetime = [flash_file_datetime[i] + flash_timedelta[i] for i in range(len(flash_timedelta))]

    return (flash_datetime)



def data_loader(file_list):
    '''
    PARAMS:
        file_list: The list of files returned from the data_loader_list function\
    RETURNS:
        df: A dataframe of all the files within that list
    
    '''
    df = pd.DataFrame()
    
    for i in range(len(file_list)):
        dset = nc.Dataset(file_list[i],'r')
        
        #Grabbing basic info to sort our flashes within a specificed domain (rectangular in lat/lon space)
        flash_lats = dset.variables['flash_lat'][:]
        flash_lons = dset.variables['flash_lon'][:]

        flash_locs, flash_lats, flash_lons = latlon_bounds_custom(flash_lats,flash_lons)
        #NOTE: You'll need to constrain all calculations to point within these bounds (using flash_locs)

        flash_lats_rad = flash_lats * (np.pi/180)
        flash_lons_rad = flash_lons * (np.pi/180)
        
        #Grabbing the flash start time and flash end times (time=np.timedelta64('ns'))
        flash_start_times = GLM_LCFA_times(dset.time_coverage_start, dset.variables['flash_time_offset_of_first_event'][flash_locs])
        flash_ids = dset.variables['flash_id'][flash_locs]
        
        file_start_time = file_list[i][-50:-34]
        fstart_time_array = np.full(len(flash_lats),file_start_time)
        
        #Creating a dictionary
        d = {'start_time':flash_start_times,
            'lat':flash_lats,
            'lon':flash_lons,
            'lat_rad':flash_lats_rad,
            'lon_rad':flash_lons_rad,
            'flash_id':flash_ids,
            'fstart':fstart_time_array
            }
        
        #Putting it into a dataframe
        df_new = pd.DataFrame.from_dict(d)
        
        #Creating the index for the file so it's unique to each value
        df_new = index_creator(df_new,file_list[i])
        
        #Adding the data to the file
        df = pd.concat((df, df_new), axis=0)
        
        #closing the file
        dset.close()
    
    return df


def data_loader_gridsearch(file_list, bounds):
    '''
    PARAMS:
        file_list: The list of files returned from the data_loader_list function
        bounds: String of values containing custom spatial bounds in the format [ul_lat,ul_lon,lr_lat,lr_lon]
    RETURNS:
        df: A dataframe of all the files within that list
    
    '''
    df = pd.DataFrame()
    
    for i in range(len(file_list)):
        dset = nc.Dataset(file_list[i],'r')
        
        #Grabbing basic info to sort our flashes within a specificed domain (rectangular in lat/lon space)
        flash_lats = dset.variables['flash_lat'][:]
        flash_lons = dset.variables['flash_lon'][:]

        #Cutting down the data to case-specific bounds (with padding of 0.5 degrees)
        flash_locs, flash_lats, flash_lons = latlon_bounds_custom(flash_lats, flash_lons, bounds, 0.5)
        #NOTE: You'll need to constrain all calculations to point within these bounds (using flash_locs)

        flash_lats_rad = flash_lats * (np.pi/180)
        flash_lons_rad = flash_lons * (np.pi/180)
        
        #Grabbing the flash start time and flash end times (time=np.timedelta64('ns'))
        flash_start_times = GLM_LCFA_times(dset.time_coverage_start, dset.variables['flash_time_offset_of_first_event'][flash_locs])
        flash_end_times = GLM_LCFA_times(dset.time_coverage_start, dset.variables['flash_time_offset_of_last_event'][flash_locs])
        flash_ids = dset.variables['flash_id'][flash_locs]
        flash_areas = dset.variables['flash_area'][flash_locs]
        flash_quality_flag = dset.variables['flash_quality_flag'][flash_locs]
        num_events, num_groups = events_per_flash(event_parent_ids=dset.variables['event_parent_group_id'][:],
                                                  group_ids=dset.variables['group_id'][:],
                                                  group_parent_ids=dset.variables['group_parent_flash_id'][:],
                                                  flash_ids = flash_ids)
        
        file_start_time = file_list[i][-50:-34]
        fstart_time_array = np.full(len(flash_lats),file_start_time)
        
        #Creating a dictionary
        d = {'start_time':flash_start_times,
             'end_time':flash_end_times,
            'lat':flash_lats,
            'lon':flash_lons,
            'lat_rad':flash_lats_rad,
            'lon_rad':flash_lons_rad,
            'flash_id':flash_ids,
            'fstart':fstart_time_array,
            'flash_area':flash_areas,
            'flash_quality_flag': flash_quality_flag,
            'num_events': num_events,
            'num_groups': num_groups
            }
        
        #Putting it into a dataframe
        df_new = pd.DataFrame.from_dict(d)
        
        #Creating the index for the file so it's unique to each value
        df_new = index_creator(df_new,file_list[i])
        
        #Adding the data to the file
        df = pd.concat((df, df_new), axis=0)
        
        #closing the file
        dset.close()
    
    return df
# In[ ]:

def index_creator(df,time):
    '''
    Getting the start time of the current file and making it into an array equal to the number of rows
    PARAMS:
        df: dataframe contianing the data from the files
    RETURNS:
        df: new dataframe with the file starttime and id number as the index for each flash
    '''
    
    #Getting a list of strings contianing the file start times
    fstart_time_array = [str(a) for a in df['fstart'].values]
    
    #Getting a list of strings contianing the flash_ids available
    flash_ids = [str(a).zfill(5) for a in df['flash_id'].values]
    
    if len(flash_ids)>0:
        #Combining the two strings together element by element to make a bigger one
        new_index = np.char.add(fstart_time_array, flash_ids)

        #Adding the new index to the array based on fi(le)start and fl(ash)id and making it the index
        df['fistart_flid'] = new_index
        df = df.set_index('fistart_flid')
    
    else:
        df = df.rename(index={'fistart_flid'})
    
    return df


# A function that retrieves the shape and slope of each flash

def LCFA_shape_slope(group_energy, group_parent_ids, flash_ids, group_times, flash_start_times, flash_end_times):
    '''
    Getting the shape and slope parameter (see Ringhausen et al. 2021) from the flashes
    PARAMS:
        group_energy: group energy data (fJ)
        group_parent_ids: group parent IDs to relate to the falsh_ids
        flash_ids: flash_IDs
        group_times: group times down to ns (datetime)
        flash_start_times: flash start times down to ns (datetime)
        flash_end_times: flash end t imes down to ns (datetime)
    RETURNS
        flash_slopes: array of flash slopes (fJ/s)
        flash_shapes: array of flash shape indicies
    '''
    
    #The arrays that we will be filling
    flash_slopes = []
    flash_shapes = []

    for i in range(len(flash_ids)): #Starting with a per flash basis
    
        #Getting the locations of the groups for each flash
        flash2group_locs = np.where(group_parent_ids == flash_ids[i])[0] 

        #Grabbing the energy (fJ) and time (down to ns) of each group in the flash
        flash_group_energy = group_energy[flash2group_locs]
        flash_group_times = group_times[flash2group_locs]

        #Calculating the mid-time-point in the flash
        flash_half_point = flash_start_times[i] + (flash_end_times[i] - flash_start_times[i])/2

        #Separating the groups between happening in the first/second half of the flash
        first_half_group_locs = np.where(flash_group_times <= flash_half_point)[0]
        second_half_group_locs = np.where(flash_group_times > flash_half_point)[0]

        #Finding the index of the max group energy in the first/second half of the flash
        first_half_max_loc = np.argmax(flash_group_energy[first_half_group_locs])
        second_half_max_loc = np.argmax(flash_group_energy[second_half_group_locs])

        #Calculating the difference in energy and time between the max group flash energy b/w the first/second half of each flash
        delta_energy = flash_group_energy[second_half_group_locs][second_half_max_loc] - flash_group_energy[first_half_group_locs][first_half_max_loc]
        delta_time = flash_group_times[second_half_group_locs][second_half_max_loc] - flash_group_times[first_half_group_locs][first_half_max_loc]

        #Calculating the slope of the flash in units of fJ/s
        slope = delta_energy/(delta_time.item()/10**9)
        flash_slopes = np.append(flash_slopes,slope)
        
        #Calculating the shape of the flash
        shape = len(first_half_group_locs)/len(flash2group_locs)
        flash_shapes = np.append(flash_shapes,shape)
        
    return flash_slopes, flash_shapes


# In[ ]:


#This function uses the parent child relationships b/w flashes-to-groups and groups-to-events to
#count the number of events and groups per flash
def events_per_flash(event_parent_ids, group_ids, group_parent_ids, flash_ids):
    '''
    Calculating the events per flash using all the ids
    PARAMS:
        event_parent_ids: IDs of the event parents (tied to groups)
        group_ids: IDs of the groups
        group_parent_ids: IDs of the group parents (tied to flashes)
        flash_ids: IDs of the flashes
    RETURNS:
        num_events: The number of events in each flash
    '''
    
    num_events = [] #Empty array we will fill with the number of events per flash
    num_groups = [] #Empty array we will fill with the number of groups per flash

    #Looping on a per flash basis
    for flash_id in flash_ids:
        flash_to_group_loc = np.where(group_parent_ids==flash_id)[0]
        group_id_parents = group_ids[flash_to_group_loc]

        num_groups = np.append(num_groups, len(group_id_parents))

        events_in_flash = 0
        #looping on a per group basis to count the events
        for group_id in group_id_parents:
            group_to_event_loc = np.where(event_parent_ids==group_id)[0]
            events_in_group = len(group_to_event_loc)

            events_in_flash += events_in_group

        num_events = np.append(num_events,events_in_flash)
        
    return num_events, num_groups


# In[ ]:


def ff_hunter(df, search_start_time, search_end_time, search_r, search_m):
    '''
    Funciton used to identify first flash events. Using the Ball Tree from scikitlearn
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
    PARAMS:
        df: Dataframe containing the start_time, lat, lon, lat_rad, lon_rad
        search_start_time: start time used to depict which flashes are being investigated
        search_end_time: end time used to depict which flashes are being investigated
        search_r: Search radius of the ball tree (km)
        search_m: Time window to search in (minutes)
    RETURNS
        ff_df: A dataframe containing only the first flash events
    '''
    R = 6371.0087714 #Earths radius in km
    
    #Need to institute this below
    t_delta = timedelta(minutes=search_m)

    ff_df = pd.DataFrame()
    
    #Pruning our indicies to only the ones that are after the current search timeframe
    df_search = df.loc[(df['start_time'] >= search_start_time) & (df['start_time'] <= search_end_time)]
    
    #This loop goes through based upon the index of the provided dataframe and finds the first flashes
    #The output is a dataframe of first flash events 
    for i in df_search.index.values:
        #Getting the current lat, lon, and index
        c_pt = df.loc[i][['lat_rad','lon_rad']].values
        c_stime = df.loc[i][['start_time']].values[0]

        #Removing the flashes that happened 30 min before and anything after the current flash from consideration
        time_prev = df.loc[i]['start_time'] - t_delta #Finding the time from the previous 30 minutes
        df_cut = df.loc[(df.loc[i][['start_time']].values[0] >= df['start_time']) & 
                         (df['start_time'] >= time_prev)]
        
        #Making a smaller tree to reduce the required memory (and increase speed) of the ball tree
        dx = 0.5 #Change in latitude max. Using a blanket benchmark to reduce the number of distance calculations made
        df_cut = df_cut.loc[(df_cut['lat'] <= (df_search.loc[i])['lat']+dx) &
                           (df_cut['lat'] >= (df_search.loc[i]['lat']-dx)) &
                           (df_cut['lon'] <= (df_search.loc[i]['lon']+dx)) &
                           (df_cut['lon'] >= (df_search.loc[i]['lon']-dx))]

        #Setting up and running a ball tree
        btree = BallTree(df_cut[['lat_rad','lon_rad']].values, leaf_size=2, metric='haversine')
        indicies = btree.query_radius([c_pt], r = search_r/R)

        #If only the point itself is returned within the search distance, then
        if len(indicies[0])==1:
            ff_df = pd.concat((ff_df,df.loc[df.index==i]),axis=0)
            
    return ff_df

def ff_hunter_gridsearch(df, search_start_time, search_end_time, search_r, search_m, search_flash_r, bounds):
    '''
    Funciton used to identify first flash events. Using the Ball Tree from scikitlearn
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
    PARAMS:
        df: Dataframe containing the start_time, lat, lon, lat_rad, lon_rad
        search_start_time: start time used to depict which flashes are being investigated
        search_end_time: end time used to depict which flashes are being investigated
        search_r: Search radius of the ball tree (km)
        search_m: Time window to search in (minutes)
    RETURNS
        ff_df: A dataframe containing only the first flash events
    '''
    R = 6371.0087714 #Earths radius in km
    
    #Need to institute this below
    t_delta = timedelta(minutes=search_m)

    ff_df = pd.DataFrame()
    print (df['start_time'])
    #Pruning our indicies to only the ones that are after the current search timeframe
    df_search = df.loc[(df['start_time'].values >= search_start_time) & (df['start_time'].values <= search_end_time)]
    #Pruning the search indicies to only those within the case bounds
    flash_locs, flash_lats, flash_lons = latlon_bounds_custom(df['lat'].values, df['lon'].values, bounds, 0.0)
    df_search = df.iloc[flash_locs]
    
    #This loop goes through based upon the index of the provided dataframe and finds the first flashes
    #The output is a dataframe of first flash events 
    for i in df_search.index.values:
        #Getting the current lat, lon, and index
        c_pt = df.loc[i][['lat_rad','lon_rad']].values
        c_stime = df.loc[i][['start_time']].values[0]

        #Removing the flashes that happened 30 min before and anything after the current flash from consideration
        time_prev = df.loc[i]['start_time'] - t_delta #Finding the time from the previous t minutes
        df_cut = df.loc[(df.loc[i][['start_time']].values[0] >= df['start_time']) & 
                         (df['start_time'] >= time_prev)]
        
        #Making a smaller tree to reduce the required memory (and increase speed) of the ball tree
        dx = 1.0 #Change in latitude max. Using a blanket benchmark to reduce the number of distance calculations made
        df_cut = df_cut.loc[(df_cut['lat'] <= (df_search.loc[i])['lat']+dx) &
                           (df_cut['lat'] >= (df_search.loc[i]['lat']-dx)) &
                           (df_cut['lon'] <= (df_search.loc[i]['lon']+dx)) &
                           (df_cut['lon'] >= (df_search.loc[i]['lon']-dx))]

        #Setting up and running a ball tree
        btree = BallTree(df_cut[['lat_rad','lon_rad']].values, leaf_size=2, metric='haversine')
        
        if search_flash_r == 0: #If there's no value for the flash radius value, then use the traditional radius values
            indicies = btree.query_radius([c_pt], r = search_r/R)
        elif search_r == 0: #If the traditional radius value is zero, then use the circular radius of the flash area plus a buffer
            c_area = df.loc[i][['flash_area']].values[0]
            flash_r_from_area = np.sqrt((c_area/1000000)/np.pi) #Getting the radius (km) from the flash area in sq meters
            indicies = btree.query_radius([c_pt], r = (search_flash_r + flash_r_from_area)/R)


        #If only the point itself is returned within the search distance, then it's a first flash event!
        if len(indicies[0])==1:
            ff_df = pd.concat((ff_df,df.loc[df.index==i]),axis=0)
            
    return ff_df

def ff_next_flashes(df, ff_df, s_time, e_time, search_r, search_m):
    '''
    Funciton used to count the number of flashes AFTER the first flash. Using the Ball Tree from scikitlearn
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
    PARAMS:
        df: Dataframe containing the start_time, lat, lon, lat_rad, lon_rad
        ff_df: The identified first flash events
        search_start_time: start time used to depict which flashes are being investigated
        search_end_time: end time used to depict which flashes are being investigated
        search_r: Search radius of the ball tree (km)
        search_m: Time window to search in (minutes)
    RETURNS
        ff_df: A dataframe containing only the first flash events
    '''
    R = 6371.0087714 #Earths radius in km
    dx = 0.5 #Change in latitude max. Using a blanket benchmark to reduce the number of distance calculations made
    
    t_delta = timedelta(minutes=search_m)
    
    time_bins = np.arange(5,35,5)
    
    
    #Looping through by index from the first flash array
    for i in ff_df.index.values:
        #Getting the start and end times for the window relative to the first flash event
        stime_window = ff_df.loc[i]['start_time']
        etime_window = stime_window + t_delta
        
        #Current point of the data
        c_pt = ff_df.loc[i][['lat_rad','lon_rad']].values
        
        #Cutting down the full dataframe by time (NOTE: THIS REMOVES THE FIRSTFLASH FROM CONSIDERATION)
        df_tcut = df.loc[(df['start_time']>stime_window) & (df['start_time']<=etime_window)]
        
        #Making a smaller tree to reduce the required memory (and increase speed) of the ball tree
        df_cut = df_tcut.loc[(df_tcut['lat'] <= (ff_df.loc[i])['lat']+dx) &
                           (df_tcut['lat'] >= (ff_df.loc[i]['lat']-dx)) &
                           (df_tcut['lon'] <= (ff_df.loc[i]['lon']+dx)) &
                           (df_tcut['lon'] >= (ff_df.loc[i]['lon']-dx))]
        
        #Running a check that we actually need to run the ball tree
        if df_cut.shape[0] > 0:
            #Setting up and running a ball tree
            btree = BallTree(df_cut[['lat_rad','lon_rad']].values, leaf_size=2, metric='haversine')
            indicies = btree.query_radius([c_pt], r = search_r/R)[0]

        # If the cutdown dataset prior to the ball tree has no data, then there are no flashes around and no idicies 
        else:
            indicies = []
        
        #Counting the number flashes that happened after the first flash
        if len(indicies) > 0:
            #Extracting the flashes that happened within 30 min and 30 km AFTER the first flash
            df_after = df_cut.iloc[indicies]
            
            #Finding the number of flashes within each time bin
            for t in time_bins:
                #Getting the ending datetime for the time bins
                end_dt = stime_window + timedelta(minutes=int(t))
                #Trimming down to only get the flashes between each time range
                df_after_cut = df_after.loc[df_after['start_time'] <= end_dt]
                #Adding the number of flashes that happened after and within the time range to the first flash dataframe
                ff_df.loc[i,str(t)+'_after'] = df_after_cut.shape[0]
        
        # If there were no flashes within the 30 km and 30 min range, then all values go to zero
        else:
            for t in time_bins:
                ff_df.loc[i,str(t)+'_after'] = 0
                
    return ff_df