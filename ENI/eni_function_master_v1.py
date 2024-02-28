#!/usr/bin/env python
# coding: utf-8

#=================================================================================
#This script is a hub for all major ENI functions used during the first-flash project
# Author: Kevin Thiel (kevin.thiel@ou.edu)
# Created: Janaury 2024
#=================================================================================



import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from glob import glob
import global_land_mask as globe
import json
from sklearn.neighbors import BallTree


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



def latlon_bounds(flash_lats, flash_lons):
    '''
    Takes in the flash latitudes and longitudes determines which ones are within the domain
    PARAMS:
        flash_lats: array of flash latitudes (floats)
        flash_lons: array of flash longitudes (floats) 
    RETURNS:
        latlon_locs: array of indicies from which the input lat/lon values are within the domain
    '''
    #Specifying the lat/lon bounds for the first-flash domain
    lat_max = 50
    lat_min = 24
    lon_max = -66
    lon_min = -125
    
    #Finding the indicies where the lat/lon locs sit within the bounds
    latlon_locs = np.where((flash_lats<=lat_max)&(flash_lats>=lat_min)&(flash_lons<=lon_max)&(flash_lons>=lon_min))[0]
    
    return latlon_locs



def dset_land_points(flash_lats,flash_lons):
    '''
    Takes in the flash latitudes and longitudes to find the ones that are on land
    PARAMS:
        flash_lats: array of flash latitudes (floats)
        flash_lons: array of flash longitudes (floats)
    RETURNS:
        land_index: array of indexes that are classified as on land (ints)
    '''
    #Using the is_land method to find the indicies from the arrays that are on land
    land_index = globe.is_land(flash_lats, flash_lons)
    
    return land_index



def eni_loader(start_time, end_time, input_loc):
    '''
    A function for collecting ENI data from a specified time period and range (CONUS), and putting it into a single dataframe
    PARAMS:
        start_time: Beginning of the time period (datetime object)
        end_time: End of the time period (datetime object)
        input_loc: Location of the ENI data (str)
    RETURNS:
        df: The compiled flash data from the csv files (DataFrame)
    '''
    #Creating a list of times to rotate through
    time_list = pd.date_range(start=start_time, end=end_time,freq='1min').to_list()
    
    #Empty DataFrame that we'll fill
    df = pd.DataFrame(columns=('Lightning_Time_String','Latitude','Longitude','Height','Flash_Type','Amplitude','Flash_Solution','Confidence','File_String'))
    
    #Looping through the list of available times
    for cur_time in time_list:
        y,m,d,doy,hr,mi = datetime_converter(cur_time) #Turning the current time into a string
        
        #Specifying the folder by date
        file_loc = input_loc + y + m + d + '/'
        #file_loc = input_loc #DEVMODE
        
        #Creating the file string we'll use in the glob function
        file_str = y + m + d + 'T' + hr + mi + '.csv'
        
        #Collecting the files on the given day
        collected_file = sorted(glob(file_loc+file_str))
        
        if len(collected_file)==0:
            print (file_loc+file_str)
            print ('ERROR: NO FILE FOUND')
            #No file found means a data gap. To ensure data quality we need a dummy entry that we can
            # use in ENI_ff_hunter later
            fake_dictionary = {
                'Lightning_Time_String': [cur_time.strftime('%Y-%m-%dT%H:%M:%S.000000000')],
                'Latitude':[0],
                'Longitude':[0],
                'Height':[-1],
                'Flash_Type':[4],
                'Amplitude':[0],
                'Flash_Solution':[r'{"st": "'+cur_time.strftime('%Y-%m-%dT%H:%M:%S.000000000')+r'"}'],
                'Confidence':[-1],
                'File_String':[ y + m + d + '/'+ file_str]
            }
            new_df = pd.DataFrame(data=fake_dictionary) #Fake dataframe that will be appened to file

        #When there's actually data, we load it
        else:
            cfile_str = collected_file[0]

            #Reading in the collected file
            new_df = pd.read_csv(cfile_str)

            #Removing the data outside of the domain of study
            bound_index = latlon_bounds(new_df['Latitude'].values, new_df['Longitude'].values)
            new_df = new_df.iloc[bound_index,:]

            #Adding the current file string to the new dataframe so we can find it more easily later
            file_str_list = np.full(new_df.shape[0], y + m + d + '/'+ file_str)
            new_df['File_String'] = file_str_list

        #Appending the new DataFrame to the combined one
        df = pd.concat((df,new_df),axis=0)
        
    return df

def eni_loader_v2(start_time, end_time, input_loc):
    '''
    A function for collecting ENI data from a specified time period and range (CONUS), and putting it into a single dataframe
    PARAMS:
        start_time: Beginning of the time period (datetime object)
        end_time: End of the time period (datetime object)
        input_loc: Location of the ENI data (str)
    RETURNS:
        df: The compiled flash data from the csv files (DataFrame)
    '''
    #Creating a list of times to rotate through
    time_list = pd.date_range(start=start_time, end=end_time,freq='1D').to_list()
    
    #Empty DataFrame that we'll fill
    df = pd.DataFrame(columns=('type','timestamp','latitude','longitude','peakcurrent','icheight','numbersensors','icmultiplicity','cgmultiplicity','starttime','endtime','duration','ullatitude','ullongitude','lrlatitude','lrlongitude'))
  
    #Looping through the list of available times
    for cur_time in time_list:
        y,m,d,doy,hr,mi = datetime_converter(cur_time) #Turning the current time into a string
        
        #Specifying the folder by date
        file_loc = input_loc
        #file_loc = input_loc #DEVMODE
        
        #Creating the file string we'll use in the glob function
        file_str = 'eni_flash_flash'+y + m + d + '.csv'
        
        #print (file_loc+file_str)

        #Collecting the files on the given day
        collected_file = sorted(glob(file_loc+file_str))
        
        if len(collected_file)==0:
            print ('ERROR: File Missing')
            print (file_loc+file_str)
            continue

        cfile_str = collected_file[0]

        #Reading in the collected file
        new_df = pd.read_csv(cfile_str)

        #Removing the data outside of the domain of study
        bound_index = latlon_bounds(new_df['latitude'].values, new_df['longitude'].values)
        new_df = new_df.iloc[bound_index,:]

        #Adding the current file string to the new dataframe so we can find it more easily later
        
        if new_df.shape[0]>0:
            file_str_list = np.full(new_df.shape[0], file_str)
            new_df['File_String'] = file_str_list
        else:
            new_df['File_String'] = None

        #Appending the new DataFrame to the combined one
        df = pd.concat((df,new_df),axis=0)
        
    return df


def time_str_converter(time_str):
    '''
    Converts an array of time strings into an array of datetimes
    PARAMS:
        time_str: Array of strings that represent the times from ENI datasets (str). EX: df['Lightning_Time_String'].values
    RETURNS:
        time_array: An array of datetime objects from the original time_str (datetime)
    '''
    time_array = [pd.to_datetime(time_str[i]) for i in range(len(time_str))]
    
    return time_array


def json_data_loader(df, var):
    '''
    Extracts data from the json files hidden in the dataframe
    INPUTS:
        df: eni dataframe with the Flash_Solution variable
        var: which variable you would like to extract
    RETURNS:
        out_data: Array of values from all jsons within the dataset
    '''
    #Making an array that we'll fill with the real data
    out_data = np.empty(df.shape[0], dtype=object)
    
    #Grabbing the data we'll be transforming
    f_soln_str = df['Flash_Solution'].values
    
    #Looping through each flash in the dataset
    for i in range(len(f_soln_str)):
        out_data[i] = json.loads(f_soln_str[i])[var]
        
    return out_data


def eni_ff_hunter(df, search_start_time, search_end_time, search_r, search_m):
    '''
    Funciton used to identify first flash events. Using the Ball Tree from scikitlearn
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
    Adapted from first-flash/GLM/first_flash_funtion_master.py:ff_hunter
    PARAMS:
        df: Dataframe containing the start_time, lat, lon, lat_rad, lon_rad
        search_start_time: start time used to depict which flashes are being investigated
        search_end_time: end time used to depict which flashes are being investigated
        search_r: Search radius of the ball tree (km)
        search_m: Time window to search in (minutes)
    RETURNS:
        ff_df: A dataframe containing only the first flash events
    '''
    R = 6371.0087714 #Earths radius in km
    
    #Need to institute this below
    t_delta = timedelta(minutes=search_m)

    ff_df = pd.DataFrame(columns=('Lightning_Time_String','Latitude','Longitude','Height','Flash_Type','Amplitude','Flash_Solution','Confidence','File_String','start_time','lat_rad','lon_rad'))
    
    #Pruning our indicies to only the ones that are after the current search timeframe
    start_time = json_data_loader(df, 'st') #Adding the flash start times to the dataframe
    df['start_time'] = time_str_converter(start_time)

    print (df['start_time'])
    print ('----------')
    print (search_start_time)

    df_search = df.loc[(df['start_time'] >= search_start_time) & (df['start_time'] <= search_end_time)]
    
    #This loop goes through based upon the index of the provided dataframe and finds the first flashes
    #The output is a dataframe of first flash events 
    for i in df_search.index.values:

        #A first check that the current flash is not a fake flash (see eni_loader)
        if (df_search.loc[i]['Latitude']==0) & (df_search.loc[i]['Longitude']==0):
            # print ('Check1')
            # print (df_search.loc[i])
            continue

        #Getting the current lat, lon, and index
        c_pt = df.loc[i][['lat_rad','lon_rad']].values
        c_stime = df.loc[i][['start_time']].values[0]

        #Removing the flashes that happened 30+ min before and anything after the current flash from consideration
        time_prev = df.loc[i]['start_time'] - t_delta #Finding the time from the previous 30 minutes
        df_cut = df.loc[(df.loc[i][['start_time']].values[0] >= df['start_time']) & 
                         (df['start_time'] >= time_prev)]
        
        #A second check that there were not any fake flashes (missing files) in the last 30 minutes
        df_fake_check = df_cut.loc[(df_cut['Longitude']==0) & (df_cut['Latitude']==0)]
        if df_fake_check.shape[0]>0:
            # print ('Check2')
            # print (c_stime)
            continue
        
        #Making a smaller tree to reduce the required memory (and increase speed) of the ball tree
        dx = 0.5 #Change in latitude max. Using a blanket benchmark to reduce the number of distance calculations made
        df_cut = df_cut.loc[(df_cut['Latitude'] <= (df_search.loc[i])['Latitude']+dx) &
                           (df_cut['Latitude'] >= (df_search.loc[i]['Latitude']-dx)) &
                           (df_cut['Longitude'] <= (df_search.loc[i]['Longitude']+dx)) &
                           (df_cut['Longitude'] >= (df_search.loc[i]['Longitude']-dx))]

        #Setting up and running a ball tree
        btree = BallTree(df_cut[['lat_rad','lon_rad']].values, leaf_size=2, metric='haversine')
        indicies = btree.query_radius([c_pt], r = search_r/R)

        #If only the point itself is returned within the search distance, then
        if len(indicies[0])==1:
            ff_df = pd.concat((ff_df,df.loc[df.index==i]),axis=0)
            
    return ff_df

def eni_ff_hunter_v2(df, search_start_time, search_end_time, search_r, search_m):
    '''
    Funciton used to identify first flash events. Using the Ball Tree from scikitlearn
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html
    Adapted from first-flash/GLM/first_flash_funtion_master.py:ff_hunter
    PARAMS:
        df: Dataframe containing the start_time, lat, lon, lat_rad, lon_rad
        search_start_time: start time used to depict which flashes are being investigated
        search_end_time: end time used to depict which flashes are being investigated
        search_r: Search radius of the ball tree (km)
        search_m: Time window to search in (minutes)
    RETURNS:
        ff_df: A dataframe containing only the first flash events
    '''
    R = 6371.0087714 #Earths radius in km
    
    #Need to institute this below
    t_delta = timedelta(minutes=search_m)

    ff_df = pd.DataFrame(columns=('type','timestamp','latitude','longitude','peakcurrent','icheight','numbersensors','icmultiplicity','cgmultiplicity','starttime','endtime','duration','ullatitude','ullongitude','lrlatitude','lrlongitude'))
    
    #Pruning our indicies to only the ones that are after the current search timeframe
    start_time = df['starttime'].values #Adding the flash start times (as datetime objects) to the dataframe
    df['start_time'] = time_str_converter(start_time)
    df_search = df.loc[(df['start_time'] >= search_start_time) & (df['start_time'] <= search_end_time)]
    
    #This loop goes through based upon the index of the provided dataframe and finds the first flashes
    #The output is a dataframe of first flash events 
    for i in df_search.index.values:

        #Getting the current lat, lon, and index
        c_pt = df.loc[i][['lat_rad','lon_rad']].values
        c_stime = df.loc[i][['start_time']].values[0]

        #Removing the flashes that happened 30+ min before and anything after the current flash from consideration
        time_prev = df.loc[i]['start_time'] - t_delta #Finding the time from the previous 30 minutes
        df_cut = df.loc[(df.loc[i][['start_time']].values[0] >= df['start_time']) & 
                         (df['start_time'] >= time_prev)]
          
        #Making a smaller tree to reduce the required memory (and increase speed) of the ball tree
        dx = 0.5 #Change in latitude max. Using a blanket benchmark to reduce the number of distance calculations made
        df_cut = df_cut.loc[(df_cut['latitude'] <= (df_search.loc[i])['latitude']+dx) &
                           (df_cut['latitude'] >= (df_search.loc[i]['latitude']-dx)) &
                           (df_cut['longitude'] <= (df_search.loc[i]['longitude']+dx)) &
                           (df_cut['longitude'] >= (df_search.loc[i]['longitude']-dx))]

        #Setting up and running a ball tree
        btree = BallTree(df_cut[['lat_rad','lon_rad']].values, leaf_size=2, metric='haversine')
        indicies = btree.query_radius([c_pt], r = search_r/R)

        #If only the point itself is returned within the search distance, then
        if len(indicies[0])==1:
            ff_df = pd.concat((ff_df,df.loc[df.index==i]),axis=0)
            
    return ff_df
