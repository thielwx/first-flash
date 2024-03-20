#!/usr/bin/env python
# coding: utf-8

#===========================================================
# This script produces the images necessary for animating an
# entire case such that manual analysis of first-flash identification 
# can be done. Based on full-case-animation.py.
#
# Author: Kevin Thiel (kevin.thiel@ou.edu)
# Created: March 2024
#===========================================================

# In[87]:


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from datetime import timedelta
from metpy.plots import USCOUNTIES
import numpy as np
from glob import glob
import matplotlib.patches as mpatches
import yaml
import sys
import os


# User input time!
args = sys.argv

#Getting the name of the case
case = args[1]

#Import .yaml file
with open('full-animation-settings-manual-analysis.yaml', 'r') as f:
    sfile = yaml.safe_load(f)

#Running a check that we have an available case, if not end the program there
available_cases = sfile['available-cases']    
checker = True
for c in available_cases:
    if case == c:
        checker = False
if checker == True:
    print ('ERROR: Unknown case, please select from available list:')
    print (available_cases)
    exit()
        
#Data locations
cmip_loc = sfile[case]['cmip_loc']
e_all_loc = sfile[case]['glm_east_all_loc']
lma_station_loc = sfile[case]['lma_station_loc']
lma_flashes = sfile[case]['lma_loc']

#Output location
output_loc = sfile[case]['output_loc']
if not os.path.exists(output_loc):
    os.makedirs(output_loc)
#Other case variables
stime_str = sfile[case]['start_time']
etime_str = sfile[case]['end_time']
bounds_extent = sfile[case]['bounds']
cmip_data = sfile[case]['cmip_data_switch']
case_name = sfile[case]['case_name']
conus_checker = sfile[case]['conus_check']

#Getting the time variables established
start_time = datetime.strptime(stime_str, '%Y%m%d%H%M')
end_time = datetime.strptime(etime_str, '%Y%m%d%H%M')
time_list = pd.date_range(start=start_time, end=end_time, freq='60s')
dt10 = timedelta(minutes=10)
dt30 = timedelta(minutes=30)
dt120 = timedelta(minutes=120)


# # Function Land

# In[2]:


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


# In[3]:


def datetime_converter(time):
    '''
    This function takes in a datetime object and returns strings of time features
    PARAMS:
        time: input time (datetime object)
    Returns
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


# In[78]:


def data_cutdown(lats, lons, times, start_time, end_time, bounds_extent):
    '''
    Used to cutdown the data by time
    PARAMS:
        lats
        lons
        times
        start_time
        end_time
        bounds_extent
    RETURNS
        out_lats
        out_lons
    '''
    times = np.array(times)
    #Getting the locs where the times are settled
    t_locs = np.where((times>=start_time)&(times<=end_time))[0]
    
    #Getting the lats and lons that occupy those times
    out_lats = lats[t_locs]
    out_lons = lons[t_locs]
    
    out_lats, out_lons =  latlon_bounds_custom(out_lats, out_lons, bounds_extent)
    
    return out_lats, out_lons


# In[74]:


def latlon_bounds_custom(flash_lats, flash_lons, extent):
    '''
    Takes in the flash latitudes and longitudes determines which ones are within the domain
    PARAMS:
        flash_lats: array of flash latitudes (floats)
        flash_lons: array of flash longitudes (floats)
        extent: 
    RETURNS:
        flash_lats: array of flash lats within the domain
        flash_lons: array of flash lons within the domain
    '''
    lat_max = extent[3]
    lat_min = extent[2]
    lon_max = extent[1]
    lon_min = extent[0]
    
    latlon_locs = np.where((flash_lats<=lat_max)&(flash_lats>=lat_min)&(flash_lons<=lon_max)&(flash_lons>=lon_min))[0]
    
    return flash_lats[latlon_locs], flash_lons[latlon_locs]


# In[30]:


def CMIP_loader(cur_time, loc, conus_checker, cmip_data_temp):
    CMI = np.nan
    x = np.nan
    y = np.nan
    extent = np.nan
    sat_h = np.nan
    sat_lon = np.nan
    geo_crs = np.nan
    
    
    #If the conus check is false, then you have mesoscale scene data and can go down to the minute
    if conus_checker==False:
        y, m, d, doy, hr, mi = datetime_converter(cur_time)
        file_loc = loc + y+m+d + '/' + '*s'+ y+doy+hr+mi+ '*.nc'

    else:
        dt_int = int(t.strftime('%M'))%5 #Using the mod operator to tell us how much to adjust the time
        adjusted_time = cur_time - timedelta(minutes=dt_int-1)
        y, m, d, doy, hr, mi = datetime_converter(adjusted_time)
        file_loc = loc + y+m+d + '/' + '*s'+ y+doy+hr+mi+ '*.nc'

    collected_files = glob(file_loc)
    
    if len(collected_files)>0:
        dset = nc.Dataset(collected_files[0],'r')
        CMI = dset.variables['CMI'][:]
        CMI[CMI>280] = np.nan
        sat_lon = dset.variables['goes_imager_projection'].longitude_of_projection_origin
        sat_sweep_axis = dset.variables['goes_imager_projection'].sweep_angle_axis
        sat_h = dset.variables['goes_imager_projection'].perspective_point_height
        geo_crs = ccrs.Geostationary(central_longitude=sat_lon,satellite_height=sat_h)
        
        #ABI cordinates
        x = dset.variables['x'][:] * sat_h
        y = dset.variables['y'][:] * sat_h
        extent = (np.min(x), np.max(x), np.min(y), np.max(y))
        
    else:
        print ('ERROR: DATA MISSING')
        print (file_loc)
        cmip_data_temp = False
    
    
    return CMI, x, y, extent, sat_h, sat_lon, geo_crs, cmip_data_temp



#This function takes in a file start time and the first/last event times to create a list of datetime objects
def LMA_times_postprocess(file_times, times):
    '''
    Creates a list of datetime objects from the LCFA L2 file times, and the start time of the file
    PARAMS:
        file_times: listed times on the lma file (str)
        times: times (seconds since the day begins) from the lma file (float)
    RETURNS
        flash_datetime: a list of datetimes based on the flash/group/event times in the LCFA L2 file down to ns (datetime)
    '''
    
    #Converting to nanoseconds to use for timedelta
    nanosecond_times = times*(10**9)
    
    #Creating datetime object for the file time
    flash_file_datetime = [np.datetime64(datetime.strptime(file_times[i][:8], '%Y%m%d')) for i in range(len(file_times))]
    
    #Creating timedetla objects from our array
    flash_timedelta = [np.timedelta64(int(val), 'ns') for val in nanosecond_times]
    
    #Creating an array of datetime objects with the (more) exact times down to the microsecond
    #flash_datetime = [flash_file_datetime+dt for dt in flash_timedelta]
    flash_datetime = [flash_file_datetime[i] + flash_timedelta[i] for i in range(len(flash_timedelta))]

    return (flash_datetime)


#=======================================================================
# # Processing and plotting

# In[32]:


#Loading all the GLM flashes from that day
e_all = pd.read_csv(e_all_loc)

#Loading the LMA flashes from that day
lma_all = pd.read_csv(lma_flashes, index_col=0)

#Making the times into datetimes
e_all['start_time'] = pd.to_datetime(e_all['start_time'])
lma_all['flash_start_time'] = LMA_times_postprocess(lma_all['file_time'].values,lma_all['start'].values)
# In[50]:


#Getting what we need to plot the lma data
lma_df = pd.read_csv(lma_station_loc)
lma_df = lma_df.loc[lma_df['active']=='A']

lat_avg = lma_df['lat'].mean()
lon_avg = lma_df['lon'].mean()

R = 6371.0087714 #Earths radius in km
d = 100 #Search Radius

r = (d/R)*(180/np.pi)


# In[86]:


#Looping through the list of times to find the right data and then plot it
for t in time_list:
    yr,m,d,doy,hr,mi = datetime_converter(t)
    cmip_data_temp = False # A temporary switch in case we need to turn off CMIP13 data because of a data gap
    print(t)
    
    #Getting all GLM flashes from the last 10 minutes
    glmeast_all_lat, glmeast_all_lon = data_cutdown(e_all['lat'].values,e_all['lon'].values,e_all['start_time'], t-dt10,t, bounds_extent)
    lma_lat, lma_lon = data_cutdown(lma_all['ctr_lat'].values, lma_all['ctr_lon'].values, lma_all['flash_start_time'], t-dt10, t, bounds_extent)
   
    if cmip_data==True:
        #Getting the ABI data
        cmi, x, y, extent, sat_h, sat_lon, geo_crs, cmip_data_temp = CMIP_loader(t, cmip_loc, conus_checker, cmip_data_temp)
    
    pltcar_crs = ccrs.PlateCarree()    
    
    #Plot Time!
    fig = plt.figure(figsize=(8,8))
    fig.patch.set_facecolor('white')
    if cmip_data_temp == True:
        ax = plt.axes(projection=geo_crs)
    else:
        ax = plt.axes(projection=pltcar_crs)
    
    if (cmip_data==True) & (cmip_data_temp==True):
        ax.imshow(cmi,extent=extent,cmap=plt.get_cmap('nipy_spectral_r', 24), alpha=0.4, vmin=180, vmax=300, zorder=0)
        
    ax.scatter(x=glmeast_all_lon,y=glmeast_all_lat, color='r', alpha=0.4, s=10, zorder=2,transform=pltcar_crs, label='GLM East Flashes (10min)', marker='x')
    ax.scatter(x=lma_lon, y=lma_lat, color='k', alpha=0.4, s=10, zorder=1, transform=pltcar_crs, label='LMA Flashes (10min)', marker='o')    

    ax.scatter(x=lma_df['lon'], y=lma_df['lat'], color='g', label='LMA Stations', zorder=1, transform=pltcar_crs, marker='^', s=20)
    ax.add_patch(mpatches.Circle(xy=[lon_avg, lat_avg], radius=r, color='k', alpha=0.5, transform=ccrs.PlateCarree(), zorder=4, label='100 km LMA radius', fill=False))

    
    plt.title(case_name + ' Case Study\n'+str(t) + ' UTC', loc='left', fontsize=16)
    ax.set_extent(bounds_extent)
    ax.coastlines()
    ax.add_feature(USCOUNTIES, edgecolor='gray')
    ax.add_feature(cfeature.STATES, edgecolor ='r',linewidth=1.5)
    ax.legend(fontsize=10,loc='lower right')
    
    fig_name = case+'_'+yr+m+d+'-'+hr+mi+'_all.png'
    plt.savefig(output_loc+fig_name)
    plt.close()
