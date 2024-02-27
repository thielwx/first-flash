#!/usr/bin/env python
# coding: utf-8

# A generizable script that can be used for any case study to look at first flash events within the one-minute period identified from the full-case-animation-v1.py script, with the ability to handle GLM, LMA, and ENI data.

# In[18]:


import yaml
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from datetime import timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES
import matplotlib.pyplot as plt
import sys
import numpy as np
import os


# In[3]:


#Getting all of our constants and file locations established

args = sys.argv
#args = ['','20220322-perils', '1'] #Devmode
#
case = args[1]
fl_num = args[2]

#Importing the appropriate yaml file
with open('flash-display-summary-settings.yaml', 'r') as f:
    sfile = yaml.safe_load(f)

#Event-specific data (spatial and temporal)
stime_str = sfile[case][fl_num]['start_time']
etime_str = sfile[case][fl_num]['end_time']
search_bounds = sfile[case][fl_num]['search_bounds']
plot_bounds = sfile[case][fl_num]['plot_bounds']
case_name = sfile[case]['case_name']

#Input data locations from yaml file
glm_east_ff_loc = sfile[case]['glm_east_ff_loc']
glm_west_ff_loc = sfile[case]['glm_west_ff_loc']
lma_flash_loc = sfile[case]['lma_flash_loc']
lma_event_loc = sfile[case]['lma_event_loc']
lma_station_loc = sfile[case]['lma_station_loc']
eni_loc = sfile[case][fl_num]['eni_loc']

#Output locations
out_data_loc = sfile['out_data_loc']
out_plot_loc = sfile['out_plot_loc']

#Getting the time variables established
start_time = np.datetime64(stime_str)
end_time = np.datetime64(etime_str)


# In[4]:


# #ALL OF THIS IS FOR DEVMODE
# stime_str = '2022-03-22T20:06:44.2'
# etime_str = '2022-03-22T20:06:47.1'
# search_bounds = [-89.0,-87.5,32.7,34.2]
# plot_bounds = [-90.0,-86,31,35]
# case_name = 'PERiLS'

# glm_east_ff_loc = '../../../local-data/20220322-perils/GLM16_first-flash-data-land_v2_s202203220000_e202203240000_c202401191535.nc'
# glm_west_ff_loc = '../../../local-data/20220322-perils/GLM17_first-flash-data-land_v2_s202203220000_e202203240000_c202401191540.nc'
# lma_flash_loc = '../../../local-data/20220322-perils/perils-LMA_RAW-flash_202203220000_c202401191517_source-min-10.csv'
# lma_event_loc = '../../../local-data/20220322-perils/perils-LMA_RAW-event_202203220000_c202401191517_source-min-10.csv'
# lma_station_loc = '../../../local-data/20220322-perils/perils-LMA-20220322.csv'
# eni_loc = '../../../local-data/20220322-perils/eni_flash_flash20220322.csv'

# out_data_loc = ''
# out_plot_loc = ''

# #Getting the time variables established
# start_time = np.datetime64(stime_str)
# end_time = np.datetime64(etime_str)


# In[5]:


def time_space_cutdown(lats, lons, times, start_time, end_time, extent):
    '''
    Takes in the flash latitudes and longitudes determines which ones are within the domain
    PARAMS:
        lats: array of flash latitudes (floats)
        lons: array of flash longitudes (floats)
        times: list of times (list of datetimes)
        start_time: start time (datetime)
        end_time: end time (datetime)
        extent: bounds of the search spatially (list of floats)
    RETURNS:
        locs: Indicies of the flashes within those bounds
    '''
    
    #Getting the bounds into something more readable...
    lat_max = extent[3]
    lat_min = extent[2]
    lon_max = extent[1]
    lon_min = extent[0]
    
    #Getting the indicies of the flashes
    locs = np.where((lats<=lat_max)&(lats>=lat_min)&(lons<=lon_max)&(lons>=lon_min)&
            (times<=end_time)&(times>=start_time))[0]
    
    return locs

def space_cutdown(lats, lons, extent):
    '''
    Takes in the flash latitudes and longitudes determines which ones are within the domain
    PARAMS:
        lats: array of flash latitudes (floats)
        lons: array of flash longitudes (floats)
        extent: bounds of the search spatially (list of floats)
    RETURNS:
        locs: Indicies of the flashes within those bounds
    '''
    
    #Getting the bounds into something more readable...
    lat_max = extent[3]
    lat_min = extent[2]
    lon_max = extent[1]
    lon_min = extent[0]
    
    #Getting the indicies of the flashes
    locs = np.where((lats<=lat_max)&(lats>=lat_min)&(lons<=lon_max)&(lons>=lon_min))[0]
    
    return locs


# # GLM First-Flash Processing

# In[6]:


#This function takes in a file start time and the first/last event times to create a list of datetime objects
def GLM_times(file_times, times):
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

    return np.array(flash_datetime)


# In[7]:


def get_group_event_locs(dset, floc):
    f_id = dset.variables['flash_id'][floc]
    f_file = dset.variables['flash_parent_file'][floc]
    g_locs = np.where((dset.variables['group_parent_flash_id']==f_id)&(dset.variables['group_parent_file']==f_file))[0]
    g_ids = np.unique(dset.variables['group_id'][g_locs])#Getting all of the group ids within a flash
    e_locs = np.array([],dtype=int)
    
    #Need to loop through each group within the flash to find the events
    for g_id in g_ids:
        cur_event_loc = np.where((dset.variables['event_parent_group_id'][:]==g_id)&(dset.variables['event_parent_file'][:]==f_file))[0]
        e_locs = np.append(e_locs,cur_event_loc)  
    return g_locs, e_locs


# In[8]:


def glm_df_creator(dset, f_locs, g_locs, e_locs):
    flash_dict = {
        'flash_lat': dset.variables['flash_lat'][f_locs],
        'flash_lon': dset.variables['flash_lon'][f_locs],
        'flash_time_start': GLM_times(dset.variables['flash_parent_file'][f_locs], dset.variables['flash_time_offset_of_first_event'][f_locs]),
        'flash_time_end': GLM_times(dset.variables['flash_parent_file'][f_locs], dset.variables['flash_time_offset_of_last_event'][f_locs]),
        'flash_area': dset.variables['flash_area'][f_locs],
        'flash_energy': dset.variables['flash_energy'][f_locs],
        'flash_quality': dset.variables['flash_quality_flag'][f_locs],
        'flash_id': dset.variables['flash_id'][f_locs]
    }
    
    group_dict = {
        'group_lat': dset.variables['group_lat'][g_locs],
        'group_lon': dset.variables['group_lon'][g_locs],
        'group_time': GLM_times(dset.variables['group_parent_file'][g_locs], dset.variables['group_time_offset'][g_locs]),
        'group_area': dset.variables['group_area'][g_locs],
        'group_energy': dset.variables['group_energy'][g_locs],
        'group_id': dset.variables['group_id'][g_locs],
        'group_quality': dset.variables['group_quality_flag'][g_locs]
    }
    
    event_dict = {
        'event_lat': dset.variables['event_lat'][e_locs],
        'event_lon': dset.variables['event_lon'][e_locs],
        'event_energy': dset.variables['event_energy'][e_locs],
        'event_time': GLM_times(dset.variables['event_parent_file'][e_locs], dset.variables['event_time_offset'][e_locs])
    }
    
    flash_df = pd.DataFrame(data=flash_dict)
    group_df = pd.DataFrame(data=group_dict)
    event_df = pd.DataFrame(data=event_dict)
    
    return flash_df, group_df, event_df


# In[9]:


#Loading in the first flash datasets from goes east/west glms
glm_eff = nc.Dataset(glm_east_ff_loc, 'r')
glm_wff = nc.Dataset(glm_west_ff_loc, 'r')

#Getting all the times
glm_e_time = GLM_times(glm_eff.variables['flash_parent_file'][:], glm_eff.variables['flash_time_offset_of_first_event'][:])
glm_w_time = GLM_times(glm_wff.variables['flash_parent_file'][:], glm_wff.variables['flash_time_offset_of_first_event'][:])

#Getting the e/w lats/lons
glm_e_lat = glm_eff.variables['flash_lat'][:]
glm_e_lon = glm_eff.variables['flash_lon'][:]
glm_w_lat = glm_wff.variables['flash_lat'][:]
glm_w_lon = glm_wff.variables['flash_lon'][:]

#Cutting down to the approporiate indicies of the first flash events
glm_e_flocs = time_space_cutdown(glm_e_lat, glm_e_lon, glm_e_time, np.datetime64(start_time), np.datetime64(end_time), search_bounds)
glm_w_flocs = time_space_cutdown(glm_w_lat, glm_w_lon, glm_w_time, np.datetime64(start_time), np.datetime64(end_time), search_bounds)

#Getting the indicies of the groups and events
glm_e_glocs, glm_e_elocs = get_group_event_locs(glm_eff, glm_e_flocs)
glm_w_glocs, glm_w_elocs = get_group_event_locs(glm_wff, glm_w_flocs)

#Using all of the indicies to get the data from the selected flashes, and putting them into dataframes
glm_e_flash_df, glm_e_group_df, glm_e_event_df = glm_df_creator(glm_eff, glm_e_flocs, glm_e_glocs, glm_e_elocs)
glm_w_flash_df, glm_w_group_df, glm_w_event_df = glm_df_creator(glm_wff, glm_w_flocs, glm_w_glocs, glm_w_elocs)

print ('GLM First-Flashes Processed')


# # ENI Flash Processing

# In[10]:


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


# In[11]:


def flash_classification_tool(pk_current, ic_ht):
    '''
    Classifying ENI data by its peak current and ic height
    PARAMS:
        pk_current: Array of peak currents (float)
        ic_ht: Array of IC heights (float)
    RETURNS:
        fl_type: Array of classifications [-IC, +IC, -CG, +CG] (str)
    '''

    fl_type = np.empty(len(pk_current),object)
    
    fl_type[np.where((pk_current<0)&(ic_ht>0))[0]] = '-IC'
    fl_type[np.where((pk_current>0)&(ic_ht>0))[0]] = '+IC'
    fl_type[np.where((pk_current<0)&(ic_ht==0))[0]] = '-CG'
    fl_type[np.where((pk_current>0)&(ic_ht==0))[0]] = '+CG'
    
    return fl_type



# In[12]:


#Loading the ENI data
eni_dset = pd.read_csv(eni_loc)

#Getting the spatial and temporal info
eni_lat = eni_dset['latitude'].values
eni_lon = eni_dset['longitude'].values

eni_space_locs = space_cutdown(eni_lat, eni_lon, search_bounds)
eni_pre_cut = eni_dset.iloc[eni_space_locs,:]

eni_time = time_str_converter(eni_pre_cut['timestamp'].values)
eni_lat = eni_pre_cut['latitude'].values
eni_lon = eni_pre_cut['longitude'].values



#Finding the data that sit within the bounds
eni_locs = time_space_cutdown(eni_lat, eni_lon, eni_time, np.datetime64(start_time), np.datetime64(end_time), search_bounds)

#Cutting down the dataset
eni_cut = eni_pre_cut.iloc[eni_locs,:]

#Getting the flash type
eni_cut['type'] = flash_classification_tool(eni_cut['peakcurrent'].values, eni_cut['icheight'].values)
print ('ENI Flashes Processed')


# # LMA Flash Processing

# In[13]:


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


# In[14]:


def lma_event_cutter(flash_df, event_df):
    '''
    Find the events associated with the selected LMA flashes
    INPUTS:
        flash_df: DataFrame of flash-sorted LMA flashes
        event_df: DataFrame of flash-sorted LMA events
    RETURNS:
        event_out_df: DataFrame of flash-sorted LMA events that match the flashes
    '''
    #Getting the flash ids and files
    flash_ids = flash_df['flash_id'].values
    flash_files = flash_df['file_time'].values
    
    #Creating an empty DataFrame to fill
    event_out_df = pd.DataFrame()
    
    #Looping through all combinations of flash ids and file times,
    #and lining them up with those in events to pull them together
    for i in range(len(flash_ids)):
        event_out_df_new = event_df.loc[(event_df['flash_id']==flash_ids[i])&(event_df['file_time']==flash_files[i])]
        event_out_df = pd.concat((event_out_df,event_out_df_new))
    
    return event_out_df


# In[15]:


#Loading the the flashes and events
lma_f_df = pd.read_csv(lma_flash_loc)
lma_e_df = pd.read_csv(lma_event_loc)

#Getting the flash lat/lon/time
lma_f_lat = lma_f_df['ctr_lat'].values
lma_f_lon = lma_f_df['ctr_lon'].values
lma_f_time = LMA_times_postprocess(lma_f_df['file_time'],lma_f_df['start'])

#Getting the indicies of the selected flashes
lma_locs = time_space_cutdown(lma_f_lat, lma_f_lon, lma_f_time, np.datetime64(start_time), np.datetime64(end_time), search_bounds)

#Cutting down the initial dataframe of flashes
lma_fcut_df = lma_f_df.iloc[lma_locs,:]
#Cutting down the event dataframe too
lma_ecut_df = lma_event_cutter(lma_fcut_df,lma_e_df)

print ('LMA Flashes Processed')


# In[16]:


#Getting the timese and bounds needed to plot
lma_e_time = LMA_times_postprocess(lma_ecut_df['file_time'].values,lma_ecut_df['time'].values)
lat_max = search_bounds[3]
lat_min = search_bounds[2]
lon_max = search_bounds[1]
lon_min = search_bounds[0]
lma_stations = pd.read_csv(lma_station_loc)

#Setting the tick marks and their strings
tick_marks = pd.date_range(start=start_time, end=end_time, freq='100ms')
tick_mark_str = [i.strftime('%S.%f')[:-5] for i in tick_marks]


# # Lets make a plot!

# In[ ]:


#Establishing the save string and the file location
plot_save_str = out_plot_loc+'/'+case+'/'

if not os.path.exists(plot_save_str):
    os.makedirs(plot_save_str)


# In[17]:


fig = plt.figure(constrained_layout=True, figsize=(15,20))
fig.patch.set_facecolor('silver')
gs = fig.add_gridspec(nrows=16,ncols=13)
fig.suptitle(case_name + ' Case ' + fl_num + ', ' + str(start_time)+' - '+str(end_time), fontsize=16)
dot_size = 5
glm_tri = 40
glm_sq = 100

#Plot 1, The time-altitude LMA sources in the top panel of the figure
ax1 = fig.add_subplot(gs[0:2,0:13])
ax1.set_facecolor('black')

ax1.scatter(x=lma_e_time, y=lma_ecut_df['alt']/1000, c=lma_e_time, cmap=plt.cm.rainbow, s=dot_size, label='LMA Sources')
ax1.scatter(x=eni_cut.loc[eni_cut['type']=='+CG','timestamp'].values, y=np.ones(eni_cut.loc[eni_cut['type']=='+CG'].shape[0])*1, color='green', s=glm_sq, marker='^', zorder=10, edgecolors='white')
ax1.scatter(x=eni_cut.loc[eni_cut['type']=='-CG','timestamp'].values, y=np.ones(eni_cut.loc[eni_cut['type']=='-CG'].shape[0])*1, color='green', s=glm_sq, marker='v', zorder=10, edgecolors='white')
ax1.scatter(x=eni_cut.loc[eni_cut['type']=='+IC','timestamp'].values, y=np.ones(eni_cut.loc[eni_cut['type']=='+IC'].shape[0])*19, color='yellow', s=glm_sq, marker='^', zorder=10, edgecolors='black')
ax1.scatter(x=eni_cut.loc[eni_cut['type']=='-IC','timestamp'].values, y=np.ones(eni_cut.loc[eni_cut['type']=='-IC'].shape[0])*19, color='yellow', s=glm_sq, marker='v', zorder=10, edgecolors='black')
#ax1.legend(loc='lower left')
ax1.set_xlim(start_time, end_time)
ax1.set_ylim(0,20)
ax1.set_ylabel('Altitude (km)')
ax1.grid(visible=True, axis='y',color='gray',linewidth=1,alpha=0.5)
ax1.set_xticks(ticks=tick_marks, labels=tick_mark_str, fontsize=8)



#Plot 2, Time-Energy plot for GLM groups and LMA sources
ax2 = fig.add_subplot(gs[2:4,0:13])
ax2.set_facecolor('black')

ax2.scatter(x=lma_e_time, y=lma_ecut_df['power'], c=lma_e_time, cmap=plt.cm.rainbow, s=dot_size, label='LMA Sources')
ax2.set_xlim(start_time, end_time)
ax2.set_ylim(-20,50)
ax2.set_ylabel('Source Energy (dBW)')
ax2.grid(visible=True, axis='y',color='gray',linewidth=1,alpha=0.5)
ax2.set_xticks(ticks=tick_marks, labels=tick_mark_str, fontsize=8)

ax2_tw = ax2.twinx()
ax2_tw.scatter(x=glm_e_group_df['group_time'], y=glm_e_group_df['group_energy']*1e15, color='r', s=glm_tri, marker='^', label='GOES-East GLM Groups')
ax2_tw.scatter(x=glm_w_group_df['group_time'], y=glm_w_group_df['group_energy']*1e15, color='b', s=glm_tri, marker='^', label='GOES-West GLM Groups')
ax2_tw.set_ylim(1,10000)
ax2_tw.set_yscale('log')
ax2_tw.set_ylabel('Group Energy (fJ)')

#ax2.legend(loc='upper left')
ax2_tw.legend(loc='upper right')

#Plot 3, Longitude-Altitude plot of the LMA sources
ax3 = fig.add_subplot(gs[4:6,0:10])
ax3.set_facecolor('black')
ax3.scatter(x=lma_ecut_df['lon'], y=lma_ecut_df['alt']/1000, c=lma_e_time, cmap=plt.cm.rainbow, s=dot_size, zorder=0)

ax3.set_xlim(lon_min, lon_max)
ax3.set_ylim(0,20)
ax3.set_ylabel('Altitude (km)')
ax3.grid(visible=True, axis='y',color='gray',linewidth=1, alpha=0.5)
ax3.grid(visible=True, axis='x',color='gray',linestyle='--',linewidth=2, alpha=0.5)

ax3_tw = ax3.twinx()
ax3_tw.hist(x=glm_e_group_df['group_lon'], bins=np.arange(lon_min,lon_max+0.05,0.05), zorder=1, alpha=0.3, color='r')
ax3_tw.hist(x=glm_w_group_df['group_lon'], bins=np.arange(lon_min,lon_max+0.05,0.05), zorder=1, alpha=0.3, color='b')
ax3_tw.set_ylabel('GLM Group Density')
#ax3_tw.set_ylim(0,80)

#Plot 4, Overview Map of Plot Area
ax4 = fig.add_subplot(gs[4:6,10:13], projection=ccrs.PlateCarree())
ax4.set_facecolor('black')
ax4.scatter(x=lma_stations['lon'], y=lma_stations['lat'], color='white', label='LMA Stations', marker='^', s=dot_size)
ax4.scatter(x=glm_e_flash_df['flash_lon'], y=glm_e_flash_df['flash_lat'], color='r', s=dot_size)
ax4.scatter(x=glm_w_flash_df['flash_lon'], y=glm_w_flash_df['flash_lat'], color='b', s=dot_size)
ax4.set_xlim(plot_bounds[0], plot_bounds[1])
ax4.set_ylim(plot_bounds[2], plot_bounds[3])
ax4.add_feature(USCOUNTIES, edgecolor='g', zorder=0)
ax4.add_feature(cfeature.STATES, edgecolor ='r',linewidth=1.5, zorder=0)
#ax4.legend(loc='lower left', fontsize=8)

#Plot 5, Lat Lon of LMA, GLM, and ENI data
ax5 = fig.add_subplot(gs[6:16,0:10], projection=ccrs.PlateCarree())
ax5.set_facecolor('black')

ax5.scatter(x=glm_e_event_df['event_lon'], y=glm_e_event_df['event_lat'], color='r', s=glm_sq, marker='s', label='GOES-East GLM Events')
ax5.scatter(x=glm_w_event_df['event_lon'], y=glm_w_event_df['event_lat'], color='b', s=glm_sq, marker='s', label='GOES-West GLM Events')
ax5.scatter(x=lma_ecut_df['lon'], y=lma_ecut_df['lat'], c=lma_e_time, cmap=plt.cm.rainbow, s=dot_size, zorder=10, label='LMA Sources')
ax5.scatter(x=lma_stations['lon'], y=lma_stations['lat'], color='white', label='LMA Stations', marker='^', s=glm_tri, zorder=0)
ax5.scatter(x=eni_cut.loc[eni_cut['type']=='+CG','longitude'].values, y=eni_cut.loc[eni_cut['type']=='+CG','latitude'].values, color='green', s=glm_sq, marker='^', label='ENI +CG', zorder=10, edgecolors='white')
ax5.scatter(x=eni_cut.loc[eni_cut['type']=='-CG','longitude'].values, y=eni_cut.loc[eni_cut['type']=='-CG','latitude'].values, color='green', s=glm_sq, marker='v', label='ENI -CG', zorder=10, edgecolors='white')
ax5.scatter(x=eni_cut.loc[eni_cut['type']=='+IC','longitude'].values, y=eni_cut.loc[eni_cut['type']=='+IC','latitude'].values, color='yellow', s=glm_sq, marker='^', label='ENI +IC', zorder=10, edgecolors='black')
ax5.scatter(x=eni_cut.loc[eni_cut['type']=='-IC','longitude'].values, y=eni_cut.loc[eni_cut['type']=='-IC','latitude'].values, color='yellow', s=glm_sq, marker='v', label='ENI -IC', zorder=10, edgecolors='black')


ax5.set_xlim(lon_min, lon_max)
ax5.set_ylim(lat_min, lat_max)
ax5.legend(loc='upper right')
ax5.add_feature(USCOUNTIES, edgecolor='g', zorder=0)
ax5.add_feature(cfeature.STATES, edgecolor ='r',linewidth=1.5, zorder=0)
gl5 = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl5.top_labels=False
gl5.right_labels=False

#Plot 6, latitude-altitude plot of the LMA events
ax6 = fig.add_subplot(gs[6:16,10:13])
ax6.set_facecolor('black')
ax6.scatter(y=lma_ecut_df['lat'], x=lma_ecut_df['alt']/1000, c=lma_e_time, cmap=plt.cm.rainbow, s=dot_size)
ax6.set_ylim(lat_min, lat_max)
ax6.set_xlim(0,20)
ax6.set_xlabel('Altitude (km)')

ax6_tw = ax6.twiny()
ax6_tw.hist(x=glm_e_group_df['group_lat'], bins=np.arange(lat_min,lat_max+0.05,0.05), zorder=1, alpha=0.3, color='r', orientation='horizontal')
ax6_tw.hist(x=glm_w_group_df['group_lat'], bins=np.arange(lat_min,lat_max+0.05,0.05), zorder=1, alpha=0.3, color='b', orientation='horizontal')
ax6_tw.set_xlabel('GLM Group Density')
#ax6_tw.set_xlim(0,80)

ax6.grid(visible=True, axis='y',color='gray',linestyle='--',linewidth=2, alpha=0.5)
ax6.grid(visible=True, axis='x',color='gray',linewidth=1, alpha=0.5)
ax6.tick_params(left = False, right = True , labelleft = False, labelright=True) 

#plt.savefig('ithinkthiswilldo.png') #DEVMODE
plt.savefig(plot_save_str+case+'-'+fl_num+'summary.png')


# # Saving the datasets

# In[ ]:


#Establishing the save string and the file location
save_str = out_data_loc+'/'+case+'-'+fl_num+'/'

if not os.path.exists(save_str):
    os.makedirs(save_str)


# In[19]:


#GLM East
glm_e_flash_df.to_csv(save_str+case+'-'+fl_num+'-GLMEastFlashes.csv')
glm_e_group_df.to_csv(save_str+case+'-'+fl_num+'-GLMEastGroups.csv')
glm_e_event_df.to_csv(save_str+case+'-'+fl_num+'-GLMEastEvents.csv')
#GLM West
glm_w_flash_df.to_csv(save_str+case+'-'+fl_num+'-GLMWestFlashes.csv')
glm_w_group_df.to_csv(save_str+case+'-'+fl_num+'-GLMWestGroups.csv')
glm_w_event_df.to_csv(save_str+case+'-'+fl_num+'-GLMWestEvents.csv')
#ENI
eni_cut.to_csv(save_str+case+'-'+fl_num+'-ENIFlashes.csv')
#LMA
lma_ecut_df.to_csv(save_str+case+'-'+fl_num+'-LMAEvents.csv')
lma_fcut_df.to_csv(save_str+case+'-'+fl_num+'-LMAFlashes.csv')

