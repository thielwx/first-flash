#!/usr/bin/env python
# coding: utf-8

#  A generizable script that can be used for any case study to animate first flash events created from 'flash-display-summary.py'

# In[78]:


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


# In[79]:


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
    flash_datetime = [flash_file_datetime[i] + flash_timedelta[i] for i in range(len(flash_timedelta))]

    return (flash_datetime)


# In[80]:


def time_cutdown(df, start_time, end_time):
    df_cut = df.loc[(df['ptime']>=start_time)&(df['ptime']<=end_time)]
    return df_cut


# In[81]:


args = sys.argv
# args = ['','20220322-perils', 'f1'] #Devmode
# #
case = args[1]
fl_num = args[2]


# In[115]:


#Importing the appropriate yaml file
with open('flash-display-summary-settings.yaml', 'r') as f:
    sfile = yaml.safe_load(f)
    
#Event-specific data (spatial and temporal)
stime_str = sfile[case][fl_num]['start_time']
etime_str = sfile[case][fl_num]['end_time']
search_bounds = sfile[case][fl_num]['search_bounds']
plot_bounds = sfile[case][fl_num]['plot_bounds']
case_name = sfile[case]['case_name']
data_loc = sfile['out_data_loc']
plot_loc = sfile['out_plot_loc']
lma_station_loc = sfile[case]['lma_station_loc']
hist_max = sfile[case][fl_num]['hist_max']

data_str = data_loc+'/'+case+'-'+fl_num+'/'


#ALL IN DEVMODE
# stime_str = '2022-03-22T20:06:44.2'
# etime_str = '2022-03-22T20:06:47.1'
# search_bounds = [-89.0,-87.5,32.7,34.2]
# plot_bounds = [-90.0,-86,31,35]
# case_name = 'PERiLS'
# data_loc = ''
# plot_loc = ''
# lma_station_loc = '../../../local-data/perils-LMA-20220322-stations.csv'

# data_str = '' #DEVMODE


# In[116]:


#Getting the time variables established
start_time = np.datetime64(stime_str)
end_time = np.datetime64(etime_str)

time_list = pd.date_range(start=start_time, end=end_time, freq='5ms')
dt = np.timedelta64(5,'ms')

#Getting the times and bounds needed to plot
lat_max = search_bounds[3]
lat_min = search_bounds[2]
lon_max = search_bounds[1]
lon_min = search_bounds[0]
lma_stations = pd.read_csv(lma_station_loc)
lma_stations_active = lma_stations.loc[(lma_stations['active']=='A')|(lma_stations['active']=='NAS')]

#Setting the tick marks and their strings
tick_marks = pd.date_range(start=start_time, end=end_time, freq='100ms')
tick_mark_str = [i.strftime('%S.%f')[:-5] for i in tick_marks]

#Creating where the files will be saved to
plot_save_str = plot_loc + case + '/' + case + '-' + fl_num + '-v1/'
if not os.path.exists(plot_save_str):
    os.makedirs(plot_save_str)


# In[84]:


#Reading in all of the flash characteristics as pandas dataframes
#GLM East fl(ashes) gr(oups) ev(ents)
ge_fl = pd.read_csv(data_str+case+'-'+fl_num+'-GLMEastFlashes.csv')
ge_gr = pd.read_csv(data_str+case+'-'+fl_num+'-GLMEastGroups.csv')
ge_ev = pd.read_csv(data_str+case+'-'+fl_num+'-GLMEastEvents.csv')
#GLM West fl(ashes) gr(oups) ev(ents)
gw_fl = pd.read_csv(data_str+case+'-'+fl_num+'-GLMWestFlashes.csv')
gw_gr = pd.read_csv(data_str+case+'-'+fl_num+'-GLMWestGroups.csv')
gw_ev = pd.read_csv(data_str+case+'-'+fl_num+'-GLMWestEvents.csv')

#Earth Networks Flashes
eni = pd.read_csv(data_str+case+'-'+fl_num+'-ENIFlashes.csv')

#LMA f(lashes) and e(vents)
lma_e = pd.read_csv(data_str+case+'-'+fl_num+'-LMAEvents.csv')
lma_f = pd.read_csv(data_str+case+'-'+fl_num+'-LMAFlashes.csv')


# In[86]:


#Creating a new time variable 'ptime' that is used for plotting (format: np.datetime64)
ge_fl['ptime'] = [np.datetime64(ge_fl['flash_time_start'].values[i]) for i in range(ge_fl.shape[0])]
ge_gr['ptime'] = [np.datetime64(ge_gr['group_time'].values[i]) for i in range(ge_gr.shape[0])]
ge_ev['ptime'] = [np.datetime64(ge_ev['event_time'].values[i]) for i in range(ge_ev.shape[0])]

gw_fl['ptime'] = [np.datetime64(gw_fl['flash_time_start'].values[i]) for i in range(gw_fl.shape[0])]
gw_gr['ptime'] = [np.datetime64(gw_gr['group_time'].values[i]) for i in range(gw_gr.shape[0])]
gw_ev['ptime'] = [np.datetime64(gw_ev['event_time'].values[i]) for i in range(gw_ev.shape[0])]

eni['ptime'] = [np.datetime64(eni['timestamp'].values[i]) for i in range(eni.shape[0])]

lma_e['ptime'] = LMA_times_postprocess(lma_e['file_time'], lma_e['time'])
lma_f['ptime'] = LMA_times_postprocess(lma_f['file_time'], lma_f['start'])


# In[119]:


#Looping through the times every 5ms
for i in range(len(time_list)):
    cur_time = time_list[i] #Getting the current time
    
    #Cutting down the data by time
    ge_gr_cut = time_cutdown(ge_gr,start_time,cur_time)
    ge_ev_cut = time_cutdown(ge_ev,start_time,cur_time)
    gw_gr_cut = time_cutdown(gw_gr,start_time,cur_time)
    gw_ev_cut = time_cutdown(gw_ev,start_time,cur_time)
    eni_cut = time_cutdown(eni, start_time,cur_time)
    lma_e_cut = time_cutdown(lma_e,start_time,cur_time)
    
    #The most recent activity to also plot
    ge_ev_new = time_cutdown(ge_ev,cur_time-dt,cur_time)
    gw_ev_new = time_cutdown(gw_ev,cur_time-dt,cur_time)
    eni_new = time_cutdown(eni,cur_time-dt,cur_time)
    lma_e_new = time_cutdown(lma_e,cur_time-dt,cur_time)
    
    #PLOT TIME!!!
    fig = plt.figure(constrained_layout=True, figsize=(15,20))
    fig.patch.set_facecolor('silver')
    gs = fig.add_gridspec(nrows=16,ncols=13)
    fig.suptitle(case_name + ' Case ' + fl_num + ', ' + start_time.strftime('%F %T.%f')[:-3] + ' - ' + cur_time.strftime('%F %T.%f')[:-3], fontsize=16)
    dot_size = 5
    glm_tri = 40
    glm_sq = 100
    lma_color = plt.cm.spring_r

    #Plot 1, The time-altitude LMA sources in the top panel of the figure
    ax1 = fig.add_subplot(gs[0:2,0:13])
    ax1.set_facecolor('black')

    ax1.scatter(x=lma_e_cut['ptime'], y=lma_e_cut['alt']/1000, c=lma_e_cut['ptime'], cmap=lma_color, s=dot_size, label='LMA Sources')
    ax1.scatter(x=lma_e_new['ptime'], y=lma_e_new['alt']/1000, c='white', s=dot_size*2)
    
    ax1.scatter(x=eni_cut.loc[eni_cut['type']=='+CG','ptime'].values, y=np.ones(eni_cut.loc[eni_cut['type']=='+CG'].shape[0])*1, color='green', s=glm_sq, marker='^', zorder=10, edgecolors='white')
    ax1.scatter(x=eni_cut.loc[eni_cut['type']=='-CG','ptime'].values, y=np.ones(eni_cut.loc[eni_cut['type']=='-CG'].shape[0])*1, color='green', s=glm_sq, marker='v', zorder=10, edgecolors='white')
    ax1.scatter(x=eni_cut.loc[eni_cut['type']=='+IC','ptime'].values, y=np.ones(eni_cut.loc[eni_cut['type']=='+IC'].shape[0])*19, color='yellow', s=glm_sq, marker='^', zorder=10, edgecolors='black')
    ax1.scatter(x=eni_cut.loc[eni_cut['type']=='-IC','ptime'].values, y=np.ones(eni_cut.loc[eni_cut['type']=='-IC'].shape[0])*19, color='yellow', s=glm_sq, marker='v', zorder=10, edgecolors='black')
    
    ax1.axvline(x=cur_time, color='white', alpha=0.5, zorder=0, linewidth=1)
    
    #ax1.legend(loc='lower left')
    ax1.set_xlim(start_time, end_time)
    ax1.set_ylim(0,20)
    ax1.set_ylabel('Altitude (km)')
    ax1.grid(visible=True, axis='y',color='gray',linewidth=1,alpha=0.5)
    ax1.set_xticks(ticks=tick_marks, labels=tick_mark_str, fontsize=8)



    #Plot 2, Time-Energy plot for GLM groups and LMA sources
    ax2 = fig.add_subplot(gs[2:4,0:13])
    ax2.set_facecolor('black')

    ax2.scatter(x=lma_e_cut['ptime'], y=lma_e_cut['power'], c=lma_e_cut['ptime'], cmap=lma_color, s=dot_size, label='LMA Sources')
    ax2.scatter(x=lma_e_new['ptime'], y=lma_e_new['power'], c='white', s=dot_size*2)
    
    ax2.axvline(x=cur_time, color='white', alpha=0.5, zorder=0, linewidth=1)
    ax2.set_xlim(start_time, end_time)
    ax2.set_ylim(-20,50)
    ax2.set_ylabel('Source Energy (dBW)')
    ax2.grid(visible=True, axis='y',color='gray',linewidth=1,alpha=0.5)
    ax2.set_xticks(ticks=tick_marks, labels=tick_mark_str, fontsize=8)

    ax2_tw = ax2.twinx()
    ax2_tw.scatter(x=ge_gr_cut['ptime'], y=ge_gr_cut['group_energy']*1e15, color='r', s=glm_tri, marker='^', label='GOES-East GLM Groups')
    ax2_tw.scatter(x=gw_gr_cut['ptime'], y=gw_gr_cut['group_energy']*1e15, color='b', s=glm_tri, marker='^', label='GOES-West GLM Groups')
    ax2_tw.set_ylim(1,10000)
    ax2_tw.set_yscale('log')
    ax2_tw.set_ylabel('Group Energy (fJ)')

    #ax2.legend(loc='upper left')
    ax2_tw.legend(loc='upper right')

    #Plot 3, Longitude-Altitude plot of the LMA sources
    ax3 = fig.add_subplot(gs[4:6,0:10])
    ax3.set_facecolor('black')
    ax3.scatter(x=lma_e_cut['lon'], y=lma_e_cut['alt']/1000, c=lma_e_cut['ptime'], cmap=lma_color, s=dot_size, zorder=0)
    ax3.scatter(x=lma_e_new['lon'], y=lma_e_new['alt']/1000, c='white', s=dot_size*2, zorder=0)

    ax3.set_xlim(lon_min, lon_max)
    ax3.set_ylim(0,20)
    ax3.set_ylabel('Altitude (km)')
    ax3.grid(visible=True, axis='y',color='gray',linewidth=1, alpha=0.5)
    ax3.grid(visible=True, axis='x',color='gray',linestyle='--',linewidth=2, alpha=0.5)

    ax3_tw = ax3.twinx()
    ax3_tw.hist(x=ge_gr_cut['group_lon'], bins=np.arange(lon_min,lon_max+0.05,0.05), zorder=1, alpha=0.3, color='r')
    ax3_tw.hist(x=gw_gr_cut['group_lon'], bins=np.arange(lon_min,lon_max+0.05,0.05), zorder=1, alpha=0.3, color='b')
    ax3_tw.set_ylabel('GLM Group Density')
    ax3_tw.set_ylim(0,hist_max)

    #Plot 4, Overview Map of Plot Area
    ax4 = fig.add_subplot(gs[4:6,10:13], projection=ccrs.PlateCarree())
    ax4.set_facecolor('black')
    ax4.scatter(x=lma_stations_active['lon'], y=lma_stations_active['lat'], color='white', label='LMA Stations', marker='^', s=dot_size)
    ax4.scatter(x=ge_fl['flash_lon'], y=ge_fl['flash_lat'], color='r', s=dot_size*5)
    ax4.scatter(x=gw_fl['flash_lon'], y=gw_fl['flash_lat'], color='b', s=dot_size*5)
    ax4.set_xlim(plot_bounds[0], plot_bounds[1])
    ax4.set_ylim(plot_bounds[2], plot_bounds[3])
    ax4.add_feature(USCOUNTIES, edgecolor='g', zorder=0)
    ax4.add_feature(cfeature.STATES, edgecolor ='r',linewidth=1.5, zorder=0)
    #ax4.legend(loc='lower left', fontsize=8)

    #Plot 5, Lat Lon of LMA, GLM, and ENI data
    ax5 = fig.add_subplot(gs[6:16,0:10], projection=ccrs.PlateCarree())
    ax5.set_facecolor('black')

    ax5.scatter(x=ge_ev_cut['event_lon'], y=ge_ev_cut['event_lat'], color='r', s=glm_sq, marker='s', label='GOES-East GLM Events')
    ax5.scatter(x=gw_ev_cut['event_lon'], y=gw_ev_cut['event_lat'], color='b', s=glm_sq, marker='s', label='GOES-West GLM Events')
    ax5.scatter(x=ge_ev_new['event_lon'], y=ge_ev_new['event_lat'], color='r', s=glm_sq*2, marker='s', edgecolors='white', linewidth=3)
    ax5.scatter(x=gw_ev_new['event_lon'], y=gw_ev_new['event_lat'], color='b', s=glm_sq*2, marker='s', edgecolors='white', linewidth=3)
    
    ax5.scatter(x=lma_e_cut['lon'], y=lma_e_cut['lat'], c=lma_e_cut['ptime'], cmap=lma_color, s=dot_size, zorder=10, label='LMA Sources')
    ax5.scatter(x=lma_e_new['lon'], y=lma_e_new['lat'], c='white', s=dot_size*4, zorder=10)

    ax5.scatter(x=lma_stations_active['lon'], y=lma_stations_active['lat'], color='white', label='LMA Stations', marker='^', s=glm_tri, zorder=0)
    
    ax5.scatter(x=eni_cut.loc[eni_cut['type']=='+CG','longitude'].values, y=eni_cut.loc[eni_cut['type']=='+CG','latitude'].values, color='green', s=glm_sq, marker='^', label='ENI +CG', zorder=10, edgecolors='white')
    ax5.scatter(x=eni_cut.loc[eni_cut['type']=='-CG','longitude'].values, y=eni_cut.loc[eni_cut['type']=='-CG','latitude'].values, color='green', s=glm_sq, marker='v', label='ENI -CG', zorder=10, edgecolors='white')
    ax5.scatter(x=eni_cut.loc[eni_cut['type']=='+IC','longitude'].values, y=eni_cut.loc[eni_cut['type']=='+IC','latitude'].values, color='yellow', s=glm_sq, marker='^', label='ENI +IC', zorder=10, edgecolors='black')
    ax5.scatter(x=eni_cut.loc[eni_cut['type']=='-IC','longitude'].values, y=eni_cut.loc[eni_cut['type']=='-IC','latitude'].values, color='yellow', s=glm_sq, marker='v', label='ENI -IC', zorder=10, edgecolors='black')


    ax5.set_xlim(lon_min, lon_max)
    ax5.set_ylim(lat_min, lat_max)
    ax5.legend(loc='upper right')
    ax5.add_feature(USCOUNTIES, edgecolor='g', zorder=0)
    ax5.add_feature(cfeature.STATES, edgecolor ='r',linewidth=1.5, zorder=0)
    gl5 = ax5.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=2, color='gray', alpha=0.5, linestyle='--', zorder=0)
    gl5.top_labels=False
    gl5.right_labels=False

    #Plot 6, latitude-altitude plot of the LMA events
    ax6 = fig.add_subplot(gs[6:16,10:13])
    ax6.set_facecolor('black')
    ax6.scatter(y=lma_e_cut['lat'], x=lma_e_cut['alt']/1000, c=lma_e_cut['ptime'], cmap=lma_color, s=dot_size)
    ax6.scatter(y=lma_e_new['lat'], x=lma_e_new['alt']/1000, c='white', s=dot_size*2)
    ax6.set_ylim(lat_min, lat_max)
    ax6.set_xlim(0,20)
    ax6.set_xlabel('Altitude (km)')

    ax6_tw = ax6.twiny()
    ax6_tw.hist(x=ge_gr_cut['group_lat'], bins=np.arange(lat_min,lat_max+0.05,0.05), zorder=1, alpha=0.3, color='r', orientation='horizontal')
    ax6_tw.hist(x=gw_gr_cut['group_lat'], bins=np.arange(lat_min,lat_max+0.05,0.05), zorder=1, alpha=0.3, color='b', orientation='horizontal')
    ax6_tw.set_xlabel('GLM Group Density')
    ax6_tw.set_xlim(0,hist_max)

    ax6.grid(visible=True, axis='y',color='gray',linestyle='--',linewidth=2, alpha=0.5)
    ax6.grid(visible=True, axis='x',color='gray',linewidth=1, alpha=0.5)
    ax6.tick_params(left = False, right = True , labelleft = False, labelright=True) 

    plt.savefig(plot_save_str + case + '-' + cur_time.strftime('%Y%m%d-%H%M%S-%f') + '.png')

