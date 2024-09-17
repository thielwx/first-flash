#!/usr/bin/env python
# coding: utf-8

# A script that makes all the plots/animations necessary for the manual analysis processor
# Creating a 10 minute animation with two subplots (left and right)
# 
# Left: GLM16 all flashes (10 minutes), eni all flashes (30 minutes), GLM16 first flash (circular flash area), and ABI16.
# Right: LMA flashes (10 minutes), GLM16 first flash (circular flash area), and MRMS -10C dBZ

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gzip
import netCDF4 as nc
import yaml
from datetime import datetime
from datetime import timedelta
from glob import glob
from pyresample import SwathDefinition, kd_tree
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from metpy.plots import USCOUNTIES
import os


# In[11]:


dt_start = datetime.now()
#Importing the appropriate yaml file
with open('case-settings-manual-analysis.yaml', 'r') as f:
    sfile = yaml.safe_load(f)

cases = sfile['cases']

#For cutting down the data spatially
dxc = 0.5
dx = 1

#For setting our time intervals
dt10 = np.timedelta64(10,'m')
dt30 = np.timedelta64(30,'m')
dt50s = np.timedelta64(50, 's')


# In[4]:


def eni_puller(start_time_str, end_time_str):
    '''
    Pulls data from the current and next day since you have to load in multipe files (days) for a case
    PARAMS: 
        start_time_str: String of the start time from the settings file
        end_time_str: String of the end time from the settings file

    '''
    #Getting the basic data to use later
    data_loc = '/localdata/first-flash/data/ENI-base-stock/eni_flash_flash'

    #Loading in the data to a single dataframe
    if start_time_str[:8] != end_time_str[:8]:
        eni1 = pd.read_csv(data_loc+start_time_str[:8]+'.csv', index_col=0)
        eni2 = pd.read_csv(data_loc+end_time_str[:8]+'.csv', index_col=0)
        eni = pd.concat((eni1,eni2), axis=0)
    else:
        eni = pd.read_csv(data_loc+start_time_str[:8]+'.csv', index_col=0)

    return eni


    


# In[5]:


def abi_puller(t):
    #Getting the necessary ABI data
    CMI = [[-999]]
    x = [np.nan]
    y = [np.nan]
    extent = [np.nan]
    sat_lon = -75.0
    sat_h = 35786023.0
    geo_crs = ccrs.Geostationary(central_longitude=sat_lon,satellite_height=sat_h)
     
    
    
    #Getting the time strings needed to find the file
    

    dt_int = int(t.strftime('%M'))%5 #Using the mod operator to tell us how much to adjust the time
    if dt_int == 0:
        dt_int = 5
        
    adjusted_time = t - timedelta(minutes=dt_int-1)
    cur_date = adjusted_time.strftime('%Y%m%d')
    file_time_string = adjusted_time.strftime('%Y%j%H%M')
    
    cur_abi = 'ABI16-CMIPC13'
    file_loc = '/localdata/first-flash/data/'+cur_abi+'/'+cur_date+'/*s'+file_time_string+'*.nc'
    #print (file_loc)
    collected_files = glob(file_loc)
    
    if len(collected_files)>0:
        dset = nc.Dataset(collected_files[0],'r')
        CMI = dset.variables['CMI'][:]
        CMI[CMI>280] = np.nan
        #sat_lon = dset.variables['goes_imager_projection'].longitude_of_projection_origin
        #print ('Satellite Longitude from File:')        
        #print (sat_lon)
        sat_lon = -75.2
        sat_h = dset.variables['goes_imager_projection'].perspective_point_height
        geo_crs = ccrs.Geostationary(central_longitude=sat_lon,satellite_height=sat_h)
        
        #ABI cordinates
        x = dset.variables['x'][:] * sat_h
        y = dset.variables['y'][:] * sat_h
        extent = (np.min(x), np.max(x), np.min(y), np.max(y))
    else:
        print ('ERROR: ABI DATA MISSING')
        
    return CMI, x, y, extent, geo_crs


# In[13]:


def mrms_puller(t,cur_lat,cur_lon):
    cur_minute = t.strftime('%M')
    
    #Making sure we are only looking for even data files
    if int(cur_minute)%2 == 1:
        t_adj = t - timedelta(minutes=1)
        cur_time = t_adj.strftime('%H%M')
        cur_date = t_adj.strftime('%Y%m%d')
    else:
        cur_time = t.strftime('%H%M')
        cur_date = t.strftime('%Y%m%d')
        
    file_loc = '/raid/swat_archive/vmrms/CONUS/'+cur_date+'/multi/Reflectivity_-10C/00.50/'+cur_date+'-'+cur_time+'*.netcdf.gz'
    #print (file_loc)
    collected_files = glob(file_loc)
    
    if len(collected_files) > 0:
        with gzip.open(collected_files[0]) as gz:
            with nc.Dataset('dummy', mode='r', memory=gz.read()) as dset:
                #Extracting the data from the file                
                data = dset.variables['Reflectivity_-10C'][:]
              
                x_pix = dset.variables['pixel_x'][:] #Pixel locations (indicies) for LATITUDE
                y_pix = dset.variables['pixel_y'][:] #Pixel locations (indicies) for LONGITUDE

                u_lat = dset.Latitude #Upper-most latitude
                l_lon = dset.Longitude #Left-most longitude

                #Creating the arrays for the lat and lon coordinates
                y = dset.dimensions['Lat'].size #3500
                x = dset.dimensions['Lon'].size #7000
                lat = np.arange(u_lat, u_lat-(y*0.01),-0.01) #Going from upper to lower
                lon = np.arange(l_lon, l_lon+(x*0.01),0.01) #Going from left to right

                #Using the pixel indicides to get the pixel latitudes and longitudes
                lat_data = lat[x_pix] #Remember x_pixel represents LATITUDE
                lon_data = lon[y_pix] #Remember y_pixel represent LONGITUDE

                #Removing any false data and those outside the plotting bounds (less data to process)
                locs = np.where((data>0)&(lon_data>=cur_lon-dx)&(lon_data<=cur_lon+dx)&(lat_data>=cur_lat-dx)&(lat_data<=cur_lat+dx))[0]
                lon_data = lon_data[locs]
                lat_data = lat_data[locs]
                data = data[locs]
                
                #Creating the lon/lat grid
                lon_grid, lat_grid = np.meshgrid(lon, lat)

                #Defining the swaths of our listed and gridded data lat/lons
                MRMS_grid_swath = SwathDefinition(lons=lon_grid, lats=lat_grid)
                MRMS_point_swath = SwathDefinition(lons=lon_data, lats=lat_data)

                #Putting the data into a grid
                output = kd_tree.resample_nearest(source_geo_def=MRMS_point_swath,
                                        data=data,
                                        target_geo_def=MRMS_grid_swath,
                                        radius_of_influence=1e3)
                
                extent_mrms = [np.min(lon), np.max(lon), np.min(lat), np.max(lat)] 
                
    else:
        print ('ERROR: MRMS DATA MISSING')
        output = [[-999]]
        extent_mrms = [-999]
            
    return output, extent_mrms


# In[7]:


def save_string(cur, i, t, case):
    #fistart_flid = cur['fistart_flid']
    fistart_flid = cur.name
    time_str = t.strftime('%Y%m%d-%H%M%S')
    start_str = '/localdata/first-flash/figures/manual-analysis-v1/'+case+'/'+str(i).zfill(2)+'-'+fistart_flid +'/'
    save_str = start_str + time_str + '.png'

    # Uncomment when running on devlab4
    if not os.path.exists(start_str):
        os.makedirs(start_str)

    return save_str


# In[15]:


def plotter(cur, dx, i, case, g16, eni):
    cur_time = cur['time64']
    #print (cur_time)
    cur_lat = cur['lat']
    cur_lon = cur['lon']
    cur_area = cur['flash_area'] / 1e6 #Putting into km sq
    
    # Creating the time list. Every minute, along with every ten seconds in the final minute 
    # and the five seconds before and after the first flash event
    time_list_start = pd.date_range(start=cur_time-dt10, end=cur_time-np.timedelta64(1,'m'), freq='60s').tolist()
    time_list_mid = pd.date_range(start=cur_time-dt50s, end=cur_time-np.timedelta64(10,'s'), freq='10s').tolist()
    time_list_mid2 = pd.date_range(start=cur_time-np.timedelta64(5,'s'), end=cur_time+np.timedelta64(5,'s'), freq='5s').tolist()
    time_list_end = pd.date_range(start=cur_time, end=cur_time+np.timedelta64(5,'m'), freq='60s').to_list()[1:]
    time_list = time_list_start+time_list_mid+time_list_mid2+time_list_end
    
    #Caucluations for the approx area of the first flash event for continuity
    R = 6371.0087714 #Earths radius in km
    d = np.sqrt(cur_area/np.pi) #Search Radius
    r = (d/R)*(180/np.pi)
    
    tf_size = 5

    title_size = 16
    
    #Looping through the individual times
    for t in time_list[:]:
        #Step 1: Get the data from the dataframes
        g16_cut = g16.loc[(g16['lat']>=cur_lat-dx)&(g16['lat']<=cur_lat+dx)&
                         (g16['lon']>=cur_lon-dx)&(g16['lon']<=cur_lon+dx)&
                         (g16['time64']>=t-dt10)&(g16['time64']<=t)]
        eni_cut = eni.loc[(eni['latitude']>=cur_lat-dx)&(eni['latitude']<=cur_lat+dx)&
                         (eni['longitude']>=cur_lon-dx)&(eni['longitude']<=cur_lon+dx)&
                         (eni['time64']>=t-dt10)&(eni['time64']<=t)]
        
        #Formatting the countdown time so it's more readable
        dt_seconds = (t-cur_time).seconds
        if t < cur_time:
            dt_seconds_countdown = 86400 - dt_seconds
            minutes, seconds = divmod(dt_seconds_countdown, 60)
            countdown_str='-{:02}:{:02}'.format(int(minutes), int(seconds))
        elif t==cur_time:
            countdown_str='+00:00'
        else:
            minutes, seconds = divmod(dt_seconds, 60)
            countdown_str='+{:02}:{:02}'.format(int(minutes), int(seconds))
            
        #=====================================================
        #Step 2: Getting the data from the MRMS and ABI
        CMI, x, y, abi_extent, geo_crs = abi_puller(t)
        refl_10, mrms_extent = mrms_puller(t, cur_lat, cur_lon)
        save_str = save_string(cur, i, t, case)
        
        #=====================================================
        #Step 3: Making the plots
        plt_car_crs = ccrs.PlateCarree()
        plot_extent = [cur_lon-dxc, cur_lon+dxc, cur_lat-dxc, cur_lat+dxc]
        
        fig = plt.figure(constrained_layout=True, figsize=(16,7))
        fig.patch.set_facecolor('silver')
        gs = fig.add_gridspec(nrows=8, ncols=16)
        fig.suptitle('Case ID: '+str(cur.name) + 
                     '      Countdown to First-Flash: '+countdown_str+
                     '      Current Time: '+str(t.strftime('%H:%M:%S')) + ' UTC', fontsize=title_size)
        
        #Subplot 1
        ax1 = fig.add_subplot(gs[:,:8], projection=geo_crs)
        ax1.coastlines()
        ax1.add_feature(cfeature.STATES, edgecolor ='r',linewidth=1.5, zorder=0)
        ax1.add_feature(USCOUNTIES, edgecolor='g', zorder=0)
        ax1.set_extent(plot_extent, crs=plt_car_crs)
        
        ax1.scatter(x=g16_cut['lon'], y=g16_cut['lat'], transform=plt_car_crs, alpha=1, label='GOES-16 GLM Flashes (10 min.)', marker='.', s=tf_size, c='r')
        ax1.scatter(x=eni_cut['longitude'], y=eni_cut['latitude'], transform=plt_car_crs, alpha=1, label='Earth Networks Flashes (10 min.)', marker='o', s=tf_size, c='k')
        ax1.scatter(x=cur_lon, y=cur_lat, transform=plt_car_crs, c='k', s=150, marker='x', alpha=0.25)
        ax1.legend(loc='upper right')
        ax1.set_title('GOES-16 GLM / Earth Networks / ABI Clean-IR T$_{B}$', fontsize=title_size)
        

        if np.nanmax(CMI)>0:
            a = ax1.imshow(CMI,extent=abi_extent,cmap=plt.get_cmap('nipy_spectral_r', 60), alpha=0.6, vmin=180, vmax=300, zorder=0, transform=geo_crs)
            plt.colorbar(a)
        else:
            a = ax1.imshow([[0]],extent=[-0.1,0.1,-0.1,0.1],cmap=plt.get_cmap('nipy_spectral_r', 60), alpha=0.6, vmin=180, vmax=300, zorder=0, transform=geo_crs)
            a.set_visible(False)
            plt.colorbar(a)
        if t>=cur_time:
            ax1.add_patch(mpatches.Circle(xy=[cur_lon, cur_lat], radius=r, color='r', alpha=0.25, transform=plt_car_crs, zorder=1, fill=True))
        
        

        #Subplot 2
        ax2 = fig.add_subplot(gs[:,8:], projection=plt_car_crs)
        ax2.coastlines()
        ax2.set_extent(plot_extent, crs=plt_car_crs)
        ax2.add_feature(USCOUNTIES, edgecolor='g', zorder=0)
        ax2.add_feature(cfeature.STATES, edgecolor ='r',linewidth=1.5, zorder=0)
        
        ax2.scatter(x=g16_cut['lon'], y=g16_cut['lat'], transform=plt_car_crs, alpha=1, label='GOES-16 GLM Flashes (10 min.)', marker='.', s=tf_size, c='r')
        ax2.scatter(x=eni_cut['longitude'], y=eni_cut['latitude'], transform=plt_car_crs, alpha=1, label='Earth Networks Flashes (10 min.)', marker='o', s=tf_size, c='k')
        ax2.scatter(x=cur_lon, y=cur_lat, transform=plt_car_crs, c='k', s=150, marker='x', zorder=2, alpha=0.25)
        
        ax2.legend(loc='upper right')
        ax2.set_title('GOES-16 GLM / Earth Networks / MRMS -10$^{o}$C Reflectivity', fontsize=title_size)
        ax2.gridlines(crs=plt_car_crs, draw_labels=True, linewidth=1, color='white', alpha=0.25, linestyle='--')
        
        if refl_10[0][0]!=-999:
            refl_10[refl_10<10] = np.nan
            b = ax2.imshow(refl_10, extent=mrms_extent, transform=plt_car_crs, cmap=plt.get_cmap('turbo', 30), vmin=10, vmax=60, zorder=0, alpha=0.6)
            
        else:
            b = ax2.imshow([[0]], extent=[-1,1,-1,1], transform=plt_car_crs,cmap=plt.get_cmap('turbo', 30), vmin=10, vmax=60, zorder=0, alpha=0.6)
            b.set_visible(False)
        plt.colorbar(b)

        if t>=cur_time:
            ax2.add_patch(mpatches.Circle(xy=[cur_lon, cur_lat], radius=r, color='r', alpha=0.25, transform=ccrs.PlateCarree(), zorder=1, fill=True))
            
        #print (save_str)
        plt.savefig(save_str) 
        plt.close()   


# In[9]:
dt_start = datetime.now()
#Outerloop on a per-case basis
for case in cases[:]:
    print ('====='+case+'=====')
    ff_loc = '/localdata/first-flash/data/manual-analysis-v1/'+case+'/'
    glm16_all_file = sfile[case]['glm16_all']

    #Reading in the datasets and getting the time data necessary
    ff = pd.read_csv(ff_loc+case+'-ffRAW-v00combos-v1.csv', index_col=0)
    ff.index.name = 'fistart_flid'
    ff['time64'] = [np.datetime64(i) for i in ff['start_time'].values]

    g16 = pd.read_csv('/localdata/first-flash/data/GLM16-cases-allflash/'+glm16_all_file, index_col=0)
    g16['time64'] = [np.datetime64(i) for i in g16['start_time'].values]

    eni = eni_puller(sfile[case]['start_time'],sfile[case]['end_time'])
    eni['time64'] = [np.datetime64(i) for i in eni['timestamp'].values]

    #Looping through each first flash event
    for i in range(ff.shape[0])[:]:
        print (str(i)+'/'+str(ff.shape[0]-1))
        cur = ff.iloc[i]
        plotter(cur, dx, i, case, g16, eni)

print ('Figures made')
print (datetime.now()-dt_start)
