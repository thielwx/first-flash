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
import sys
from datetime import datetime
from glob import glob
from pyresample import SwathDefinition, kd_tree
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from metpy.plots import USCOUNTIES
import os


# In[11]:


args = sys.argv
#args = ['', '20220322-perils'] #DEVMODE

case = args[1]

#For cutting down the data spatially
dxc = 0.5
dx = 1

#For setting our time intervals
dt10 = np.timedelta64(10,'m')
dt30 = np.timedelta64(30,'m')
dt50s = np.timedelta64(50, 's')

#Grabbing the case-specific data
if case == '20220322-perils':
    start_time = np.datetime64('2022-03-22 19:00')
    end_time = np.datetime64('2022-03-22 22:00')
    abi_meso_num = '2'
    
    #DEVMODE
    # ff_loc = '20220322-perils-flashes-manual-analysis.csv'
    # g16_all_loc = '../../test-data/manual-analysis-test/GLM16allflashes_v1_s202203221600_s202203230800_c202402131406.csv'
    # lma_loc = '../../test-data/manual-analysis-test/perils-LMA_RAW-flash_202203220000_c202401191517_source-min-10.csv'
    # eni_loc = '../../test-data/manual-analysis-test/eni_flash_flash20220322.csv'
    
    #LIVE
    ff_loc = '/localdata/first-flash/data/manual-analysis/20220322-perils-flashes-manual-analysis.csv'
    g16_all_loc = '/localdata/first-flash/data/GLM16-cases-allflash/GLM16allflashes_v1_s202203221600_s202203230800_c202402131406.csv'
    lma_loc = '/localdata/first-flash/data/perils-LMA-RAW/20220322/perils-LMA_RAW-flash_202203220000_c202401191517_source-min-10.csv'
    eni_loc = '/localdata/first-flash/data/ENI-base-stock/eni_flash_flash20220322.csv'


elif case == '20220423-oklma':
    start_time = np.datetime64('2022-04-23 22:00')
    end_time = np.datetime64('2022-04-24 01:00')
    abi_meso_num = '1'
    
    #DEVMODE
    # ff_loc = '20220423-oklma-flashes-manual-analysis.csv'
    ff_loc = '/localdata/first-flash/data/manual-analysis/20220423-oklma-flashes-manual-analysis.csv'
    g16_all_loc = '/localdata/first-flash/data/GLM16-cases-allflash/GLM16allflashes_v1_s202204232000_s202204241000_c202402271317.csv'
    lma_loc = '/localdata/first-flash/data/OK-LMA-RAW/20220423/OK-LMA_RAW-flash_202204230000_c202403141024_source-min-10.csv'
    eni_loc = '/localdata/first-flash/data/ENI-base-stock/eni_flash_flash20220423.csv'

else:
    print('ERROR: INCORRECT DATA')
    exit()


# In[4]:


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


# In[5]:


def abi_puller(t, abi_meso_num):
    #Getting the necessary ABI data
    CMI = [[-999]]
    x = [np.nan]
    y = [np.nan]
    extent = [np.nan]
    sat_lon = -75.0
    sat_h = 35786023.0
    geo_crs = ccrs.Geostationary(central_longitude=sat_lon,satellite_height=sat_h)
     
    
    
    #Getting the time strings needed to find the file
    cur_date = t.strftime('%Y%m%d')
    file_time_string = t.strftime('%Y%j%H%M')
    
    cur_abi = 'ABI16-CMIPM13-'+abi_meso_num
    file_loc = '/localdata/first-flash/data/'+cur_abi+'/'+cur_date+'/*s'+file_time_string+'*.nc'
    #print (file_loc)
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
        print ('ERROR: ABI DATA MISSING')
        
    return CMI, x, y, extent, geo_crs


# In[13]:


def mrms_puller(t,cur_lat,cur_lon):
    cur_date = t.strftime('%Y%m%d')
    cur_time = t.strftime('%H%M')
    
    #Making sure we are only looking for even data files
    if int(cur_time)%2 == 1:
        cur_time = str(int(cur_time)-1).zfill(2)
        
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
                                        radius_of_influence=5e3)
                
                extent_mrms = [np.min(lon), np.max(lon), np.min(lat), np.max(lat)] 
                
    else:
        print ('ERROR: MRMS DATA MISSING')
        output = [[-999]]
        extent_mrms = [-999]
            
    return output, extent_mrms


# In[7]:


def save_string(cur, i, t, case):
    fistart_flid = cur['fistart_flid']
    time_str = t.strftime('%Y%m%d-%H%M%S')
    start_str = '/localdata/first-flash/figures/manual-analysis/'+case+'/'+str(i)+'-'+fistart_flid +'/'
    save_str = start_str + time_str + '.png'

    # Uncomment when running on devlab4
    if not os.path.exists(start_str):
        os.makedirs(start_str)

    return save_str


# In[15]:


def plotter(cur, dx, i, case):
    cur_time = cur['time64']
    cur_lat = cur['lat']
    cur_lon = cur['lon']
    cur_area = cur['flash_area'] / 1e6 #Putting into km sq
    
    #Creating the time list. Every minute, along with every ten seconds in the final minute
    time_list_start = pd.date_range(start=cur_time-dt10, end=cur_time-np.timedelta64(1,'m'), freq='60s').tolist()
    time_list_mid = pd.date_range(start=cur_time-dt50s, end=cur_time-np.timedelta64(10,'s'), freq='10s').tolist()
    time_list_end = pd.date_range(start=cur_time, end=cur_time+np.timedelta64(5,'m'), freq='60s').to_list()
    time_list = time_list_start+time_list_mid+time_list_end
    
    #Caucluations for the approx area of the first flash event for continuity
    R = 6371.0087714 #Earths radius in km
    d = np.sqrt(cur_area/np.pi) #Search Radius
    r = (d/R)*(180/np.pi)
    
    tf_size = 5
    
    #Looping through the individual times
    for t in time_list:
        #Step 1: Get the data from the dataframes
        g16_cut = g16.loc[(g16['lat']>=cur_lat-dx)&(g16['lat']<=cur_lat+dx)&
                         (g16['lon']>=cur_lon-dx)&(g16['lon']<=cur_lon+dx)&
                         (g16['time64']>=t-dt10)&(g16['time64']<=t)]
        lma_cut = lma.loc[(lma['ctr_lat']>=cur_lat-dx)&(lma['ctr_lat']<=cur_lat+dx)&
                         (lma['ctr_lon']>=cur_lon-dx)&(lma['ctr_lon']<=cur_lon+dx)&
                         (lma['time64']>=t-dt10)&(lma['time64']<=t)]
        eni_cut = eni.loc[(eni['latitude']>=cur_lat-dx)&(eni['latitude']<=cur_lat+dx)&
                         (eni['longitude']>=cur_lon-dx)&(eni['longitude']<=cur_lon+dx)&
                         (eni['time64']>=t-dt10)&(eni['time64']<=t)]
        
        #=====================================================
        #Step 2: Getting the data from the MRMS and ABI
        CMI, x, y, abi_extent, geo_crs = abi_puller(t, abi_meso_num)
        refl_10, mrms_extent = mrms_puller(t, cur_lat, cur_lon)
        save_str = save_string(cur, i, t, case)
        
        #=====================================================
        #Step 3: Making the plots
        plt_car_crs = ccrs.PlateCarree()
        plot_extent = [cur_lon-dxc, cur_lon+dxc, cur_lat-dxc, cur_lat+dxc]
        
        fig = plt.figure(constrained_layout=True, figsize=(16,7))
        fig.patch.set_facecolor('silver')
        gs = fig.add_gridspec(nrows=8, ncols=16)
        fig.suptitle(str(cur['fistart_flid']) + 
                     '\n'+str(t-cur_time)+
                     '\n'+str(t.strftime('%H:%M:%S')))
        
        #Subplot 1
        ax1 = fig.add_subplot(gs[:,:8], projection=geo_crs)
        ax1.coastlines()
        ax1.add_feature(cfeature.STATES, edgecolor ='r',linewidth=1.5, zorder=0)
        ax1.add_feature(USCOUNTIES, edgecolor='g', zorder=0)
        ax1.set_extent(plot_extent, crs=plt_car_crs)
        
        ax1.scatter(x=g16_cut['lon'], y=g16_cut['lat'], transform=plt_car_crs, alpha=1, label='GLM16 Flashes (10 min.)', marker='.', s=tf_size, c='r')
        ax1.scatter(x=eni_cut['longitude'], y=eni_cut['latitude'], transform=plt_car_crs, alpha=1, label='ENI Flashes (10 min.)', marker='o', s=tf_size, c='k')
        ax1.scatter(x=cur_lon, y=cur_lat, transform=plt_car_crs, c='k', s=150, marker='x', alpha=0.25)
        ax1.legend(loc='upper right')
        ax1.set_title('GLM / ENI / CMIP13')
        
        if CMI[0,0]!=-999:
            a = ax1.imshow(CMI,extent=abi_extent,cmap=plt.get_cmap('nipy_spectral_r', 24), alpha=0.4, vmin=180, vmax=300, zorder=0)
            plt.colorbar(a)
        
        if t>=cur_time:
            ax1.add_patch(mpatches.Circle(xy=[cur_lon, cur_lat], radius=r, color='r', alpha=0.25, transform=ccrs.PlateCarree(), zorder=1, fill=True))
        
        #Subplot 2
        ax2 = fig.add_subplot(gs[:,8:], projection=plt_car_crs)
        ax2.coastlines()
        ax2.set_extent(plot_extent, crs=plt_car_crs)
        ax2.add_feature(USCOUNTIES, edgecolor='g', zorder=0)
        ax2.add_feature(cfeature.STATES, edgecolor ='r',linewidth=1.5, zorder=0)
        
        ax2.scatter(x=lma_cut['ctr_lon'], y=lma_cut['ctr_lat'], c='b', s=tf_size, marker='.', zorder=3, label='LMA Flashes (10 min.)')
        ax2.scatter(x=cur_lon, y=cur_lat, transform=plt_car_crs, c='k', s=150, marker='x', zorder=2, alpha=0.25)
        ax2.legend(loc='upper right')
        ax2.set_title('LMA / -10C dBZ')
        
        if refl_10[0,0]!=-999:
            b = ax2.imshow(refl_10, extent=mrms_extent, transform=plt_car_crs, cmap=plt.get_cmap('turbo', 30), vmin=10, vmax=60, zorder=0, alpha=0.4)
            plt.colorbar(b)

        if t>=cur_time:
            ax2.add_patch(mpatches.Circle(xy=[cur_lon, cur_lat], radius=r, color='r', alpha=0.25, transform=ccrs.PlateCarree(), zorder=1, fill=True))
            
        print (save_str)
        plt.savefig(save_str) 
        plt.close()   


# In[9]:


#Reading in the datasets and getting the time data necessary
ff = pd.read_csv(ff_loc, index_col=0)
ff['time64'] = [np.datetime64(i) for i in ff['start_time'].values]
g16 = pd.read_csv(g16_all_loc, index_col=0)
g16['time64'] = [np.datetime64(i) for i in g16['start_time'].values]
lma = pd.read_csv(lma_loc, index_col=0)
lma['time64'] = LMA_times_postprocess(lma['file_time'].values, lma['start'].values)
eni = pd.read_csv(eni_loc, index_col=0)
eni['time64'] = [np.datetime64(i) for i in eni['timestamp'].values]


# In[16]:


#Looping through each first flash event
for i in range(ff.shape[0]):
    print (str(i+1)+'/'+str(ff.shape[0]))
    cur = ff.iloc[i]
    
    plotter(cur, dx, i, case)
