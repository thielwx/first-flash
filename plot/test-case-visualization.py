#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gzip
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
from glob import glob
from pyresample import SwathDefinition, kd_tree
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gzip
from metpy.plots import USCOUNTIES


# In[2]:


#Constants
start_time = '20220514-1500'
end_time = '20220515-0300'
dt = '5min'

t_list = pd.date_range(start=start_time, end=end_time, freq=dt)

ul_lat = 43.00
ul_lon = -104.00
lr_lat = 32.00
lr_lon = -77.00
dx = 0.2
lat_list = np.arange(lr_lat+dx,ul_lat+dx,dx)
lon_list = np.arange(ul_lon,lr_lon,dx)

#Loading in the glm16 flashes from the event
glm16_all_loc = '/localdata/first-flash/data/GLM16-cases-allflash/'
#glm16_all_loc = '../../local-data/20220504-mltest/' # DEVMODE
glm16_all =  'GLM16allflashes_v1_s202205141400_e202205150400_c202412301730.csv'
g16 = pd.read_csv(glm16_all_loc+glm16_all, index_col=0)
g16['time64'] = [np.datetime64(i) for i in g16['start_time'].values]

contour_levels = [2,5,10,20,30,40,50,75]


# In[3]:


#Reading in the test file which has the output from the ML model
trf_loc = '/localdata/first-flash/data/ml-manual-analysis/'
#trf_loc = '../../local-data/20220504-mltest/' #DEVMODE

trf = pd.read_csv(trf_loc+'20220514-test-conus-ma-grids-v3-ABI-MRMS-GLM-202412301805-output-trf102.csv', index_col=0)


# In[4]:


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


# In[5]:


def mrms_puller(t):
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
                
#Hiding for now and will cut down if we need to run mutiple cases
#                 #Removing any false data and those outside the plotting bounds (less data to process)
#                 locs = np.where((data>0)&(lon_data>=cur_lon-dx)&(lon_data<=cur_lon+dx)&(lat_data>=cur_lat-dx)&(lat_data<=cur_lat+dx))[0]
#                 lon_data = lon_data[locs]
#                 lat_data = lat_data[locs]
#                 data = data[locs]
                
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
        lon_grid = [[-999]]
        lat_grid = [[-999]]
            
    return output, extent_mrms, lon_grid, lat_grid


# In[1]:


def fmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"


# In[6]:


def plotter(trf1_grid, trf2o_grid, trf2w_grid, trf2s_grid, lon_grid, lat_grid, refl10C, extent_mrms, mlon_grid, mlat_grid, clat, clon, dx, title, fname, t, t_plot, t_string):
    global g16
    title_size = 16
    tick_size = 12
    pt_size = 4
    trf_extent = [np.min(lon_grid),np.max(lon_grid),np.min(lat_grid),np.max(lat_grid)]

    #Preparing the trf data
    trf_truth = ((lon_grid>=clon-dx)&(lon_grid<=clon+dx)&(lat_grid>=clat-dx)&(lat_grid<=clat+dx))
    trf1_grid[~trf_truth] = np.nan
    trf2o_grid[~trf_truth] = np.nan
    trf2w_grid[~trf_truth] = np.nan
    trf2s_grid[~trf_truth] = np.nan
    
    #Preparing the MRMS data
    if refl10C[0][0]!=-999:
        m_truth = ((mlon_grid>=clon-dx)&(mlon_grid<=clon+dx)&(mlat_grid>=clat-dx)&(mlat_grid<=clat+dx))
        refl10C[~m_truth] = np.nan
    
    #Preparing the GLM16 flash data
    g16_c = g16.loc[(g16['lon'].values>=clon-dx)&(g16['lon'].values<=clon+dx)&(g16['lat'].values>=clat-dx)&(g16['lat'].values<=clat+dx)]
    g16_c = g16_c.loc[(g16['time64']<=t)&(g16['time64']>=t-timedelta(minutes=20))]
    
    #Compiling some data to simplify the plotting (I promise I'm not crazy...yet)
    d = {
        'var':['a) TRF1 p(ltg 20min)', 'b) TRF2 p(Strong)', 'c) TRF2 p(Weak)', 'd) TRF2 p(Other)'],
        'pltx_min':[0,6,0,6],
        'pltx_max':[6,12,6,12],
        'plty_min':[0,0,9,9],
        'plty_max':[7,7,16,16],
        'cmap':['plasma','plasma','plasma','plasma']
    }
    trf_data = [trf1_grid, trf2s_grid, trf2w_grid, trf2o_grid]
    
    
    #Time to plot!
    crs = ccrs.PlateCarree()
    plot_extent = [clon-dx, clon+dx, clat-dx, clat+dx]
    
    fig = plt.figure(figsize=(16,12))
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(nrows=16,ncols=12)
    fig.suptitle('Case: '+title+'  -  '+t_plot, fontsize=title_size)
    
    #Rolling through each of the four panels on the plot
    for i in range(4):
        #print (i)
        #Upper left - trf1
        ax = fig.add_subplot(gs[d['plty_min'][i]:d['plty_max'][i],d['pltx_min'][i]:d['pltx_max'][i]], projection=crs)

        ax.coastlines()
        ax.add_feature(cfeature.STATES, edgecolor ='r',linewidth=1.5, zorder=0)
        ax.add_feature(USCOUNTIES, edgecolor='g', zorder=0)
        ax.set_extent(plot_extent, crs=crs)

        #Plotting the GLM16 data
        ax.scatter(x=g16_c['lon'], y=g16_c['lat'], transform=crs, alpha=0.5, label='G16 GLM Flashes (20 min.)', marker='.', s=pt_size, c='k')

        #Plotting the MRMS data
        if refl10C[0][0]!=-999:
                refl10C[refl10C<10] = np.nan
                b = ax.imshow(refl10C, extent=extent_mrms, transform=crs, cmap=plt.get_cmap('turbo', 30), vmin=10, vmax=60, zorder=0, alpha=1.0)

        else:
            b = ax.imshow([[0]], extent=[-1,1,-1,1], transform=crs,cmap=plt.get_cmap('turbo', 30), vmin=10, vmax=60, zorder=0, alpha=1.0)
            b.set_visible(False)
        plt.colorbar(b)

        #Plotting the TRFdata
        if i == 0:
            a = ax.contour(lon_grid, lat_grid, trf_data[i], levels=[2,10,20,30,50,75], cmap=d['cmap'][i], transform=crs)
            ax.clabel(a, a.levels, fmt=fmt, fontsize=4)
        else:
            c = ax.imshow(trf_data[i][:,::-1].T, extent=trf_extent, transform=crs, cmap=d['cmap'][i],vmin=2, vmax=100, zorder=10, alpha=0.8)
        
        ax.legend(loc='lower right', fontsize=tick_size)
        ax.set_title(d['var'][i], fontsize=title_size, loc='left')
    plt.savefig('/localdata/first-flash/figures/ml-trf-output/'+fname+'/'+fname+'_'+t_string+'.png')
    plt.close()


# In[ ]:


#looping through each time step
for t in t_list[:]:
    print (t)
    y, m, d, doy, hr, mi = datetime_converter(t)
    t_string = y+m+d+'-'+hr+mi
    t_plot = m+'-'+d+'-'+y+' '+hr+':'+mi+' UTC'
    
    #Getting the data from only that timestamp
    trf_cut = trf.loc[trf['timestamp']==t_string]
    
    #Getting the grids from trf_cut for the ML probabilities and the lats/lons
    trf1_grid = np.reshape(trf_cut['trf1_p'].values, (len(lon_list),len(lat_list)))*100
    trf2o_grid = np.reshape(trf_cut['trf2_p_other'].values, (len(lon_list),len(lat_list)))*100
    trf2w_grid = np.reshape(trf_cut['trf2_p_weak'].values, (len(lon_list),len(lat_list)))*100
    trf2s_grid = np.reshape(trf_cut['trf2_p_strong'].values, (len(lon_list),len(lat_list)))*100
    
    lon_grid = np.reshape(trf_cut['lon'].values, (len(lon_list),len(lat_list)))
    lat_grid = np.reshape(trf_cut['lat'].values, (len(lon_list),len(lat_list)))
    
    #Getting the MRMS data
    refl10C, extent_mrms, mlon_grid, mlat_grid = mrms_puller(t)
    
    #Plotting the data
    plotter(trf1_grid, trf2o_grid, trf2w_grid, trf2s_grid, lon_grid, lat_grid, refl10C, extent_mrms, mlon_grid, mlat_grid, 37.5, -99.5, 2., 'SW Kansas', 'swks', t, t_plot, t_string)
    plotter(trf1_grid, trf2o_grid, trf2w_grid, trf2s_grid, lon_grid, lat_grid, refl10C, extent_mrms, mlon_grid, mlat_grid, 41.5, -84.5, 2., 'NW Ohio', 'nwoh', t, t_plot, t_string)

