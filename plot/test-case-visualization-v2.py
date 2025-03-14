#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
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
end_time = '20220515-0400'
dt = '5min'

t_list = pd.date_range(start=start_time, end=end_time, freq=dt)

#ul_lat = 43.00
#ul_lon = -104.00
#lr_lat = 32.00
#lr_lon = -77.00
#For NWAR case
ul_lat = 38.00
ul_lon = -96.00
lr_lat = 33.00
lr_lon = -90.00

dx = 0.2
lat_list = np.arange(lr_lat+dx,ul_lat+dx,dx)
lon_list = np.arange(ul_lon,lr_lon,dx)

#Loading in the glm16 flashes from the event
glm16_all_loc = '/localdata/first-flash/data/GLM16-cases-allflash/'
#glm16_all_loc = '../../local-data/20220504-mltest/' # DEVMODE
glm16_all =  'GLM16allflashes_v1_s202205141400_e202205150400_c202412301730.csv'
g16 = pd.read_csv(glm16_all_loc+glm16_all, index_col=0)
g16['time64'] = [np.datetime64(i) for i in g16['start_time'].values]

contour_levels = [2,5,10,25,50,75]


# In[3]:


#Reading in the test file which has the output from the ML model
trf_loc = '/localdata/first-flash/data/ml-manual-analysis/'
#trf_loc = '../../local-data/20220504-mltest/' #DEVMODE

#trf = pd.read_csv(trf_loc+'20220514-test-conus-ma-grids-v3-ABI-MRMS-GLM-202412301805-output-trf105-binary.csv', index_col=0)
trf = pd.read_csv(trf_loc+'20220514-test-nwar-ma-grids-v3-ABI-MRMS-GLM-202502211232-output.csv', index_col=0) #Short term for NWAR case
trf2_pthresh = 0.35
trf.loc[trf['trf2_p'] >= trf2_pthresh, 'trf2_c_custom'] = 1.0
trf.loc[trf['trf2_p'] < trf2_pthresh, 'trf2_c_custom'] = 0.0

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
                                        radius_of_influence=2e3)
                
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


def plotter(trf1_grid, trf2_pgrid, trf2_cgrid, lon_grid, lat_grid, refl10C, extent_mrms, mlon_grid, mlat_grid, clat, clon, dx, title, fname, t, t_plot, t_string):
    global g16
    title_size = 16
    tick_size = 12
    pt_size = 4
    trf_extent = [np.min(lon_grid),np.max(lon_grid),np.min(lat_grid),np.max(lat_grid)]
    trf1_grid_cut = trf1_grid.copy()
    trf2_pgrid_cut = trf2_pgrid.copy()
    trf2_cgrid_cut = trf2_cgrid.copy()
    refl10C_cut = refl10C.copy()

    #Preparing the trf data
    trf_truth = ((lon_grid>=clon-dx)&(lon_grid<=clon+dx)&(lat_grid>=clat-dx)&(lat_grid<=clat+dx))
    trf1_grid_cut[~trf_truth] = np.nan
    trf2_pgrid_cut[~trf_truth] = np.nan
    trf2_cgrid_cut[~trf_truth] = np.nan
    
    #Preparing the MRMS data
    if refl10C[0][0]!=-999:
        m_truth = ((mlon_grid>=clon-dx)&(mlon_grid<=clon+dx)&(mlat_grid>=clat-dx)&(mlat_grid<=clat+dx))
        refl10C_cut[~m_truth] = np.nan
    
    #Preparing the GLM16 flash data
    g16_c = g16.loc[(g16['lon'].values>=clon-dx)&(g16['lon'].values<=clon+dx)&(g16['lat'].values>=clat-dx)&(g16['lat'].values<=clat+dx)]
    g16_c = g16_c.loc[(g16['time64']<=t)&(g16['time64']>=t-timedelta(minutes=20))]
    
    #Compiling some data to simplify the plotting (I promise I'm not crazy...yet)
    d = {
        'var':['   TRF1 p(ltg 20min)', '   TRF2 c(Strong/Weak Convection)'],
        'pltx_min':[0,6],
        'pltx_max':[6,12],
        'plty_min':[0,0],
        'plty_max':[7,7],
        'cmap':['plasma','PuOr_r']
    }
    trf_data = [trf1_grid_cut, trf2_cgrid_cut]
    
    
    #Time to plot!
    crs = ccrs.PlateCarree()
    plot_extent = [clon-dx, clon+dx, clat-dx, clat+dx]
    
    fig = plt.figure(figsize=(16,8))
    fig.patch.set_facecolor('white')
    
    gs = fig.add_gridspec(nrows=8,ncols=12)
    fig.suptitle('Case: '+title+'  -  '+t_plot, fontsize=title_size)
    
    #Rolling through each of the four panels on the plot
    for i in range(2):
        #print (i)
        #Upper left - trf1
        ax = fig.add_subplot(gs[d['plty_min'][i]:d['plty_max'][i],d['pltx_min'][i]:d['pltx_max'][i]], projection=crs)

        ax.coastlines()
        ax.add_feature(USCOUNTIES, edgecolor='g', zorder=0)
        ax.add_feature(cfeature.STATES, edgecolor ='r',linewidth=2., zorder=0)
        
        ax.set_extent(plot_extent, crs=crs)

        #Plotting the GLM16 data
        ax.scatter(x=g16_c['lon'], y=g16_c['lat'], transform=crs, alpha=0.5, label='G16 GLM Flashes (20 min.)', marker='.', s=pt_size, c='k')

        #Plotting the MRMS data
        if refl10C[0][0]!=-999:
                refl10C_cut[refl10C_cut<10] = np.nan
                b = ax.imshow(refl10C_cut, extent=extent_mrms, transform=crs, cmap=plt.get_cmap('turbo', 30), vmin=10, vmax=60, zorder=0, alpha=1.0)

        else:
            b = ax.imshow([[0]], extent=[-1,1,-1,1], transform=crs,cmap=plt.get_cmap('turbo', 30), vmin=10, vmax=60, zorder=0, alpha=1.0)
            b.set_visible(False)
        cbar = plt.colorbar(b)
        cbar.set_label('MRMS -10$^\circ$C Reflectvity (dBZ)', fontsize=tick_size)
        cbar.ax.tick_params(labelsize=tick_size)

        #Plotting the TRFdata
        if i == 0:
            a = ax.contour(lon_grid, lat_grid, trf_data[i], levels=[2,5,10,25,50,75], cmap=d['cmap'][i], transform=crs)
            ax.clabel(a, a.levels, fmt=fmt, fontsize=8)
            ax.legend(loc='lower right', fontsize=tick_size, facecolor='white', framealpha=1)
        else:
            c = ax.imshow(trf_data[i][:,::-1].T, extent=trf_extent, transform=crs, cmap=d['cmap'][i], alpha=0.8)
            legend_elements = [Patch(facecolor='darkorange', edgecolor='black', alpha=0.8, label='Strong Convection'), Patch(facecolor='indigo', edgecolor='black', alpha=0.8, label='Weak Convection')]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=tick_size, facecolor='white', framealpha=1)
        
        
        ax.set_title(d['var'][i], fontsize=title_size, loc='left')
    plt.savefig('/localdata/first-flash/figures/ml-trf-output/'+fname+'/'+fname+'_'+t_string+'.png')
    plt.close()


# In[ ]:


#looping through each time step
for t in t_list[:]: #72:73 = 2100 UTC
    print (t)
    y, m, d, doy, hr, mi = datetime_converter(t)
    t_string = y+m+d+'-'+hr+mi
    t_plot = m+'-'+d+'-'+y+' '+hr+':'+mi+' UTC'
    
    #Getting the data from only that timestamp
    trf_cut = trf.loc[trf['timestamp']==t_string]
    
    #Getting the grids from trf_cut for the ML probabilities and the lats/lons
    trf1_grid = np.reshape(trf_cut['trf1_p'].values, (len(lon_list),len(lat_list)))*100
    trf2_pgrid = np.reshape(trf_cut['trf2_p'].values, (len(lon_list),len(lat_list)))*100
    #trf2_cgrid = np.reshape(trf_cut['trf2_c'].values, (len(lon_list),len(lat_list))) * 100 #v2 (50% threshold)
    trf2_cgrid = np.reshape(trf_cut['trf2_c_custom'].values, (len(lon_list),len(lat_list))) *100 #v3 (custom threshold)

    lon_grid = np.reshape(trf_cut['lon'].values, (len(lon_list),len(lat_list)))
    lat_grid = np.reshape(trf_cut['lat'].values, (len(lon_list),len(lat_list)))
    
    #Getting the MRMS data
    refl10C, extent_mrms, mlon_grid, mlat_grid = mrms_puller(t)
    
    #Plotting the data
    #plotter(trf1_grid, trf2_pgrid, trf2_cgrid, lon_grid, lat_grid, refl10C, extent_mrms, mlon_grid, mlat_grid, 37.5, -99.5, 2., 'SW Kansas', 'swks-v3', t, t_plot, t_string)
    #plotter(trf1_grid, trf2_pgrid, trf2_cgrid, lon_grid, lat_grid, refl10C, extent_mrms, mlon_grid, mlat_grid, 41.0, -84.5, 2., 'NW Ohio', 'nwoh-v3', t, t_plot, t_string)
    plotter(trf1_grid, trf2_pgrid, trf2_cgrid, lon_grid, lat_grid, refl10C, extent_mrms, mlon_grid, mlat_grid, 35.5, -93.5, 2., 'NW Arkansas', 'nwar-v3', t, t_plot, t_string)
    plotter(trf1_grid, trf2_pgrid, trf2_cgrid, lon_grid, lat_grid, refl10C, extent_mrms, mlon_grid, mlat_grid, 34.69, -94.27, .6, 'Mena, Arkansas Thunderstorm', 'mear-v3', t, t_plot, t_string)
    plotter(trf1_grid, trf2_pgrid, trf2_cgrid, lon_grid, lat_grid, refl10C, extent_mrms, mlon_grid, mlat_grid, 36.06, -94.31, .6, 'Fayetteville, Arkansas Thunderstorm', 'faar-v3', t, t_plot, t_string)

