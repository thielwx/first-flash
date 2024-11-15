# A file full of the functions needed to sample our MRMS and ABI data
# from the manually analyzed dataset. Functions taken from ABI-ff-combo-v1.py
# and MRMS-ff-combo-v1.py


#Importing functions
import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing as mp
from glob import glob
import gzip
from sklearn.neighbors import BallTree
import satpy.modifiers.parallax as plax
from pyproj import Proj
from pyresample import SwathDefinition, kd_tree
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore') 

#========================================================================
# Takes in the glm ff times and gets the corresponding ABI file start times (pre0 and pre10)
def abi_file_times_ff(f_time):
    
    #Getting the time difference from the 0 and 5 ones place
    dt_int = int(f_time.strftime('%M'))%5
    
    #Changing those minutes that are on the 0 and 5 ones place so they're five minutes away
    if dt_int == 0:
        dt_int=5

    #Getting the target file times from the most recent file and the file ten minutes before that
    file_time_pre0 = (f_time - timedelta(minutes=int(dt_int)-1)).strftime('s%Y%j%H%M')
    file_time_pre10 = (f_time - timedelta(minutes=int(dt_int)+9)).strftime('s%Y%j%H%M')
    
    return file_time_pre0, file_time_pre10
#========================================================================


#========================================================================
# A function that finds the abi file based on its time string
def abi_file_hunter(abi_time_str):
    #Turning the time string into a datetime object so we can get the date of the file
    abi_file_datetime = datetime.strptime(abi_time_str, 's%Y%j%H%M')
    y, m, d, doy, hr, mi = datetime_converter(abi_file_datetime)
    
    #Creating the file strings we'll put through glob
    acha_loc = '/localdata/first-flash/data/ABI16-ACHAC/'+y+m+d+ '/*' + abi_time_str + '*'
    cmip_loc = '/localdata/first-flash/data/ABI16-CMIPC13/'+y+m+d+ '/*' + abi_time_str + '*'
    
    #Finding the product files via glob
    acha_file = glob(acha_loc)
    cmip_file = glob(cmip_loc)

    #Tagging the times where data is missing for later
    if len(acha_file)<1:
        print ('ACHA DATA MISSING')
        print (acha_loc)
        acha_file = ['MISSING']
    if len(cmip_file)<1:
        print ('CMIP DATA MISSING')
        print (cmip_loc)
        cmip_file = ['MISSING']     
        
    return acha_file[0], cmip_file[0]
#========================================================================


#========================================================================
# This function takes in the file names and retrieves the data from the file (x,y,val)
def abi_file_loader(acha_file,cmip_file):
    
    #loading the cmip13 data
    if cmip_file != 'MISSING':
        cmip_x, cmip_y, cmip_var, cmip_lons, cmip_lats = abi_importer(cmip_file, 'CMI', np.nan)
    else:
        cmip_lats = np.array([-999])
        cmip_lons = np.array([-999])
        cmip_var = np.array([-999])
    
    #loading the acha data
    if acha_file != 'MISSING':
        acha_x, acha_y, acha_var, acha_lons, acha_lats = abi_importer(acha_file, 'HT', np.nan)
    else:
        acha_lats = np.array([-999])
        acha_lons = np.array([-999])
        acha_var = np.array([-999])
    
    #If no acha data are available, we'll put in the artificial bounds of 280 based on Thiel et al 2020    
    if  (acha_file == 'MISSING') and (cmip_file != 'MISSING'):
        cmip_var = cmip_var.flatten(order='C')
        cmip_lats = cmip_lats.flatten(order='C')
        cmip_lons = cmip_lons.flatten(order='C')
        cmip_lats = cmip_lats[cmip_var>280]
        cmip_lons = cmip_lons[cmip_var>280]
        cmip_var = cmip_var[cmip_var>280]
        
    
    #If the CMIP and ACHA data are there, resampling the ACHA data to the CMIP 2km grid and use as a clear sky mask
    if (cmip_file != 'MISSING') and (acha_file != 'MISSING'):
        #Resampling the ACHA the CMIP grid
        acha_var = resample(acha_var, acha_lats, acha_lons, cmip_lats, cmip_lons)
        #Appling a mask to the cmip data based on the acha data
        cmip_var[np.isnan(acha_var)] = np.nan
        #Flattening the arrays for the output
        cmip_var = cmip_var[acha_var>0]
        cmip_lats = cmip_lats[acha_var>0]
        cmip_lons = cmip_lons[acha_var>0]
        acha_var = acha_var[acha_var>0]
        
    # If we dont have cmip data but do have acha data, just flatten the data and swap them with the cmip_x/y
    elif (cmip_file == 'MISSING') and (acha_file != 'MISSING'):
        cmip_lons = acha_lons[acha_var>0]
        cmip_lats = acha_lats[acha_var>0]
        acha_var = acha_var[acha_var>0]
        
    return cmip_lats, cmip_lons, acha_var, cmip_var

def abi_file_loader_v2(acha_file,cmip_file):

    #loading the cmip13 data
    cmip_x, cmip_y, cmip_var, cmip_lons, cmip_lats = abi_importer(cmip_file, 'CMI', np.nan)
    
    #loading the acha data
    acha_x, acha_y, acha_var, acha_lons, acha_lats = abi_importer(acha_file, 'HT', np.nan)        
    
    #If the CMIP and ACHA data are there, resampling the ACHA data to the CMIP 2km grid and use as a clear sky mask
    #Resampling the ACHA the CMIP grid
    acha_var = resample(acha_var, acha_lats, acha_lons, cmip_lats, cmip_lons)
    #Appling a mask to the cmip data based on the acha data
    cmip_var[np.isnan(acha_var)] = np.nan
    #Flattening the arrays for the output
    cmip_var = cmip_var[acha_var>0]
    cmip_lats = cmip_lats[acha_var>0]
    cmip_lons = cmip_lons[acha_var>0]
    acha_var = acha_var[acha_var>0]
        
    return cmip_lats, cmip_lons, acha_var, cmip_var

#Short function to shorten abi_file_loader
def abi_importer(file, var, fill_val):
    dset = nc.Dataset(file, 'r')
    x = dset.variables['x'][:]
    y = dset.variables['y'][:]
    var = np.ma.filled(dset.variables[var][:,:], fill_value=fill_val)
    lons, lats = latlon(dset)
    return x, y, var, lons, lats


def resample(field, orig_lats, orig_lons, target_lats, target_lons):

    #Creating 2D grids: From original import
    #orig_lats2d, orig_lons2d = np.meshgrid(orig_lats, orig_lons)

    #Creating the swath definitions
    target_swath = SwathDefinition(lons=target_lons,lats=target_lats)
    orig_swath = SwathDefinition(lons=orig_lons,lats=orig_lats)

    #Resampling using a KD-tree to fit the data to a grid
    output = kd_tree.resample_nearest(source_geo_def=orig_swath,
                            data=field,
                            target_geo_def=target_swath,
                            radius_of_influence=4e4)
    
    output[output==0.] = np.nan
    
    return output


#A function that takes in a netCDF Dataset and returns its x/y coordinates as as lon/lat
def latlon(data):    
    #Getting our variables from the netcdf file
    sat_h = data.variables['goes_imager_projection'].perspective_point_height
    sat_lon = data.variables['goes_imager_projection'].longitude_of_projection_origin
    sat_sweep = data.variables['goes_imager_projection'].sweep_angle_axis
    
    X = data.variables['x'][:] * sat_h
    Y = data.variables['y'][:] * sat_h

    #Setting up our projections
    p = Proj(proj='geos', h=sat_h, lon_0=sat_lon, sweep=sat_sweep)
    YY, XX = np.meshgrid(Y, X)
    lons, lats = p(XX, YY, inverse=True)
    
    #Cleaning up the data
    lons[lons > 10000] = np.nan
    lats[lats > 10000] = np.nan
    lons[lons>180] = lons[lons>180] - 360
    
    return lons.T, lats.T
#========================================================================

#========================================================================
# A function that finds the necessary ACHA and CMIP13 values within 20 km of a point
def abi_data_sampler(abi_lats, abi_lons, acha_vals, cmip_vals, cur_fl_lat, cur_fl_lon):
    max_range = 20 #Range in km to search
    R = 6371.0087714 #Earths radius in km
    dx = 1.0 #Search range in degrees (to cut down the amount of ABI data we're searching)
    
    #Cutting down the ABI searchable data
    abi_locs = np.where((abi_lons>=cur_fl_lon-dx) & (abi_lons<=cur_fl_lon+dx) & (abi_lats<=cur_fl_lat+dx) & (abi_lats>=cur_fl_lat-dx))[0]
    
    #If no data are found, assign all values to nans
    if len(abi_locs)==0:
        cmip_min = np.nan
        cmip_05 = np.nan
        acha_max = np.nan
        acha_95 = np.nan

    else:
        #If there's no ABI but there is CMIP data, set the ACHA to 9000 m for the parallax correction
        if acha_vals[0]==-999:
            lon_search, lat_search = plax.get_parallax_corrected_lonlats(sat_lon=-75.0, sat_lat=0.0, sat_alt=35786023.0,
                                    lon=abi_lons[abi_locs], lat=abi_lats[abi_locs], height=9000)
            # print ('Ding!')
            # print (lon_search)
            # print (lat_search)
        
        #If we have data for both CMIP and ACHA, then use ACAH heights for the parallax correction
        else:
            #Running the parallax correction
            lon_search, lat_search = plax.get_parallax_corrected_lonlats(sat_lon=-75.0, sat_lat=0.0, sat_alt=35786023.0,
                                                lon=abi_lons[abi_locs], lat=abi_lats[abi_locs], height=acha_vals[abi_locs])
        
        #Converting the abi lat/lon to radians
        abi_lats_rad = lat_search * (np.pi/180)
        abi_lons_rad = lon_search * (np.pi/180)
        #Converting first flash lat/lon to radians
        fl_lat_rad = cur_fl_lat * (np.pi/180)
        fl_lon_rad = cur_fl_lon * (np.pi/180)

        #Configuring the lat/lon data for the BallTree
        ff_latlons = np.reshape([fl_lat_rad,fl_lon_rad], (-1, 2))
        abi_latlons = np.vstack((abi_lats_rad, abi_lons_rad)).T
        
        #Implement a Ball Tree to capture the maximum/minimum and 95th/05th percentiles within the range of 20km
        btree = BallTree(abi_latlons, leaf_size=2, metric='haversine')
        indicies = btree.query_radius(ff_latlons, r = max_range/R)
        idx = indicies[0]
            
        #If no data in ball tree range OR the file is empty then set all to nans, if there's data, sample it!
        
        if (acha_vals[0] != -999) and (len(idx)!=0):
            acha_max = np.nanmax(acha_vals[idx])
            acha_95 = np.nanpercentile(a=acha_vals[idx], q=95)
        else: 
            acha_max = -999
            acha_95 = -999
            #print ('ACHA NANs')
            
        if (cmip_vals[0] != -999) and (len(idx)!=0):
            cmip_min = np.nanmin(cmip_vals[idx])
            cmip_05 = np.nanpercentile(a=cmip_vals[idx], q=5)
        else:
            cmip_min = -999
            cmip_05 = -999
            #print ('CMIP NANs')
        
    return cmip_min, cmip_05, acha_max, acha_95
#========================================================================

#========================================================================
# Getting the lat/lons of all the hits from the manual analysis within 1000 km and the prev/next 20 minutes of
# the first falsh point
def glm_ff_pts(hit, f_time, cur_lat, cur_lon):
    #Parsing the data by time
    dt = timedelta(minutes=20)
    hit_tcut = hit.loc[(hit['file_datetime'] >= f_time-dt) & (hit['file_datetime'] <= f_time+dt)]
    
    #Parsing the points by space
    max_range = 1000 #Range in km to search
    R = 6371.0087714 #Earths radius in km
    
    idx = ball_tree_runner(cur_lat, cur_lon, hit_tcut['lat'].values, hit_tcut['lon'].values, max_range)
    
    #Cutting down the hit dataframe spatially
    hit_tscut = hit_tcut.iloc[idx]
    
    return hit_tscut['lat'].values, hit_tscut['lon'].values


# Function that removes all ABI points <= 1000 km from the first flash points and <= 20 km from any first flash point
def abi_subset(abi_lats, abi_lons, acha_vals, cmip_vals, ff_lats, ff_lons, cur_lat, cur_lon):    
    #Parsing the points by space
    R = 6371.0087714 #Earths radius in km
    if abi_lats[0]!=-999:
    
        #Setting up a dataframe to run the conus mask on the data
        d = {
            'lats': abi_lats,
            'lons': abi_lons,
            'acha': acha_vals,
            'cmip': cmip_vals
            }

        abi_df = pd.DataFrame(data=d)
        abi_df_clipped = conus_mask(abi_df)

        #Running first ball tree to get all abi pts within 1000 km of current lat/lon
        idx = ball_tree_runner(cur_lat, cur_lon, abi_df_clipped['lats'].values, abi_df_clipped['lons'].values, 1000)

        #Cutting down the target datasets to only those within 1000 km
        abi_lats = abi_df_clipped['lats'].values[idx]
        abi_lons = abi_df_clipped['lons'].values[idx]
        acha_vals = abi_df_clipped['acha'].values[idx]
        cmip_vals = abi_df_clipped['cmip'].values[idx]

        idx_accumulator = []
        #Removing the values within 20 of any first flash in the 20 minute window
        for i in range(len(ff_lats)):
            max_range = 20 #Range in km to search

            #Running the second ball tree to find points are <20 km a first flash point to exclude for later
            idx = ball_tree_runner(ff_lats[i], ff_lons[i], abi_lats, abi_lons, max_range)
            idx_accumulator = np.append(idx_accumulator, idx, axis=0)

        if len(idx_accumulator)>0:
            #Removing the values within 20 km of the first flashes
            idx_accumulator = idx_accumulator.astype('int')
            idx_removed = np.unique(idx_accumulator)
            abi_lats = np.delete(abi_lats, idx_removed)
            abi_lons = np.delete(abi_lons, idx_removed)
            acha_vals = np.delete(acha_vals, idx_removed)
            cmip_vals = np.delete(cmip_vals, idx_removed)

        #Selecting ten points in the dataset to pull from randomly
        random_ints = np.random.choice(np.arange(0, len(abi_lats)), 10, replace=False)

        #Getting the ten randomly sampled data points
        r_lats = abi_lats[random_ints]
        r_lons = abi_lons[random_ints]

        acha_max_samples = np.ones(len(r_lats))*np.nan
        acha_95_samples = np.ones(len(r_lats))*np.nan
        cmip_min_samples = np.ones(len(r_lats))*np.nan
        cmip_05_samples = np.ones(len(r_lats))*np.nan

        #Looping through the randomly selected points to get the nearby points
        for i in range(len(r_lats)):
            #Ball tree to get the samples within 20 km of randomly sampled points
            idx = ball_tree_runner(r_lats[i], r_lons[i], abi_lats, abi_lons, 20)
        
            #If no data in ball tree range OR the file is empty then set all to nans, if there's data, sample it!
            if (acha_vals[0] != -999) and (len(idx)!=0):
                acha_max = np.nanmax(acha_vals[idx])
                acha_95 = np.nanpercentile(a=acha_vals[idx], q=95)
            else: 
                acha_max = -999
                acha_95 = -999
                #print ('ACHA NANs')
            if (cmip_vals[0] != -999) and (len(idx)!=0):
                cmip_min = np.nanmin(cmip_vals[idx])
                cmip_05 = np.nanpercentile(a=cmip_vals[idx], q=5)
            else:
                cmip_min = -999
                cmip_05 = -999

            acha_max_samples[i] = acha_max
            acha_95_samples[i] = acha_95
            cmip_min_samples[i] = cmip_min
            cmip_05_samples[i] = cmip_05

        #Making a dictionary to pass everything cleanly later
        d = {
            'CMIP_min': cmip_min_samples,
            'CMIP_05': cmip_05_samples,
            'ACHA_max': acha_max_samples,
            'ACHA_95': acha_95_samples,
            'random_lat': r_lats,
            'random_lon': r_lons
        }
        df = pd.DataFrame(data=d)
    
    else:
        df = pd.DataFrame()
    return df

#Function to quickly run ball trees and make the code easier to work with
def ball_tree_runner(c_lat, c_lon, t_lats, t_lons, max_range):
    R = 6371.0087714 #Earths radius in km
    
    #Converting the center lat/lon to radians
    c_lat_rad = c_lat * (np.pi/180)
    c_lon_rad = c_lon * (np.pi/180)
    
    #Converting the target lat/lon to radians
    t_lat_rad = t_lats * (np.pi/180)
    t_lon_rad = t_lons * (np.pi/180)    

    #Configuring the lat/lon data for the BallTree
    cur_latlons = np.reshape([c_lat_rad, c_lon_rad], (-1, 2))
    target_latlons = np.vstack((t_lat_rad, t_lon_rad)).T
    
    if target_latlons.shape[0]>0:
        #Implement a Ball Tree to capture within the range of 500km
        btree = BallTree(target_latlons, leaf_size=2, metric='haversine')
        indicies = btree.query_radius(cur_latlons, r = max_range/R)
        idx = indicies[0]
    else:
        idx = []
    
    return idx

def conus_mask(df):
    #Converting the DataFrame to a GeoDataFrame and assigning the coordinate reference system
    gdata = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lons, df.lats))
    gdata.set_crs(epsg=4326, inplace=True)

    #Loading in the shapefile, setting the matching coordinate reference system, and creating a CONUS mask
    conus = gpd.read_file('../tl_2024_us_state/')
    #conus = gpd.read_file('../../../../git-thielwx/first-flash/tl_2024_us_state/') #DEVMODE
    conus = conus.to_crs('EPSG:4326')
    notCONUS = ['Alaska', 'Hawaii', 'Puerto Rico', 'Commonwealth of the Northern Mariana Islands', 'Guam', 'United States Virgin Islands', 'American Samoa']
    mask = conus['NAME'].isin(notCONUS)
    conus = conus[~mask]

    #Clipping the first flash data to the CONUS mask
    df_clipped = gpd.clip(gdata, conus)

    return df_clipped
#========================================================================
#========================================================================
#========================================================================
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


#Getting the right times and file strings to feed into the MRMS_data_loader
def mrms_setup(f_time):
    y, m, d, doy, hr, mi = datetime_converter(f_time)
    fstring_start = '/raid/swat_archive/vmrms/CONUS/'+y+m+d+'/multi/'
    
    emi = str(int(mi) - int(mi)%2)
    
    m_time = datetime.strptime(y+m+d+hr+emi, '%Y%m%d%H%M')
    
    return fstring_start, m_time


def MRMS_data_loader(time, fstring_start ,var):
    
    #Using the current time and variable to get the MRMS file we need
    y, m, d, doy, hr, mi = datetime_converter(time)
    file_time_str = y+m+d+'-'+hr+mi+'*.netcdf.gz'
    file_str = fstring_start+var+'/00.50/'+file_time_str
    
    #Searching for the file
    file_locs = glob(file_str)
    if len(file_locs)==0:
        print ('ERROR: File not found')
        print (file_str)
        lat_data = [-999]
        lon_data = [-999]
        data = [-999]
    elif len(file_locs)>1:
        print ('ERROR: More than one file found')
        lat_data = [-999]
        lon_data = [-999]
        data = [-999]
    else:
        #This is what I call a 'pro-gamer' move...loading the netcdfs while zipped on another machine
        with gzip.open(file_locs[0]) as gz:
            with nc.Dataset('dummy', mode='r', memory=gz.read()) as dset:
                #loading in the data from the MRMS netcdf file
                x_pix = dset.variables['pixel_x'][:] #Pixel locations (indicies) for LATITUDE
                y_pix = dset.variables['pixel_y'][:] #Pixel locations (indicies) for LONGITUDE
                data = dset.variables[var][:]

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
                
                #Removing any false data (less data to process)
                locs = np.where((data>0))[0]
                lon_data = lon_data[locs]
                lat_data = lat_data[locs]
                data = data[locs]

    return lat_data, lon_data, data


#A function to find the maximum/95th percentile values within a specified range:
def mrms_max_finder(cur_fl_lat, cur_fl_lon, mrms_lats, mrms_lons, mrms_data):
    dx = 0.5 #Search range in degrees (to cut down the amount of MRMS data we're searching)
    max_range  = 20
    
    #Cutting down the mrms searchable data and converting lat/lon to radians
    mrms_locs = np.where((mrms_lons>=cur_fl_lon-dx) & (mrms_lons<=cur_fl_lon+dx) & (mrms_lats<=cur_fl_lat+dx) & (mrms_lats>=cur_fl_lat-dx))[0]
    if len(mrms_locs)==0:
        mrms_data_max = np.nan
        mrms_data_95 = np.nan
    
    else:
        mrms_lats_cur = mrms_lats[mrms_locs]
        mrms_lons_cur = mrms_lons[mrms_locs]
        mrms_data_search = mrms_data[mrms_locs]
        
        #Run a ball tree to find the mrms points within 20 km of the current point
        idx = ball_tree_runner(cur_fl_lat, cur_fl_lon, mrms_lats_cur, mrms_lons_cur, max_range)

        if len(idx)==0:
            mrms_data_max = np.nan
            mrms_data_95 = np.nan
        else:
            mrms_data_max = np.nanmax(mrms_data_search[idx])
            mrms_data_95 = np.nanpercentile(a=mrms_data_search[idx], q=95)
    
    return mrms_data_max, mrms_data_95


#Parsing the data based on the first flash locations and randomly sampled points from abi_data_subset
def mrms_data_subset(mrms_lats, mrms_lons, mrms_data, random_df, var):
    #Setting up empty arrays that we'll fill
    val_max = np.ones(random_df.shape[0])*np.nan
    val_95 = np.ones(random_df.shape[0])*np.nan
    
    #If we have mrms data then we'll run the ball tree to collect
    if mrms_lats[0]!=-999:
        max_range = 20
        #Getting the random lat/lon points
        r_lats = random_df['random_lat'].values
        r_lons = random_df['random_lon'].values

        #Looping through each randomly sampled point to find the mrms data in each range
        for i in range(random_df.shape[0]):
            idx = ball_tree_runner(r_lats[i], r_lons[i], mrms_lats, mrms_lons, max_range)
            if len(idx) > 0:
                #Sampling the data and placing them into the arrays
                val_max[i] = np.nanmax(mrms_data[idx])
                val_95[i] = np.nanpercentile(a=mrms_data[idx], q=95)
    
    #Placing the values in the dataframe
    random_df[var+'_max'] = val_max
    random_df[var+'_95'] = val_95
    
    return random_df