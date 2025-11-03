#================================================
# This script uses first flash locations and find sfcOA data
# Parts of this script came from the MRMS/MRMS-ff-combo-v2.py
#
# Created: October 2025
# Author: Kevin Thiel (kevin.thiel@ou.edu)
#================================================

import numpy as np
import pandas as pd
import netCDF4 as nc
import sys
import os
from glob import glob
from datetime import datetime
from datetime import timedelta
import multiprocessing as mp
from sklearn.neighbors import BallTree
import os
import gempakio as gpk
import warnings
warnings.filterwarnings('ignore') 

#Constants
version = 1
oa_files_loc = '/localdata/first-flash/data/sfcOA-local/'
ff_combo_loc = '/localdata/first-flash/data/GLM-ff-east-west-combined/GLM-East-West_first-flash-data_v32_s202201010000_e202301010000_c202410071459.nc'

oa_vars_df = pd.read_csv('ff-sfcoa-vars.csv') #A csv of the variables we'll be pulling from in the sfcoaruc files
oa_vars_input = oa_vars_df['FIELD'].values
oa_vars_output = np.array([[var+'_T0', var+'_T1', var+'_T2', var+'_T3'] for var in oa_vars_input], dtype='str').flatten() #Adding the time lag subsets(T0,T1,T2,T3)

#Function land
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

# Takes in the glm ff times and gets the corresponding sfcOA file start times (T0, T1, T2, T3)
def sfc_oa_file_times(f_time):
    f_time2 = [pd.to_datetime(t) for t in f_time]

    #Creating list from first flash times
    oa_times_t0 = [t.strftime('sfcoaruc_%y%m%d%H') for t in f_time2]
    oa_times_t1 = [(t-timedelta(minutes=60)).strftime('sfcoaruc_%y%m%d%H') for t in f_time2]
    oa_times_t2 = [(t-timedelta(minutes=120)).strftime('sfcoaruc_%y%m%d%H') for t in f_time2]
    oa_times_t3 = [(t-timedelta(minutes=180)).strftime('sfcoaruc_%y%m%d%H') for t in f_time2]

    return oa_times_t0, oa_times_t1, oa_times_t2, oa_times_t3
    
#Takes in the sfcoa lats/lons and the first flash lat/lon and gives the index of the closest point
# on the OA grid
def oa_ff_finder(f_lat, f_lon, oa_lats, oa_lons, fistart_flid, oa_flat_idx):
	dx = 0.5	
	#Converting to radians for the ball tree
	oa_lats = oa_lats * (np.pi/180)
	oa_lons = oa_lons * (np.pi/180)
	fl_lat_rad = f_lat * (np.pi/180)
	fl_lon_rad = f_lon * (np.pi/180)

	#Creating an empty array to fill
	#If a first flash is off the sfcOA grid, the index given will be ZERO (which should be a NaN...I think...)
	oa_ff_locs = np.ones(len(f_lat)) * 0

	#Looping through each first flash to find their position in the surface oa grid
	for i in range(len(f_lat)):
		cur_fl_lat = fl_lat_rad[i]
		cur_fl_lon = fl_lon_rad[i]

		#Cutting down the searchable area to make the BallTrees smaller/faster
		oa_cutdown_locs = np.where((oa_lons>=cur_fl_lon-dx) & (oa_lons<=cur_fl_lon+dx) & (oa_lats<=cur_fl_lat+dx) & (oa_lats>=cur_fl_lat-dx))[0]
		if len(oa_cutdown_locs)==0:
			continue
		oa_flat_idx_search = oa_flat_idx[oa_cutdown_locs]
		oa_lats_search = oa_lats[oa_cutdown_locs]
		oa_lons_search = oa_lons[oa_cutdown_locs]

		#Preparing the data for the ball tree
		oa_latlons = np.vstack((oa_lats_search, oa_lons_search)).T
		ff_latlons = np.reshape([cur_fl_lat, cur_fl_lon], (-1, 2))

		#Implement a Ball Tree to capture the closest point
		btree = BallTree(oa_latlons, leaf_size=2, metric='haversine')
		distances, indices = btree.query(ff_latlons, k=1)
		closest_idx = indices[0][0]

		#Placing the oa index that is the closest to the first flash (using the full sfcOA grid)
		oa_ff_locs[i] = oa_flat_idx_search[closest_idx]

	return oa_ff_locs


#Allows us to open one sfcOA file at a time to fill the dataframe
def oa_df_filler(df, oa_vars_input, oa_vars_output, t0_locs, t1_locs, t2_locs, t3_locs, oa_lats, oa_lons, oa_data, fistart_flid, f_lat, f_lon, oa_ff_locs):
	#Looping through each variable so we only have to extract them once
	for var in oa_vars_input:
		print (var)
		#Loading the variable from the gempak grid
		var_data = oa_data.gdxarray(parameter=var)[0].values[0][0]

		#If there's t0 data that exists, loop through the t0 points in the dataframe
		#and sample the nearest sfcOA point for the current variable
		print ('t0')
		if len(t0_locs>0):
			for loc in t0_locs:
				#Getting the index to sample on the oa grid
				oa_loc = oa_ff_locs[loc]
				#Sampling the sfc oa data and placing the value in the dataframe
				df.loc[fistart_flid[loc],var+'_T0'] = var_data.flatten('C')[oa_loc]
		print ('t1')
		if len(t1_locs>0):
			for loc in t1_locs:
				#Getting the index to sample on the oa grid
				oa_loc = oa_ff_locs[loc]
				#Sampling the sfc oa data and placing the value in the dataframe
				df.loc[fistart_flid[loc],var+'_T1'] = var_data.flatten('C')[oa_loc]
		print ('t2')
		if len(t2_locs>0):
			for loc in t2_locs:
				#Getting the index to sample on the oa grid
				oa_loc = oa_ff_locs[loc]
				#Sampling the sfc oa data and placing the value in the dataframe
				df.loc[fistart_flid[loc],var+'_T2'] = var_data.flatten('C')[oa_loc]
		print ('t3')
		if len(t3_locs>0):
			for loc in t3_locs:
				#Getting the index to sample on the oa grid
				oa_loc = int(oa_ff_locs[loc])
				print (oa_loc)
				print ('OA Data Shape: '+ str(np.array(var_data).flatten('C')))
				#Sampling the sfc oa data and placing the value in the dataframe
				df.loc[fistart_flid[loc],var+'_T3'] = var_data.flatten('C')[oa_loc]

	return df

#Saving the oa	
def oa_data_saver(df, t_start, t_end, version):

    y, m, d, doy, hr, mi = datetime_converter(t_start)
    output_folder = y+m+d
    start_time_str = 's'+y+m+d+hr+mi
    y, m, d, doy, hr, mi = datetime_converter(t_end)
    end_time_str = 'e'+y+m+d+hr+mi
    y, m, d, doy, hr, mi = datetime_converter(datetime.now())
    cur_time_str = 'c'+y+m+d+hr+mi
    
    output_loc = '/localdata/first-flash/data/sfcOA-processed-v'+str(version)+'/'+output_folder +'/'
    output_file = 'sfcOA-ff-v'+str(version)+'-'+start_time_str+'-'+end_time_str+'-'+cur_time_str+'.csv'
    if not os.path.exists(output_loc):
        os.makedirs(output_loc, exist_ok=True)
    df.to_csv(output_loc+output_file)
    print (output_loc+output_file)


#====================================================
#The driver function for starmap that processes the data in two hour chunks
#====================================================
def sfcoa_driver(t_start, t_end):
	#Loading in the other data to the function
	global oa_vars_input
	global oa_vars_output
	global oa_files_loc
	global version
	global fstring_start
	global f_time
	global fistart_flid
	global f_lat
	global f_lon

	f_time = np.array(f_time)

	#Getting the 2-hour segment 
	df_locs = np.where((f_time>=np.datetime64(t_start)) & (f_time<np.datetime64(t_end)))[0]	

	#If there's first flash data lets run stuff. If not then don't!
	if len(df_locs)>0:

		#Creating an empty dataframe to fill
		fistart_flid_cutdown = fistart_flid[df_locs]
		df = pd.DataFrame(index=fistart_flid_cutdown, columns=oa_vars_output)

		#Getting the list of file names needed based on the first flash times for T0 T1 T2 T3
		oa_times_t0, oa_times_t1, oa_times_t2, oa_times_t3 = sfc_oa_file_times(f_time[df_locs])

		#Making list of files we need to loop through
		oa_file_loop_list = [t.strftime('sfcoaruc_%y%m%d%H') for t in pd.date_range(start=t_start-timedelta(minutes=180), end=t_end-timedelta(minutes=60), freq='1h')]

		#Looping through each potential file that we need to pull from
		for cur_oa_file in oa_file_loop_list:
			print (cur_oa_file)

			#checking that the path exists
			file_truther = os.path.exists(oa_files_loc+cur_oa_file)

			#Finding where we need to sample for each time lag
			t0_locs = np.where(np.array(oa_times_t0) == cur_oa_file)[0]
			t1_locs = np.where(np.array(oa_times_t1) == cur_oa_file)[0]
			t2_locs = np.where(np.array(oa_times_t2) == cur_oa_file)[0]
			t3_locs = np.where(np.array(oa_times_t3) == cur_oa_file)[0]
			t_lens = [len(t0_locs), len(t1_locs), len(t2_locs), len(t3_locs)]
			print (t_lens)
			#If no file exists, skip and there will be nans		
			if (file_truther == False):
				print ('ERROR: NO FILE EXISTS - '+cur_oa_file)
				continue
			#If the oa data file isn't needed then skip in the loop
			elif (np.sum(t_lens)==0):
				#print ('DONG! NO OA DATA NEEDED')
				continue
			#If data exists, open the file!
			oa_data = gpk.GempakGrid(oa_files_loc+cur_oa_file)
			oa_lats = oa_data.lat
			oa_lons = oa_data.lon
			#print ('DING!')
			print (cur_oa_file+' read')

			#Getting the indicies for each first flash location in the OA data
			oa_flat_idx = np.arange(0,len(oa_lats.flatten('C')),1)
			oa_ff_locs = oa_ff_finder(f_lat[df_locs], f_lon[df_locs], oa_lats.flatten('C'), oa_lons.flatten('C'), fistart_flid, oa_flat_idx)

			#Shipping all this stuff off to sample and fill the dataframe
			df = oa_df_filler(df, oa_vars_input, oa_vars_output, t0_locs, t1_locs,  t2_locs, t3_locs, oa_lats, oa_lons, oa_data, fistart_flid[df_locs], f_lat[df_locs], f_lon[df_locs], oa_ff_locs)
		
		#Saving the dataframe out
		oa_data_saver(df, t_start, t_end, version)

# WORK ZONE

#Getting the user inputs
args = sys.argv
s_time_str = args[1] #Start date YYYYMMDD (inclusive)
e_time_str = args[2] #End date YYYYMMDD (non-inclusive)

#Getting the necessary information from the netCDF file
nc_dset = nc.Dataset(ff_combo_loc,'r')

#Setting up the time range
start_time = datetime.strptime(s_time_str, '%Y%m%d')
#start_time = datetime.strptime('2022-01-19 00:00:00', '%Y-%m-%d %H:%M:%S') #DEVMODE
end_time = datetime.strptime(e_time_str, '%Y%m%d')
time_list_days = pd.date_range(start=start_time, end=end_time, freq='1D') #Daily list to loop through

#Getting the flash ids, lats, and lons for searching later...
fistart_flid = nc_dset.variables['flash_fistart_flid'][:]
f_lat = nc_dset.variables['flash_lat'][:]
f_lon = nc_dset.variables['flash_lon'][:]

#Getting the flash times for seraching later...
fistart_str = [i[0:15] for i in nc_dset.variables['flash_fistart_flid'][:]]
f_time = GLM_LCFA_times_postprocess(fistart_str, nc_dset.variables['flash_time_offset_of_first_event'][:])

#Looping through on a daily basis (and then two hour chunks to leverage multi-threading...)
for i in range(len(time_list_days)-1):
	
	t_range_start = time_list_days[i]
	t_range_end = time_list_days[i+1]
    
	#Breaking the day into 12, 2-hour chunks
	tlist_starmap = pd.date_range(start=t_range_start, end=t_range_end, freq='2H') #DEVMODE Change to '2H'

	#Sending the file string to the sfcoa_driver function that takes over from here...
	if __name__ == "__main__":
		with mp.Pool(12) as p:
			#p.starmap(sfcoa_driver, zip(tlist_starmap[:-1], tlist_starmap[1:]))
			p.starmap(sfcoa_driver, zip(tlist_starmap[0:1], tlist_starmap[1:2]))
			p.close()
			p.join()




