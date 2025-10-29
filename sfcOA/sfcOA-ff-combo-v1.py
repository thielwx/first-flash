#================================================
# This script uses first flash locations and find sfcOA data
# Parts of this script came from the MRMS/MRMS-ff-combo-v2.py
#
# Created: October 2025
# Author: Kevin Thiel (kevin.thiel@ou.edu)
#================================================

import numpy as np
import pandas as pd
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
    
#Allows us to open one file at a time to fill the dataframe
def oa_df_filler(df, oa_vars_input, oa_vars_output, t0_locs, t1_locs,  t2_locs, t3_locs, oa_lats, oa_lons, oa_data, fistart_flid, f_lat, f_lon):
	x=1
	return 0
	


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
		oa_file_loop_list = [t.strftime('sfcoaruc_%y%m%d%H') for t in pd.date_range(start=t_start-timedelta(minutes=180, end=t_end, freq='1h')]

		#Looping through each potneital file that we need to pull from
		for cur_oa_file in oa_file_loop_list:

			#checking that the path exists
			file_truther = os.path.exists(oa_files_loc+cur_oa_file)

			#Finding where we need to sample for each time lag
			t0_locs = np.where(oa_times_t0 == cur_oa_file)[0]
			t1_locs = np.where(oa_times_t1 == cur_oa_file)[0]
			t2_locs = np.where(oa_times_t2 == cur_oa_file)[0]
			t3_locs = np.where(oa_times_t3 == cur_oa_file)[0]
			t_lens = [len(t0_locs), len(t1_locs), len(t2_locs), len(t3_locs)]

			#If no file exists, put the data in as dummy variables			
			if (file_truther == False):
				print ('ERROR: NO FILE EXISTS - '+cur_oa_file)
				oa_lats = [-999]
				oa_lons = [-999]
				oa_data = [-999]
			#If the oa data file isn't needed then skip in the loop
			elif (np.sum(t_lens)==0):
				continue
			#If data exists, open the file!
			else:
				oa_data = gpk.GempakGrid(oa_files_loc+cur_oa_file)
				oa_lats = oa_data.lat
				oa_lons = oa_data.lon
				print ('DING!')
				print (cur_oa_file+' read')
			
			#Shipping all this stuff off to sample and fill the dataframe
			#df = oa_df_filler(df, oa_vars_input, oa_vars_output, t0_locs, t1_locs,  t2_locs, t3_locs, oa_lats, oa_lons, oa_data, fistart_flid[df_locs], f_lat[df_locs], f_lon[df_locs])
				

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
    tlist_starmap = pd.date_range(start=t_range_start, end=t_range_end, freq='24H')

	#Sending the file string to the sfcoa_driver function that takes over from here...
    if __name__ == "__main__":
        with mp.Pool(12) as p:
            p.starmap(sfcoa_driver, zip(tlist_starmap[:-1], tlist_starmap[1:]))
            p.close()
            p.join()




