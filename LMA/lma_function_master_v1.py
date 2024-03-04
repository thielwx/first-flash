# =====================================================================
# This script contains functions that are deemed useful across multiple scripts
#
# Author: Kevin Thiel (kevin.thiel@ou.edu)
# Created: December 2023
#
#======================================================================

from datetime import datetime
import pandas as pd

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

def lma_data_puller(f, key):
    '''
    A function to pull data from the open h5 file into a DataFrame
    PARAMS:
        f: The opened h5 file
        key: A key to the dictionary (of the dataset, in this case events and flashes) (str)
    RETURNS:
        df: DataFrame of the extracted data set
    '''

    #Getting the dataset within the file as specified by the key
    dset = f[key]
    
    #Getting the root name for the dataset
    root_name = list(dset)[0]
    
    #Converting the datasets to pandas dataframes
    df = pd.DataFrame(np.array(dset[root_name]))
    
    return df