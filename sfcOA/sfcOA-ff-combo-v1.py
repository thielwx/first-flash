#================================================
# This script uses first flash locations and find sfcOA data
# Created: October 2025
# Author: Kevin Thiel (kevin.thiel@ou.edu)
#================================================

import numpy as np
import pandas as pd
import sys
import os
from glob import glob


#Constants
version = 1
oa_variables = []
oa_variables_output = []
oa_file_loc = '/localdata/first-flash/data/sfcOA-local/


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



# WORK ZONE

