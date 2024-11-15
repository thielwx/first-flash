#!/usr/bin/env python
# coding: utf-8

# ====================================================
# This script contains the functions used by ma-gridded-output.py
# Author: Kevin Thiel
# Created: November 2024
# ====================================================

import yaml
import netCDF4 as nc
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import sys
import manual_analysis_function_master as ff
import multiprocessing as mp
import os
from glob import glob



def grid_maker(case, sfile, dx):
    lat_max = 50
    lat_min = 24
    lon_max = -66
    lon_min = -125

    #Creating the list of lats and lons. Putting them into a grid
    lats = np.arange(lat_min, lat_max+dx, dx)
    lons = np.arange(lon_min, lon_max+dx, dx)
    lat_grid, lon_grid = np.meshgrid(lats,lons)

    #Flattening the grid back down to 1D arrays
    lats_flat = lat_grid.flatten()
    lons_flat = lon_grid.flatten()

    #Finding the grid points within the case domains and subsetting the flattened arrays
    idx = np.where((lats_flat>=sfile[case]['lr_lat'])&(lats_flat<=sfile[case]['ul_lat'])&
                   (lons_flat>=sfile[case]['lr_lon'])&(lons_flat<=sfile[case]['ul_lon']))[0]
    lats_output = lats_flat[idx]
    lons_output = lons_flat[idx]

    return lats_output, lons_output 