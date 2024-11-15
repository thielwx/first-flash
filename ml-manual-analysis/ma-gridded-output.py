#!/usr/bin/env python
# coding: utf-8

# ====================================================
# This script takes in the manual analysis dataset and captures the data in a gridded format with
# ABI, MRMS, GLM, and mesoA (eventaully) added
# Author: Kevin Thiel
# Created: November 2024
# ====================================================

# Process:
# Create a lat/lon grid based on the manual analysis domains (0.2 degrees)
#   Output should be two 1-D arrays
# Get a list of the ABI/MRMS/mesoA/GLM files that correspond to the glm first flashes
#   Get the unique listing of the files so you only have to load the data once
# Loop through the unique times and extract the data from each time/point in the grid
# Save all data into an accumualted dataframe and save out by case


import pandas as pd
import numpy as np
import netCDF4 as nc
from glob import glob
from datetime import datetime
import yaml

#Constants
abi_variables = ['CMIP','ACHA']
abi_variables_output = ['CMIP_min', 'CMIP_05', 'ACHA_max', 'ACHA_95']
mrms_variables = ['MergedReflectivityQCComposite','Reflectivity_-10C','ReflectivityAtLowestAltitude']
mrms_variables_output = ['MergedReflectivityQCComposite_max','MergedReflectivityQCComposite_95','Reflectivity_-10C_max','Reflectivity_-10C_95','ReflectivityAtLowestAltitude_max','ReflectivityAtLowestAltitude_95']
ma_variables = ['ma_category', 'ma_convective_core']
glm_variables = ['num_glm16_flashes']

#Loading in the yaml file with the case settings
with open('../manual-analysis-v1/case-settings-manual-analysis.yaml', 'r') as f:
    sfile = yaml.safe_load(f)

#Getting the list of cases
cases = sfile['cases']

#Looping through each manual analysis case
for case in cases:
    