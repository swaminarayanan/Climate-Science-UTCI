

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 2024

@author: lab
"""
import pytz
from datetime import datetime
import xarray as xr
import rasterio
from netCDF4 import Dataset
import numpy as np
from datetime import datetime

# Step 1: Open the TIFF file
tiff_file = r'/home/lab/Desktop/Narayanswamy/LULC DATA/10-20N-70-90E/i/20N_070E 2020_LULCdata.tif'
with rasterio.open(tiff_file) as src:
    lulc_data = src.read(1)  # Reading the first band
    transform = src.transform
    crs = src.crs
    width = src.width
    height = src.height
    nodata = src.nodata

# Ensure nodata is of the correct type
if nodata is not None:
    nodata_value = np.float32(nodata)  # Ensure it's a 32-bit float
else:
    nodata_value = np.float32(-9999)  # Default nodata value

# Define the IST timezone
ist = pytz.timezone('Asia/Kolkata')

# Example time data with specific hours and IST timezone
time_data = [ist.localize(datetime(2020, 1, 1, 12, 0))]  # August 19, 2024, at 09:00 AM IST


# Step 2: Define time, latitude, and longitude data
# time_data = [datetime(2024, 8, 19)]  # Example time data
lat_data = np.linspace(10, 20, height)  # Example latitude data
lon_data = np.linspace(70, 80, width)  # Example longitude data
# lat_data = np.arange(10, 20.1, 0.25)  # Example latitude data
# lon_data = np.arange(70, 80.1, 0.25)  # Example longitude data

# Step 3: Create a new NetCDF file with time, latitude, and longitude dimensions
nc_file = r'/home/lab/Desktop/Narayanswamy/LULC DATA/10-20N-70-90E/i/20N_070E 2020_lullcdata.nc'
with Dataset(nc_file, 'w', format='NETCDF4') as nc:
    # Define dimensions
    nc.createDimension('time', len(time_data))
    nc.createDimension('lat', height)
    nc.createDimension('lon', width)

    # Define variables
    times = nc.createVariable('time', np.float64, ('time',))
    latitudes = nc.createVariable('lat', np.float32, ('lat',))
    longitudes = nc.createVariable('lon', np.float32, ('lon',))
    lulc = nc.createVariable('lulc', np.float32, ('time', 'lat', 'lon'))
    lulc.units = 'category'
    lulc.standard_name = 'land_use_land_cover'

    # Set variable data
    times[:] = [np.datetime64(t).astype('float64') for t in time_data]
    latitudes[:] = lat_data
    longitudes[:] = lon_data
    lulc[0, :, :] = lulc_data  # Assigning LULC data to the first (and only) time step

    # Adding metadata
    nc.description = 'LULC data converted from TIFF to NetCDF with time and spatial dimensions'
    nc.source = 'Generated from {}'.format(tiff_file)
    lulc.setncattr('nodata_value', nodata_value)  # Assign nodata_value correctly

    # Optionally, add CRS information
    crs_var = nc.createVariable('crs', 'i4')
    crs_var.spatial_ref = crs.to_wkt()

print(f"Conversion complete! NetCDF file saved as {nc_file}")
