#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:50:21 2024

@author: lab
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the NetCDF data
data_path = r'/home/lab/Desktop/Narayanswamy/heat_stress/heat/utci_daily.nc'
ds = xr.open_dataset(data_path)

# Assuming the variable of interest is named 'var_name'
var_name = 'utci'
utci = ds[var_name]-273.16 

utci = utci.sel(lon=slice(85.75, 86.0), lat=slice(20.25, 20.50))


utci = utci.resample(time='YE').mean()

da = utci.sel(time=slice('1990-03-01', '2020-06-30')).mean(dim=('lat', 'lon')).groupby('time.year').mean()

# Temporal dimension name (e.g., 'time')
time_dim = 'time'
years = da['year'].values
# Convert time to numeric values (e.g., year)
time_numeric = xr.DataArray(np.arange(len(da)), dims=time_dim)

# Calculate the slope (m) and intercept (c) using linear regression
slope, intercept, r_value, p_value, std_err = linregress(time_numeric, da)
a = np.arange(1990, 2020, 1)
# Create the trend line
trend_line = slope * time_numeric + intercept

# Plot the original data and the trend line
plt.figure(figsize=(10, 6))
# plt.plot(da[time_dim], p_value, label='Original Data')
plt.plot(years, da, color='b', linewidth=3.5)
plt.plot(years, trend_line, label=f'Trend Line (y = {slope:.4f}x + {intercept:.4f})\n' + f'p_value (p = {p_value:.6f})', color='red')

plt.xlabel('Time', fontweight='bold', fontsize=18)
plt.ylabel('UTCI', fontweight='bold', fontsize=18)
# plt.title(f'Temporal Trend of {var_name}')
plt.title('Heat Stress Trend in Bhubaneshwar', fontweight='bold', fontsize=18)

plt.legend()
plt.grid(False)
plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.labelsize': 20,
    'figure.labelsize': 20,
    "axes.linewidth": 2,
    "patch.linewidth": 2,
    'xtick.major.size': 12,
    'ytick.major.size': 12,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'axes.titlesize': 16, # For axes titles
    'figure.titlesize': 20 # For overall figure title
})

plt.legend (fontsize=14)
plt.show()


