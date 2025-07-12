

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Load the dataset
ds = xr.open_dataset(r'/home/lab/Desktop/Narayanswamy/LULC DATA/10-20N-70-90E/i/20N_070E 2020_lulcdata.nc')

# Select the region of interest
# ds = ds.sel(lon=slice(78.5, 80), lat=slice(12.5, 15))

# Extract the variable and calculate the mean over time
var1 = ds['lulc']
lulc = var1.mean(dim='time')

# Extract the longitude and latitude values
lons = ds['lon'].values
lats = ds['lat'].values

# Plotting
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])

# Create a contour plot
pcm1 = ax1.contourf(lons, lats, lulc, cmap='BuPu', extend='both')
plt.colorbar(pcm1, ax=ax1, orientation='vertical')
plt.show()
