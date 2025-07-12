#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 16:06:14 2024

@author: lab
"""

import xarray as xr
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import geopandas as gpd
import pickle
from shapely.geometry import Point
from cartopy.io import shapereader
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from cartopy.io import shapereader as shpreader
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from netCDF4 import Dataset as netcdf
from scipy.stats import ttest_1samp
from scipy import stats

from matplotlib import ticker
from scipy.signal import detrend

# Apply the mask to the data
india_shapefile = shapereader.Reader(r'/home/lab/Desktop/Narayanswamy/swami/Admin2.shp')
india_geometries = list(india_shapefile.geometries())
india_polygon = unary_union(india_geometries)
shp = cfeature.ShapelyFeature(india_geometries, ccrs.PlateCarree(), facecolor='none', edgecolor='black')
# Function to check if a point is inside India
def is_inside_india(lat, lon, india_polygon):
    point = Point(lon, lat)
    return india_polygon.contains(point)


ds1 = xr.open_dataset(r'/home/lab/Desktop/Narayanswamy/heat_stress/heat/rh_daily.nc')
ds2 = xr.open_dataset(r'/home/lab/Desktop/Narayanswamy/heat_stress/heat/utci_daily.nc')
ds1=ds1.sel(longitude=slice(60, 100),latitude=slice(6, 40))
ds2=ds2.sel(lon=slice(60, 100),lat=slice(6, 40))

var1 = ds2['utci'].isel(time=slice(0,3782))
var2 = ds1['rh'].isel(time=slice(0,3782))
lats = ds2['lat'].values
lons = ds2['lon'].values
slope = np.zeros((len(var1.lat), len(var1.lon)))
intercept = np.zeros((len(var1.lat), len(var1.lon)))
r_value = np.zeros((len(var1.lat), len(var1.lon)))
p_value = np.zeros((len(var1.lat), len(var1.lon)))
std_err = np.zeros((len(var1.lat), len(var1.lon)))

mask = np.zeros((len(lats), len(lons)), dtype=int)
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        if is_inside_india(lat, lon, india_polygon):
            mask[i, j] = 1
        #else:
            #mask[i, j] = 0


for i in range(len(lats)):
    for j in range(len(lons)):
        slope[i, j], intercept[i, j], r_value[i, j], p_value[i, j], std_err[i, j] = linregress(
            var1[:, i, j], var2[:, i, j])


p_value[p_value>0.001]=np.nan
p_value[~(np.isnan(p_value))]=1



#k = slope.where(mask == 1)
#l = p_value.where(mask == 1)
slope = np.where(mask == 1, slope, np.nan)
p_value = np.where(mask == 1, p_value, np.nan)
levels = np. arange(-4, 2.1, 0.5)


fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(1, 1)

ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax1.add_feature(shp, linewidth=0.5)
pcm1 = ax1.contourf(lons, lats, slope,  cmap='jet', extend='both')
# ax1.set_title('Linear regression between UTCI and RH', fontsize=20, fontweight='bold')
ax1.set_title('Linear regression between UTCI and RH', fontsize=20, fontweight='bold', y=1.05)

ax1.contourf(lons, lats, p_value, hatches=['....'], alpha=0)
ax1.set_xlabel('Longitude', fontweight='bold', fontsize=18)
ax1.set_ylabel('Latitude', fontweight='bold', fontsize=18)

ax1.set_xticks(np.arange(60, 100.1, 10))
ax1.set_yticks(np.arange(0, 40.1, 5))
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}°E'.format(x)))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}°N'.format(y)))
ax1.set_facecolor('white')

cbar1 = fig.colorbar(pcm1, ax=ax1, orientation='vertical', shrink=0.6, pad=0.02, aspect=40)
cbar1_ax = cbar1.ax
cbar1_ax.set_position([0.778, 0.2, 0.02, 0.6])  # (L-R, moving T-B, bar width,CBAR LEN)   Adjust the position and size of the colorbar

# # Adjustments
# #plt.tight_layout()
# plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.2, wspace=0.2)
plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    "axes.linewidth": 2,
    "patch.linewidth": 2,
    'xtick.major.size': 12,
    'ytick.major.size': 12,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'axes.titlesize': 16, # For axes titles
    'figure.titlesize': 20 # For overall figure title
})

plt.show()