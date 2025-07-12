# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 10:38:21 2024

@author: S Narayanaswami
"""

import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from cartopy.io import shapereader
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from cartopy.io import shapereader as shpreader
from cartopy.feature import ShapelyFeature

# Define the shapefile for India
india_shapefile = shapereader.Reader(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\Admin2.shp')
india_geometries = list(india_shapefile.geometries())
india_polygon = unary_union(india_geometries)
india_feature = cfeature.ShapelyFeature(india_geometries, ccrs.PlateCarree(), facecolor='none', edgecolor='black')

# Function to check if a point is inside India
def is_inside_india(lat, lon, india_polygon):
    point = Point(lon, lat)
    return india_polygon.contains(point)

# Load the dataset
ds = xr.open_dataset(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\heat_stress\utci_daily.nc')
t2m = ds['utci'] - 273.16  # Convert temperature from Kelvin to Celsius

# Extract data for different time periods
t2m_time1 = t2m.sel(time=slice('1990-03-01', '2000-06-30')).mean(dim='time')
t2m_time2 = t2m.sel(time=slice('2001-03-01', '2010-06-30')).mean(dim='time')
t2m_time3 = t2m.sel(time=slice('2011-03-01', '2020-06-30')).mean(dim='time')
t2m_time4 = t2m.sel(time=slice('1990-03-01', '2020-06-30')).mean(dim='time')#climatology
utci44 = t2m.sel(time=slice('1990-03-01', '2020-06-30')).mean(dim=('lat', 'lon'))

# Calculate the trends
#trend1 = t2m_time1 - t2m_time4
#trend2 = t2m_time2 - t2m_time4
#trend3 = t2m_time3 - t2m_time4
trend33 = t2m_time3 - t2m_time2
trend22 = t2m_time2 - t2m_time1
a = trend22 + trend33
# Extract latitude and longitude values
lats = ds['lat'].values
lons = ds['lon'].values

# Create a mask for the entire region
mask = np.zeros((len(lats), len(lons)), dtype=int)
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        if is_inside_india(lat, lon, india_polygon):
            mask[i, j] = 1
        else:
            mask[i, j] = 0

# Apply the mask to the data
t2m_time1_masked = t2m_time1.where(mask == 1)
t2m_time2_masked = t2m_time2.where(mask == 1)
t2m_time3_masked = t2m_time3.where(mask == 1)
t2m_time4_masked = t2m_time4.where(mask == 1)
#trend1_masked = trend1.where(mask == 1)
#trend2_masked = trend2.where(mask == 1)
#trend3_masked = trend3.where(mask == 1)
a_masked = a.where(mask == 1)

time = pd.to_datetime(ds.time.values)
time_hours = time.hour + time.minute / 60.0
time_reversed = time_hours[::-1]
utci4_time_hours = time[:len(utci44)]
levels = np.arange(0, 33.1, 2.5)
# Create a gridspec
#fig = plt.figure(figsize=(16, 10))
fig = plt.figure(figsize=(15, 8.35))
gs = gridspec.GridSpec(2, 3, width_ratios=[4, 4, 5])
# Generate your plot here
#plt.figure(figsize=(10, 6))  # Adjust figsize as needed for your plot
#plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'bo-')  # Example plot

# Plotting UTCI for different time periods
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
#pcm1 = ax1.pcolormesh(utci_time1.lon, utci_time1.lat, utci_time1, cmap='jet')
mpl.rcParams['figure.facecolor'] = 'white'
pcm1 = ax1.contourf(lons, lats, t2m_time1_masked, transform=ccrs.PlateCarree(),  levels = levels, cmap='jet', extend='both')
ax1.add_feature(india_feature)
#ax1.coastlines()
ax1.set_title('Mean UTCI (1990-1999)')
ax1.set_ylabel('Latitude')
#ax1.set_xlabel('Longitude')
ax1.annotate('', (4, 4), xycoords='axes fraction', va='center')

ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
#pcm2 = ax2.pcolormesh(utci_time2.lon, utci_time2.lat, utci_time2, cmap='jet')
mpl.rcParams['figure.facecolor'] = 'white'
pcm2 = ax2.contourf(lons, lats, t2m_time2_masked, transform=ccrs.PlateCarree(), levels = levels, cmap='jet', extend='both')
ax2.add_feature(india_feature)
#ax2.coastlines()
#ax2.set_ylabel('Latitude')
ax2.set_title('Mean UTCI (2000-2009)')
##x2.set_xlabel('Longitude')
ax2.annotate('', (4, 4), xycoords='axes fraction', va='center')

ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
#pcm3 = ax3.pcolormesh(utci_time3.lon, utci_time3.lat, utci_time3, cmap='jet')
mpl.rcParams['figure.facecolor'] = 'white'
pcm3 = ax3.contourf(lons, lats, t2m_time3_masked, transform=ccrs.PlateCarree(),  levels = levels, cmap='jet', extend='both')
ax3.add_feature(india_feature)
#ax3.coastlines()
ax3.set_title('Mean UTCI (2010-2020)')
ax3.set_ylabel('Latitude')
ax3.set_xlabel('Longitude')
ax3.annotate('', (0.1, 0.5), xycoords='axes fraction', va='center')

ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
#pcm4 = ax4.pcolormesh(utci_time4.lon, utci_time4.lat, utci_time4, cmap='jet')
mpl.rcParams['figure.facecolor'] = 'white'
pcm4 = ax4.contourf(lons, lats, t2m_time4_masked, transform=ccrs.PlateCarree(),  levels = levels, cmap='jet', extend='both')
ax4.add_feature(india_feature)
#ax4.coastlines()
ax4.set_title('Mean UTCI (1990-2020)')
#ax4.set_ylabel('Latitude')
ax4.set_xlabel('Longitude')
ax4.annotate('', (0.1, 0.5), xycoords='axes fraction', va='center')

ax5 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
#pcm5 = ax5.pcolormesh(trend.lon, trend.lat, a_masked, cmap='jet')
mpl.rcParams['figure.facecolor'] = 'white'
pcm5 = ax5.contourf(lons, lats, a_masked , transform=ccrs.PlateCarree(), cmap='jet', extend='both')
ax5.add_feature(india_feature)
#ax5.coastlines()
ax5.set_title('Climatological Change (1990-2020)')
#ax5.set_ylabel('Latitude')
ax5.set_xlabel('Longitude')
ax5.annotate('', (0.1, 0.5), xycoords='axes fraction', va='center')

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(utci4_time_hours, utci44, label='1990-2020')
ax6.set_title('Climatological UTCI')
ax6.set_xlabel('Time')
ax6.set_ylabel('Mean UTCI')
ax6.annotate('', (0.1, 0.5), xycoords='axes fraction', va='center')
ax6.legend()

# Add colorbars
cbar1 = fig.colorbar(pcm4, ax=[ax1, ax2, ax3, ax4], orientation='horizontal', pad=0.1, aspect=40, extend='both', shrink=0.025)
cbar1.set_label('Mean UTCI (°C)')
cbar1_ax = cbar1.ax
fig.subplots_adjust(bottom=0.00015)  # Adjust the bottom margin to create space for the colorbar
cbar1_ax.set_position([0.17, 0.0097, 0.38, 0.438])  # (L-R, T-B, CBAR LEN, T-B   Adjust the po,sition and size of the colorbar

cbar2 = fig.colorbar(pcm5, ax=ax5, orientation='vertical', pad=0.1, aspect=40, extend='both', shrink=0.5)
cbar2.set_label('Climatological Change (°C)')
cbar2_ax = cbar2.ax
fig.subplots_adjust(bottom=0.00015)  # Adjust the bottom margin to create space for the colorbar
cbar2_ax.set_position([0.84572, 0.45, 0.003725, 0.453])  # (L-R, T-B, CBAR LEN, T-B   Adjust the position and size of the colorbar

# Add longitude and latitude labels
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_xticks(np.arange(65.7, 97.25, 5), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(5, 37.6, 5), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}°E'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}°N'.format(y)))

    #ax.set_xticks(np.arange(65.7, 97.25, 5), draw_labels=True, crs=ccrs.PlateCarree())
    #ax.set_yticks(np.arange(5, 37.6, 5), draw_labels=True, crs=ccrs.PlateCarree())
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    #ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
    #Add coastlines, states, and gridlines
    #ax.gridlines(draw_labels=True)

# Add text annotation
#fig.text(0.5, 0, 'Data source: ERA5 UTCI variable', ha='center', fontsize=8)

# Save figure or dispay
plt.savefig("C:/Users/S Narayanaswami/OneDrive/Desktop/summer/sumr_work_fig/shp_utci_mean_spatial_line_plot.png", dpi=250)#, dpi=plt.gcf().dpi)

plt.show()