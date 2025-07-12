# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:35:42 2024

@author: S Narayanaswami
"""

import cartopy.crs as ccrs
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# Load the dataset
ds = xr.open_dataset(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\heat_stress\utci_daily.nc')

# Extract data for different time periods
utci = ds['utci'] - 273.16
time = ds['time']

utci_time1 = utci.sel(time=slice('1990-03-01', '1999-06-30')).mean(dim='time')
utci_time2 = utci.sel(time=slice('2000-03-01', '2009-06-30')).mean(dim='time')
utci_time3 = utci.sel(time=slice('2010-03-01', '2020-06-30')).mean(dim='time')
utci_time4 = utci.sel(time=slice('1990-03-01', '2020-06-30')).mean(dim='time')
#utci_time44 = utci.sel(time=slice('1990-03-01', '2020-06-30')).mean(dim='lat', 'lon')
utci44 = utci.sel(time=slice('1990-03-01', '2020-06-30')).mean(dim=('lat', 'lon'))

# Convert the time data to a normal 24-hour time format
time = pd.to_datetime(ds.time.values)  # Convert time values to datetime
time_hours = time.hour + time.minute / 60.0  # Extract hours and minutes as decimal hours
time_reversed = time_hours[::-1]  # Reverse the time array if needed

# Slice time_hours based on the y variables
utci1_time_hours = time[:len(utci_time1)]
utci2_time_hours = time[:len(utci_time2)]
utci3_time_hours = time[:len(utci_time3)]
utci4_time_hours = time[:len(utci44)]
#utci4_time_hours = time_hours[:len(utci4)]

# utci_time11 = ds['utci'].sel(time=slice('1990-03-01', '1999-06-30'))# - 273.16

# utci1= np.mean((utci_time11), axis=(1, 2))
# Calculate the trend
trend1 = utci_time2 - utci_time1
trend2 = utci_time3 - utci_time2
trend = trend1 + trend2

# Create a gridspec
fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(2, 3, width_ratios=[5, 5, 4])
# Plotting UTCI for different time periods
ax1 = plt.subplot(gs[0, 0], projection=ccrs.PlateCarree())
pcm1 = ax1.pcolormesh(utci_time1.lon, utci_time1.lat, utci_time1, cmap='jet')
ax1.coastlines()
ax1.set_title('Mean UTCI (1990-1999)')
ax1.set_ylabel('Latitude')
ax1.set_xlabel('Longitude')

ax2 = plt.subplot(gs[0, 1], projection=ccrs.PlateCarree())
pcm2 = ax2.pcolormesh(utci_time2.lon, utci_time2.lat, utci_time2, cmap='jet')
ax2.coastlines()
ax2.set_title('Mean UTCI (2000-2009)')
ax2.set_ylabel('Latitude')
ax2.set_xlabel('Longitude')

ax3 = plt.subplot(gs[1, 0], projection=ccrs.PlateCarree())
pcm3 = ax3.pcolormesh(utci_time3.lon, utci_time3.lat, utci_time3, cmap='jet')
ax3.coastlines()
ax3.set_title('Mean UTCI (2010-2020)')
ax3.set_ylabel('Latitude')
ax3.set_xlabel('Longitude')

ax4 = plt.subplot(gs[1, 1], projection=ccrs.PlateCarree())
pcm4 = ax4.pcolormesh(utci_time4.lon, utci_time4.lat, utci_time4, cmap='jet')
ax4.coastlines()
ax4.set_title('Mean UTCI (1990-2020)')
ax4.set_ylabel('Latitude')
ax4.set_xlabel('Longitude')

ax5 = plt.subplot(gs[0, 2], projection=ccrs.PlateCarree())
pcm5 = ax5.pcolormesh(trend.lon, trend.lat, trend, cmap='jet')
ax5.coastlines()
ax5.set_title('UTCI Bias (1990-2020)')
ax5.set_ylabel('Latitude')
ax5.set_xlabel('longitude')

# Add colorbar
cbar = plt.colorbar(pcm1, ax=[ax1, ax2, ax3, ax4], orientation='horizontal', extend='both', pad=0.04, shrink=0.8704)
cbar.set_label('UTC (°c/day)')
cbar = plt.colorbar(pcm5, ax=[ax5], orientation='vertical',  extend='both',pad=0.05, shrink=0.7487)
#plt.colorbar(pcm5, shrink=0.5, aspect=20)
#cbar.set_label('UTC (°K/day)')

# Add line graphs
#utci1=utci1.sel(time=slice('1990-03-01', '1999-06-30'))
#pcm6=np.mean(utci1, axis=(1,2))
#utci_daily_avg = utci
ax6 = plt.subplot(gs[1, 2])
ax6.plot(utci4_time_hours, utci44, label='1990-1999')
ax6.set_title('Mean UTCI Over Time')
# cbar = plt.colorbar(pcm6, ax=[ax6], orientation='horizontal', pad=0.08)
# Add line graphs
# ax6 = plt.subplot(gs[1, 2], projection=ccrs.PlateCarree())
# time_values = np.arange(len(utci_time1))
# ax6.plot(utci1, label='none')
# ax6.plot(time_values, utci_time2, l)abel='2000-2009')
# ax6.plot(time_values, utci_time3, label='2010-2020')
# ax6.plot(time_values, utci_time4, label='1990-2020')
# cbar = plt.colorbar(time_values, ax=[ax6], orientation='horizontal', pad=0.08)
cbar.set_label('UTC (°c/day)')
ax6.set_title('Mean UTCI Over Time')
ax6.set_xlabel('Time')
ax6.set_ylabel('Mean UTCI')
ax6.legend()
# Adjust layout and spacing
plt.tight_layout()
# Add text annotations
plt.text(0.5115, 0.1, 'Data source: UTCI', fontsize=8, transform=fig.transFigure, ha='center')

# monthly_counts = merged_df.groupby('Month').size()
# monthly_counts
# fig, ax = plt.subplots(1, 1)
# fig.set_size_inches(15,7)
# monthly_counts.plot(kind='bar', ax=ax)
# plt.show()
# fig, ax = plt.subplots(1, 1)
# fig.set_size_inches(15,7)

# monthly_counts.plot(kind='bar', ax=ax)
# monthly_counts.plot(kind='line', ax=ax, color='red', marker='o')

# ax.set_title('Total Crime by Month')
# ax.set_ylabel('Total Incidents')

# plt.show()

# # Add text annotations
# plt.text(0.5, -0.05, 'Data source: UTCI', fontsize=8, transform=fig.transFigure, ha='center')

# plt.show()
# line graphs using sliced time_hours
#ax1 = plt.subplot(gs[0, 0])
#ax1.plot(utci1_time_hours, utci1, label='1990-1999')
#ax2 = plt.subplot(gs[0, 1])
#ax2.plot(utci2_time_hours, utci2, label='2000-2009')
#ax3 = plt.subplot(gs[1, 0])
#ax3.plot(utci3_time_hours, utci3, label='2010-2020')
#ax4 = plt.subplot(gs[1, 1])
#ax4.plot(utci4_time_hours, utci4, label='1990-2020')

# Set title for each subplot
#ax1.set_title('Mean UTCI Over 1990-1999')
#ax2.set_title('Mean UTCI Over 2000-2009')
#x3.set_title('Mean UTCI Over 2010-2020')
#ax4.set_title('Mean UTCI Over 1990-2020')

# Set common labels
#for ax in axs.flat:
#    ax.set_xlabel('Time (hours)')
#    ax.set_ylabel('Mean UTCI')
#    ax.legend()



# Add text annotations
#plt.text(0.5, -0.05, 'Data source: UTCI', fontsize=8, transform=fig.transFigure, ha='center')

# Save figure or display
plt.savefig("C:/Users/S Narayanaswami/OneDrive/Desktop/summer/sumr_work_fig/utci_mean_spatial_line_plot.png", dpi=300)
plt.show()