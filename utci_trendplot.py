# import numpy as np
# from cartopy.io import shapereader as shpreader
# from cartopy.feature import ShapelyFeature
# from cartopy.io.shapereader import Reader
# from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
# from netCDF4 import Dataset as netcdf 
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
# # import pickle
# crs = ccrs.PlateCarree()
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), constrained_layout=False,
#                            subplot_kw={'projection': crs})

# # filt=pickle.load(open('/home/lakshman/HW/heatsress/filter_data_utci.pkl','rb'))
# # filt=filt['val']

# ncset= netcdf(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\heat_stress\utci_daily.nc')
# lons = ncset['lon'][:]  
# lats = ncset['lat'][:]           
# # sst  = ncset['utci'][0:1342,:,:]   
# sst1 = ncset['utci'][1343:2562,:,:]   
# sst2 = ncset['utci'][2563:3782,:,:]   

# nctime1 = ncset['time'][0:1342]
# nctime2 = ncset['time'][1343:2562]
# nctime3 = ncset['time'][2563:3782]

# t_unit = ncset['time'].units

# try :
#     t_cal =ncset['time'].calendar
# except AttributeError :
#     t_cal = u"gregorian" 

# nt, nlat, nlon = sst.shape
# nt1, nlat1, nlon1 = sst.shape
# nt2, nlat2, nlon2 = sst.shape

# ngrd = nlon*nlat

# sst_grd  = sst.reshape((nt, ngrd), order='F') 
# x        = np.linspace(1,nt,nt)
# sst_rate = np.empty((ngrd,1))
# sst_rate[:,:] = np.nan

# for i in range(ngrd): 
#     y = sst_grd[:,i]   
#     if(not np.ma.is_masked(y)):         
#         z = np.polyfit(x, y, 1)
#         sst_rate[i,0] = z[0]*122.0
            
# sst_rate = sst_rate.reshape((nlat,nlon), order='F')
# # sst_rate=sst_rate*filt
# sst_rate= sst_rate.astype('float')
# sst_rate[sst_rate == 0] = np.nan
# fname = shpreader.Reader(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\Admin2.shp')
# shp = ShapelyFeature(Reader(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\Admin2.shp').geometries(),ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.6)
# ax.add_feature(shp)
# levels=np.arange(-0.1,0.15,0.025)
# mp = ax.contourf(lons,lats,sst_rate, cmap='coolwarm',origin='lower',levels=levels,alpha=0.95)
# ax.set_title('Trend', fontweight='bold',fontsize=18)
# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks(np.arange(min(lons), max(lons) , 5, dtype=int))
# ax.set_yticks(np.arange(min(lats), max(lats), 5, dtype=int))
# ax.set_xlim([68.7,97.25])
# ax.set_ylim([8.4,37.6])
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# gl1 = ax.gridlines(draw_labels=False, edgecolor='white', alpha=0.1)
# gl1.top_labels = False
# gl1.right_labels = False
# cbar = fig.colorbar(mp, ax=ax,shrink=0.6)
# cbar.minorticks_on()
# cbar.set_label('ΔT °C/year',fontsize=14, fontweight='bold')

# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams['xtick.labelsize']=14
# plt.rcParams['ytick.labelsize']=14
# plt.rcParams['xtick.major.size'] = 8 
# plt.rcParams['ytick.major.size'] = 8  
# plt.rcParams['xtick.major.width'] = 4
# plt.rcParams['ytick.major.width'] = 4 
# plt.rcParams["axes.linewidth"] = 4
# plt.rcParams["patch.linewidth"] = 4
# # plt.savefig("/home/lakshman/HW/heatsress/utci_trend.png",dpi=300)
# plt.show()
# import numpy as np
# from cartopy.io import shapereader as shpreader
# from cartopy.feature import ShapelyFeature
# from cartopy.io.shapereader import Reader
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# from netCDF4 import Dataset as netcdf
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt

# crs = ccrs.PlateCarree()
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 20), constrained_layout=False,
#                        subplot_kw={'projection': crs})

# ncset = netcdf(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\heat_stress\utci_daily.nc')
# lons = ncset['lon'][:]
# lats = ncset['lat'][:]

# sst1 = ncset['utci'][0:1342, :, :]
# sst2 = ncset['utci'][1343:2562, :, :]
# sst3 = ncset['utci'][2563:3782, :, :]

# nctime1 = ncset['time'][0:1342]
# nctime2 = ncset['time'][1343:2562]
# nctime3 = ncset['time'][2563:3782]

# t_unit = ncset['time'].units

# try:
#     t_cal = ncset['time'].calendar
# except AttributeError:
#     t_cal = "gregorian"

# nt1, nlat1, nlon1 = sst1.shape
# nt2, nlat2, nlon2 = sst2.shape
# nt3, nlat3, nlon3 = sst3.shape

# ngrd = nlon1 * nlat1

# sst1_grd = sst1.reshape((nt1, ngrd), order='F')
# sst2_grd = sst2.reshape((nt2, ngrd), order='F')
# sst3_grd = sst3.reshape((nt3, ngrd), order='F')

# x1 = np.linspace(1, nt1, nt1)
# x2 = np.linspace(1, nt2, nt2)
# x3 = np.linspace(1, nt3, nt3)

# sst_rate1 = np.empty((ngrd, 1))
# sst_rate1[:, :] = np.nan
# sst_rate2 = np.empty((ngrd, 1))
# sst_rate2[:, :] = np.nan
# sst_rate3 = np.empty((ngrd, 1))
# sst_rate3[:, :] = np.nan

# for i in range(ngrd):
#     y1 = sst1_grd[:, i]
#     y2 = sst2_grd[:, i]
#     y3 = sst3_grd[:, i]

#     if not np.ma.is_masked(y1):
#         z1 = np.polyfit(x1, y1, 1)
#         sst_rate1[i, 0] = z1[0] * 122.0

#     if not np.ma.is_masked(y2):
#         z2 = np.polyfit(x2, y2, 1)
#         sst_rate2[i, 0] = z2[0] * 122.0

#     if not np.ma.is_masked(y3):
#         z3 = np.polyfit(x3, y3, 1)
#         sst_rate3[i, 0] = z3[0] * 122.0

# sst_rate1 = sst_rate1.reshape((nlat1, nlon1), order='F')
# sst_rate2 = sst_rate2.reshape((nlat2, nlon2), order='F')
# sst_rate3 = sst_rate3.reshape((nlat3, nlon3), order='F')

# sst_rate1 = sst_rate1.astype('float')
# sst_rate2 = sst_rate2.astype('float')
# sst_rate3 = sst_rate3.astype('float')

# sst_rate1[sst_rate1 == 0] = np.nan
# sst_rate2[sst_rate2 == 0] = np.nan
# sst_rate3[sst_rate3 == 0] = np.nan

# sstr = [sst_rate1, sst_rate2, sst_rate3]
# titles = ['trend in 1990-2000', 'trend in 2001-2010', 'trend in 2011-2020']

# fname = shpreader.Reader(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\Admin2.shp')
# shp = ShapelyFeature(Reader(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\Admin2.shp').geometries(),
#                      ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.6)
# ax.add_feature(shp)

# levels = np.arange(-1, 1.15, 0.025)
# mp = ax.contourf(lons, lats, sstr, cmap='coolwarm', origin='lower', levels=levels, alpha=0.95)
# ax.set_title('titles', fontweight='bold', fontsize=18)

# lon_formatter = LongitudeFormatter(zero_direction_label=True)
# lat_formatter = LatitudeFormatter()
# ax.xaxis.set_major_formatter(lon_formatter)
# ax.yaxis.set_major_formatter(lat_formatter)
# ax.set_xticks(np.arange(min(lons), max(lons), 5, dtype=int))
# ax.set_yticks(np.arange(min(lats), max(lats), 5, dtype=int))
# ax.set_xlim([68.7, 97.25])
# ax.set_ylim([8.4, 37.6])

# gl1 = ax.gridlines(draw_labels=False, edgecolor='white', alpha=0.1)
# gl1.top_labels = False
# gl1.right_labels = False

# cbar = fig.colorbar(mp, ax=ax, shrink=0.6)
# cbar.minorticks_on()
# cbar.set_label('ΔT °C/year', fontsize=14, fontweight='bold')

# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams['xtick.labelsize'] = 14
# plt.rcParams['ytick.labelsize'] = 14
# plt.rcParams['xtick.major.size'] = 8
# plt.rcParams['ytick.major.size'] = 8
# plt.rcParams['xtick.major.width'] = 4
# plt.rcParams['ytick.major.width'] = 4
# plt.rcParams["axes.linewidth"] = 4
# plt.rcParams["patch.linewidth"] = 4

# # plt.savefig("/home/lakshman/HW/heatsress/utci_trend.png", dpi=300)
# plt.show()
# import numpy as np
# from cartopy.io import shapereader as shpreader
# from cartopy.feature import ShapelyFeature
# from cartopy.io.shapereader import Reader
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# from netCDF4 import Dataset as netcdf
# import cartopy.crs as ccrs
# import matplotlib.pyplot as plt
# from scipy.stats import ttest_1samp

# crs = ccrs.PlateCarree()
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 8), constrained_layout=False,
#                           subplot_kw={'projection': crs})

# ncset = netcdf(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\heat_stress\utci_daily.nc')
# lons = ncset['lon'][:]
# lats = ncset['lat'][:]

# sst1 = ncset['utci'][0:1342, :, :]
# sst2 = ncset['utci'][1343:2562, :, :]
# sst3 = ncset['utci'][2563:3782, :, :]

# nctime1 = ncset['time'][0:1342]
# nctime2 = ncset['time'][1343:2562]
# nctime3 = ncset['time'][2563:3782]

# t_unit = ncset['time'].units

# try:
#     t_cal = ncset['time'].calendar
# except AttributeError:
#     t_cal = "gregorian"

# nt1, nlat1, nlon1 = sst1.shape
# nt2, nlat2, nlon2 = sst2.shape
# nt3, nlat3, nlon3 = sst3.shape

# ngrd = nlon1 * nlat1

# sst1_grd = sst1.reshape((nt1, ngrd), order='F')
# sst2_grd = sst2.reshape((nt2, ngrd), order='F')
# sst3_grd = sst3.reshape((nt3, ngrd), order='F')

# x1 = np.linspace(1, nt1, nt1)
# x2 = np.linspace(1, nt2, nt2)
# x3 = np.linspace(1, nt3, nt3)

# sst_rate1 = np.empty((ngrd, 1))
# sst_rate1[:, :] = np.nan
# sst_rate2 = np.empty((ngrd, 1))
# sst_rate2[:, :] = np.nan
# sst_rate3 = np.empty((ngrd, 1))
# sst_rate3[:, :] = np.nan

# for i in range(ngrd):
#     y1 = sst1_grd[:, i]
#     y2 = sst2_grd[:, i]
#     y3 = sst3_grd[:, i]

#     if not np.ma.is_masked(y1):
#         z1 = np.polyfit(x1, y1, 1)
#         sst_rate1[i, 0] = z1[0] * 122.0

#     if not np.ma.is_masked(y2):
#         z2 = np.polyfit(x2, y2, 1)
#         sst_rate2[i, 0] = z2[0] * 122.0

#     if not np.ma.is_masked(y3):
#         z3 = np.polyfit(x3, y3, 1)
#         sst_rate3[i, 0] = z3[0] * 122.0
# for i in range(ngrd):
#     y1 = sst1_grd[:, i]
#     y2 = sst2_grd[:, i]
#     y3 = sst3_grd[:, i]

#     if not np.ma.is_masked(y1):
#         z1 = np.polyfit(x1, y1, 1)
#         slope1 = z1[0]
#         p_value1 = ttest_1samp(y1 - slope1 * x1, 0).pvalue
#         if p_value1 < 0.05:
#             sst_rate1[i, 0] = slope1 * 122.0  # Convert slope to annual change
#         else:
#             sst_rate1[i, 0] = np.nan

#     if not np.ma.is_masked(y2):
#         z2 = np.polyfit(x2, y2, 1)
#         slope2 = z2[0]
#         p_value2 = ttest_1samp(y2 - slope2 * x2, 0).pvalue
#         if p_value2 < 0.05:
#             sst_rate2[i, 0] = slope2 * 122.0  # Convert slope to annual change
#         else:
#             sst_rate2[i, 0] = np.nan

#     if not np.ma.is_masked(y3):
#         z3 = np.polyfit(x3, y3, 1)
#         slope3 = z3[0]
#         p_value3 = ttest_1samp(y3 - slope3 * x3, 0).pvalue
#         if p_value3 < 0.05:
#             sst_rate3[i, 0] = slope3 * 122.0  # Convert slope to annual change
#         else:
#             sst_rate3[i, 0] = np.nan

# sst_rate1 = sst_rate1.reshape((nlat1, nlon1), order='F')
# sst_rate2 = sst_rate2.reshape((nlat2, nlon2), order='F')
# sst_rate3 = sst_rate3.reshape((nlat3, nlon3), order='F')

# sst_rate1 = sst_rate1.astype('float')
# sst_rate2 = sst_rate2.astype('float')
# sst_rate3 = sst_rate3.astype('float')

# sst_rate1[sst_rate1 == 0] = np.nan
# sst_rate2[sst_rate2 == 0] = np.nan
# sst_rate3[sst_rate3 == 0] = np.nan

# sstr = [sst_rate1, sst_rate2, sst_rate3]
# titles = ['Trend in 1990-2000', 'Trend in 2001-2010', 'Trend in 2011-2020']

# fname = shpreader.Reader(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\Admin2.shp')
# shp = ShapelyFeature(Reader(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\Admin2.shp').geometries(),
#                       ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.6)

# levels = np.arange(-1, 1.15, 0.025)

# for i, ax in enumerate(axes):
#     ax.add_feature(shp)
#     mp = ax.contourf(lons, lats, sstr[i], cmap='coolwarm', origin='lower', levels=levels, alpha=0.95)
#     ax.set_title(titles[i], fontweight='bold', fontsize=18)

#     lon_formatter = LongitudeFormatter(zero_direction_label=True)
#     lat_formatter = LatitudeFormatter()
#     ax.xaxis.set_major_formatter(lon_formatter)
#     ax.yaxis.set_major_formatter(lat_formatter)
#     ax.set_xticks(np.arange(min(lons), max(lons), 5, dtype=int))
#     ax.set_yticks(np.arange(min(lats), max(lats), 5, dtype=int))
#     ax.set_xlim([68.7, 97.25])
#     ax.set_ylim([8.4, 37.6])

#     gl1 = ax.gridlines(draw_labels=False, edgecolor='white', alpha=0.1)
#     gl1.top_labels = False
#     gl1.right_labels = False

# cbar = fig.colorbar(mp, ax=axes, shrink=0.6)
# cbar.minorticks_on()
# cbar.set_label('ΔT °C/year', fontsize=14, fontweight='bold')

# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams['xtick.labelsize'] = 14
# plt.rcParams['ytick.labelsize'] = 14
# plt.rcParams['xtick.major.size'] = 8
# plt.rcParams['ytick.major.size'] = 8
# plt.rcParams['xtick.major.width'] = 4
# plt.rcParams['ytick.major.width'] = 4
# plt.rcParams["axes.linewidth"] = 4
# plt.rcParams["patch.linewidth"] = 4

# # plt.savefig("/home/lakshman/HW/heatsress/utci_trend.png", dpi=300)
# plt.show()
import numpy as np
from cartopy.io import shapereader as shpreader
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from netCDF4 import Dataset as netcdf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

crs = ccrs.PlateCarree()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 8), constrained_layout=False,
                         subplot_kw={'projection': crs})

ncset = netcdf(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\heat_stress\utci_daily.nc')
lons = ncset['lon'][:]
lats = ncset['lat'][:]

sst1 = ncset['utci'][0:1342, :, :]
sst2 = ncset['utci'][1343:2562, :, :]
sst3 = ncset['utci'][2563:3782, :, :]

nctime1 = ncset['time'][0:1342]
nctime2 = ncset['time'][1343:2562]
nctime3 = ncset['time'][2563:3782]

t_unit = ncset['time'].units

try:
    t_cal = ncset['time'].calendar
except AttributeError:
    t_cal = "gregorian"

nt1, nlat1, nlon1 = sst1.shape
nt2, nlat2, nlon2 = sst2.shape
nt3, nlat3, nlon3 = sst3.shape

ngrd = nlon1 * nlat1

sst1_grd = sst1.reshape((nt1, ngrd), order='F')
sst2_grd = sst2.reshape((nt2, ngrd), order='F')
sst3_grd = sst3.reshape((nt3, ngrd), order='F')

x1 = np.linspace(1, nt1, nt1)
x2 = np.linspace(1, nt2, nt2)
x3 = np.linspace(1, nt3, nt3)

sst_rate1 = np.empty((ngrd, 1))
sst_rate1[:, :] = np.nan
sst_rate2 = np.empty((ngrd, 1))
sst_rate2[:, :] = np.nan
sst_rate3 = np.empty((ngrd, 1))
sst_rate3[:, :] = np.nan

significant_areas1 = []
significant_areas2 = []
significant_areas3 = []

for i in range(ngrd):
    y1 = sst1_grd[:, i]
    y2 = sst2_grd[:, i]
    y3 = sst3_grd[:, i]

    if not np.ma.is_masked(y1):
        z1 = np.polyfit(x1, y1, 1)
        slope1 = z1[0]
        p_value1 = ttest_1samp(y1 - slope1 * x1, 0).pvalue
        if p_value1 < 0.25:  # 90% confidence level
            sst_rate1[i, 0] = slope1 * 122.0  # Convert slope to annual change
            significant_areas1.append((i % nlon1, i // nlon1))
        else:
            sst_rate1[i, 0] = np.nan

    if not np.ma.is_masked(y2):
        z2 = np.polyfit(x2, y2, 1)
        slope2 = z2[0]
        p_value2 = ttest_1samp(y2 - slope2 * x2, 0).pvalue
        if p_value2 < 0.25:  # 90% confidence level
            sst_rate2[i, 0] = slope2 * 122.0  # Convert slope to annual change
            significant_areas2.append((i % nlon2, i // nlon2))
        else:
            sst_rate2[i, 0] = np.nan

    if not np.ma.is_masked(y3):
        z3 = np.polyfit(x3, y3, 1)
        slope3 = z3[0]
        p_value3 = ttest_1samp(y3 - slope3 * x3, 0).pvalue
        if p_value3 < 0.25:  # 90% confidence level
            sst_rate3[i, 0] = slope3 * 122.0  # Convert slope to annual change
            significant_areas3.append((i % nlon3, i // nlon3))
        else:
            sst_rate3[i, 0] = np.nan

sst_rate1 = sst_rate1.reshape((nlat1, nlon1), order='F')
sst_rate2 = sst_rate2.reshape((nlat2, nlon2), order='F')
sst_rate3 = sst_rate3.reshape((nlat3, nlon3), order='F')

sst_rate1 = sst_rate1.astype('float')
sst_rate2 = sst_rate2.astype('float')
sst_rate3 = sst_rate3.astype('float')

sst_rate1[sst_rate1 == 0] = np.nan
sst_rate2[sst_rate2 == 0] = np.nan
sst_rate3[sst_rate3 == 0] = np.nan

sstr = [sst_rate1, sst_rate2, sst_rate3]
titles = ['Trend in 1990-2000', 'Trend in 2001-2010', 'Trend in 2011-2020']

fname = shpreader.Reader(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\Admin2.shp')
shp = ShapelyFeature(Reader(r'C:\Users\S Narayanaswami\OneDrive\Desktop\summer\Admin2.shp').geometries(),
                     ccrs.PlateCarree(), edgecolor='k', facecolor='none', linewidth=0.6)

levels = np.arange(-1, 1.15, 0.025)

for i, ax in enumerate(axes):
    ax.add_feature(shp)
    mp = ax.contourf(lons, lats, sstr[i], cmap='coolwarm', origin='lower', levels=levels, alpha=0.95)
    ax.set_title(titles[i], fontweight='bold', fontsize=18)

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_xticks(np.arange(min(lons), max(lons), 5, dtype=int))
    ax.set_yticks(np.arange(min(lats), max(lats), 5, dtype=int))
    ax.set_xlim([68.7, 97.25])
    ax.set_ylim([8.4, 37.6])

    # # Draw white circles around significant areas
    # if i == 0:
    #     significant_areas = significant_areas1
    # elif i == 1:
    #     significant_areas = significant_areas2
    # else:
    #     significant_areas = significant_areas3

    # for lon_idx, lat_idx in significant_areas:
    #     ax.plot(lons[lon_idx], lats[lat_idx], 'o', markerfacecolor='none', markeredgecolor='k', markersize=6, markeredgewidth=2)

    gl1 = ax.gridlines(draw_labels=False, edgecolor='white', alpha=0.1)
    gl1.top_labels = False
    gl1.right_labels = False

cbar = fig.colorbar(mp, ax=axes, shrink=0.6)
cbar.minorticks_on()
cbar.set_label('ΔT °C/year', fontsize=14, fontweight='bold')

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 4
plt.rcParams['ytick.major.width'] = 4
plt.rcParams["axes.linewidth"] = 4
plt.rcParams["patch.linewidth"] = 4

plt.savefig("C:/Users/S Narayanaswami/OneDrive/Desktop/summer/sumr_work_fig/utci_trend.png", dpi=300)
plt.show()
