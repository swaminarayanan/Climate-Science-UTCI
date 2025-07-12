# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
import netCDF4 as nc
from wrf import (getvar,to_np,get_cartopy,latlon_coords,vertcross,
                  cartopy_xlim,cartopy_ylim,interpline,CoordPair,ALL_TIMES)
import wrf

df1 = nc.Dataset(r'/media/lab/One Touch/Krishan_Kishore/JEL_SIP/kol_cntl/wrfout_d03_2017-05-17_00%3A00%3A00')
df2  = nc.Dataset(r'/media/lab/One Touch/Krishan_Kishore/JEL_SIP/kol_cntl/wrfout_d03_2018-05-13_00%3A00%3A00')
df3= nc.Dataset(r'/media/lab/One Touch/Krishan_Kishore/JEL_SIP/kol_cntl/wrfout_d03_2020-04-23_00%3A00%3A00')



df_rh1= getvar(df1,"rh2",timeidx=44)
df_temp1=getvar(df1,"T2",timeidx=44)

df_rh2= getvar(df2,"rh2",timeidx=48)
df_temp2=getvar(df2,"T2",timeidx=48)

df_rh3= getvar(df3,"rh2",timeidx=34)
df_temp3=getvar(df3,"T2",timeidx=34)


################ plotting ############
# Meshgridding for Model data
lats=df1.variables['XLAT'][0,:,0]
longs=df1.variables['XLONG'][0,0,:]
X,Y=np.meshgrid(longs,lats)





fig, ax = plt.subplots(1,3,figsize=(18,6), gridspec_kw={'wspace': 0.001},facecolor='w',edgecolor='k',subplot_kw={'projection':
                                    ccrs.PlateCarree()})  
 
#fig.tight_layout()
#levels=[290,295,300,305,310,315] #for temp
levels=[40,50,60,70,80,90]
axlist = np.ravel(ax)
ax=axlist[0]
# plot=ax.contourf(X,Y,df_temp1,cmap='bwr',extend='both', transform=ccrs.PlateCarree(),)
# ax.set_title('2mTemp_2017051700', loc='center',fontsize=20)
plot=ax.contourf(X,Y,df_rh1,cmap='YlGnBu',levels=levels,extend='both', transform=ccrs.PlateCarree(),)
ax.set_title('2mRH_2017051700', loc='center',fontsize=20)
ax.set_yticks([22,22.5,23,23.5])#Kolkata
ax.set_xticks([87.5,88,88.5,89])#Kolkata
ax.spines['top'].set_linewidth(10)


ax=axlist[1]
plot=ax.contourf(X,Y,df_rh2,cmap='YlGnBu',levels=levels,extend='both', transform=ccrs.PlateCarree())
ax.set_title('2mRH_2018051300', loc='center',fontsize=20)
# plot=ax.contourf(X,Y,df_temp2,cmap='bwr',extend='both', transform=ccrs.PlateCarree(),)
# ax.set_title('2mTemp_2018051300', loc='center',fontsize=20)
#ax.set_xticks([87.5,88,88.5,89])
ax.set_xticks([87.5,88,88.5,89])#Kolkata


ax=axlist[2]
plot=ax.contourf(X,Y,df_rh3,cmap='YlGnBu',extend='both', transform=ccrs.PlateCarree())
ax.set_title('2mRH_2020042300', loc='center',fontsize=20)
# plot=ax.contourf(X,Y,df_temp3,cmap='bwr',extend='both', transform=ccrs.PlateCarree())
# ax.set_title('2mTemp_2020042300', loc='center',fontsize=20)
#ax.set_xticks([87.5,88,88.5,89])
ax.set_xticks([87.5,88,88.5,89])#Kolkata



#ax.set_title('2m_Relative_Humidity', loc='center',fontsize=20)

   
axbar=[axlist[0],axlist[1],axlist[2]]
cbar=plt.colorbar(plot,
                      ax=axbar,
                      #ticks=np.arange(-40, 40.1, 10),
                      
                      orientation='vertical',
                      shrink=0.4,
                      pad=0.01,
                      #extendfrac='auto'
                      )               
cbar.ax.tick_params(labelsize=20)


axbar=[axlist[0],axlist[1],axlist[2]]

for ax in axlist:
    ax.set_extent([87.197258, 89.13791656, 21.70922852 , 23.50069427], crs=ccrs.PlateCarree())
    #ax.coastlines(resolution="50m",linewidth=1)
    #ax.add_feature(cfeature.NaturalEarthFeature('physical', 'ocean', '50m', edgecolor='k', facecolor='grey'))
    gl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                   linewidth=0, color='black', alpha=0.5, linestyle='--')
    gl.top_labels=False
    gl.right_labels=False
    gl.bottom_labels=False
    gl.left_labels=False
    
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    # ax.spines['right'].set_linewidth(2)
    
   

 
plt.rcParams['axes.linewidth']=2
plt.rcParams['patch.linewidth']=2
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

plt.savefig("/home/lab/Desktop/JEL_SIP_IMD/plots/hum_compar.png", dpi=300)
plt.show()