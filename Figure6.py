#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:26:26 2026

@author: cwuchiu
"""
#%%
from numpy import linspace,nan,meshgrid, array, squeeze, where, pi, stack,zeros, loadtxt, arcsin
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import netCDF4 as nc
from wrf import (getvar,interplevel)
import pandas as pd
import wrf
import numpy as np
from scipy.integrate import dblquad, quad
import xarray as xr
import math
from scipy.stats import linregress
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
from collections import Counter
import matplotlib.colors as colors
from matplotlib import cm
import scipy.stats as st
from scipy.stats import gaussian_kde
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors
from scipy import stats

ncfile = nc.Dataset('/Dellwork5/cwuchiu/IBTrACS.WP.v04r01.nc')

ar = xr.open_dataset('/Dellwork5/cwuchiu/IBTrACS.WP.v04r01.nc')


lon = array(ncfile.variables['usa_lon'])
lat = array(ncfile.variables['usa_lat'])
year = array(ncfile.variables['season'])
number = array(ncfile.variables['number'])
usa_wind = array(ncfile.variables['usa_wind'])

source_usa = array(ncfile.variables['source_usa'])


nature = ar['nature'].data  # b'TS' = 熱帶, b'ET' = 溫帶, b'SS' = 亞熱帶
sid = ar['sid'].data
time = ar['iso_time'].data
name = ar['name'].data
basin = ar['basin'].data
basin1 = basin[:,0]
track_type = ar['track_type'].data
storm_dir = ar['storm_dir'].data
ncfile.close()

save_lon = []
save_lat = []
save_time = []
save_wind = []


JJASON = np.zeros(4187)
for i in range(JJASON.shape[0]):
    if all(6 <= int(t[5:7]) <= 11 for t in time[i, :] if t):
        JJASON[i] = 1
    else:
        JJASON[i] = 0

for j in range(1979,2025):
    ab = []
    ba = []
    ccc=[]
    www=[]
    x_lon = []
    x_lat = []
    max_wind = np.nanmax(usa_wind, axis=1)
    if j==2024:
        #tcmask = np.where((year==j) & (basin1==b'WP') & (track_type==b'PROVISIONAL'))
        tcmask = np.where((year==j) & (basin1==b'WP') & (track_type==b'PROVISIONAL') & (JJASON==1) & (max_wind >= 34))
    else:
        #tcmask = np.where((year==j) & (basin1==b'WP') & (track_type==b'main'))
        tcmask = np.where((year==j) & (basin1==b'WP') & (track_type==b'main') & (JJASON==1) & (max_wind >= 34))

    lon_tc = lon[tcmask]
    lat_tc = lat[tcmask]
    name_tc = name[tcmask]
    time_tc = time[tcmask]
    number_tc = number[tcmask]
    basin_tc = basin[tcmask]
    storm_dir_tc = storm_dir[tcmask]
    usa_wind_tc = usa_wind[tcmask]
    nature_tc = nature[tcmask]
    recurving = 0
    
    
    plt.figure(figsize = (10,6))
    x_extent = linspace(-180,180,37)
    y_extent = linspace(-90,90,19)
    ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.coastlines(resolution='50m',alpha=0.5)
    ax.set_xticks(x_extent, crs=ccrs.PlateCarree())
    ax.set_yticks(y_extent, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_extent([100,180,0,50],crs = ccrs.PlateCarree())
    
    
    #for i in range(21,22):
    for i in range(0,len(number_tc)):
        mask_pt = (lat_tc[i, :] != -9999) & (lon_tc[i, :] >= 100) & (lat_tc[i, :] >= 0) & (nature_tc[i, :] != b'ET')
        tclat = lat_tc[i, mask_pt]
        tclon = lon_tc[i, mask_pt]
        tcdir = storm_dir_tc[i, mask_pt]
        tct = time_tc[i, mask_pt]
        tcwind = usa_wind_tc[i, mask_pt]

        ab.append(tclon)
        ba.append(tclat)
        ccc.append(tct)
        www.append(tcwind)
        
        ax.plot(tclon, tclat, 'k',linewidth=1,transform=ccrs.PlateCarree(),alpha=1)
        cb = ax.scatter (tclon, tclat, s=5, transform=ccrs.PlateCarree())
    save_lon.append(ab)
    save_lat.append(ba)
    save_time.append(ccc)
    save_wind.append(www)

    ax.set_title('TCs in the Western North Pacific '+str(j)+' '+str(len(number_tc))+' TCs', fontsize=20, loc='center')
    plt.savefig('/Dellwork5/cwuchiu/SH/best_track_original/'+str(j)+'.png')
    plt.close()

for i in range(46):
    save_lon[i] = [track for track in save_lon[i] if len(track) > 0]
    save_lat[i] = [track for track in save_lat[i] if len(track) > 0] 
    save_time[i] = [track for track in save_time[i] if len(track) > 0] 
    save_wind[i] = [track for track in save_wind[i] if len(track) > 0] 




time = np.linspace(1979,2024,46)
ncfile = nc.Dataset('/Dellwork4/cwuchiu/Cluster4_WNPSH_centroid_Ridge_SHAI_noET_34kt.nc')
WNPSH_possibility = np.array(ncfile.variables['hgt_adjust_cluster']) 
cluster_case_number = np.array(ncfile.variables['cluster_case_number']) 
which_cluster = np.array(ncfile.variables['which_cluster']) 
ncfile.close()


ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_1979_2024_not_set_noET_34kt.nc')
tt = np.array(ncfile.variables['time'])
time_tc = [str(x) for x in tt]
ll = np.array(ncfile.variables['lon'])
la = np.array(ncfile.variables['lat'])

ncfile.close()


cluster = np.zeros((5,46))
cluster_month = np.zeros((5,6))
for i in range(len(time_tc)):
    cluster[int(which_cluster[i]),int(time_tc[i][0:4])-1979] += 1
    cluster_month[int(which_cluster[i]),int(time_tc[i][4:6])-6] +=1

cluster_sum = np.sum(cluster,axis=0)




     
pos_lon = [[],[],[],[],[]]
pos_lat = [[],[],[],[],[]]


for i in range(len(time_tc)):
    pos_lon[int(which_cluster[i])].append(ll[i])
    pos_lat[int(which_cluster[i])].append(la[i])


def compute_track_density(save_lon, save_lat,
                                    lon_min=100, lon_max=180,
                                    lat_min=0, lat_max=50,
                                    dlon=1.0, dlat=1.0):

    lon_bins = np.arange(lon_min, lon_max + dlon, dlon)
    lat_bins = np.arange(lat_min, lat_max + dlat, dlat)

    n_cluster = len(save_lon)   # = 4
    density_all = np.zeros((n_cluster,
                             len(lat_bins)-1,
                             len(lon_bins)-1), dtype=np.float64)
    save_total_point = np.zeros(6)
    for c in range(n_cluster):   # 每一群
        density = np.zeros((len(lat_bins)-1, len(lon_bins)-1))

        total_points = 0

        for ty_lons, ty_lats in zip(save_lon[c], save_lat[c]):  # 每個颱風
            ty_lons = np.array(ty_lons)
            ty_lats = np.array(ty_lats)

            valid = (~np.isnan(ty_lons)) & (~np.isnan(ty_lats))
            ty_lons = ty_lons[valid]
            ty_lats = ty_lats[valid]

            total_points += len(ty_lons)

            lon_idx = np.digitize(ty_lons, lon_bins) - 1
            lat_idx = np.digitize(ty_lats, lat_bins) - 1

            valid = (lon_idx >= 0) & (lon_idx < len(lon_bins)-1) & \
                    (lat_idx >= 0) & (lat_idx < len(lat_bins)-1)

            lon_idx = lon_idx[valid]
            lat_idx = lat_idx[valid]

            for i, j in zip(lat_idx, lon_idx):
                density[i, j] += 1

        # 正規化（依你原本的想法，用總軌跡點數）
        if total_points > 0:
            density /= total_points
        save_total_point[c] = total_points
        density_all[c] = density

    return density_all, lon_bins, lat_bins, save_total_point




pos_lon_all = []
pos_lat_all = []

for i in range(len(pos_lon)):
    pos_lon_all.extend(pos_lon[i])
    pos_lat_all.extend(pos_lat[i])

pos_lon_all = [pos_lon_all]
pos_lat_all = [pos_lat_all]
density1, lon_bins, lat_bins, save_total_point = compute_track_density(pos_lon, pos_lat)

density2, _, _, _ = compute_track_density(pos_lon_all, pos_lat_all)

for i in range(5):
    density1[i] /= np.sum(density1[i])/1
density2 /= np.sum(density2)/1

lon_bins = np.linspace(100, 180, 81)  # 示例
lat_bins = np.linspace(0, 50, 51)

lon_grid, lat_grid = np.meshgrid(lon_bins[:-1], lat_bins[:-1])  # shape: (20, 32)

data_diff = density1 - density2


# KDE 計算函數
def compute_kde(lon, lat, mask, weights):
    x = lon[mask]
    y = lat[mask]
    if len(x) == 0:
        return np.zeros_like(lon)
    
    # 取出對應的權重
    w = weights[mask] if weights is not None else None
    
    coords = np.vstack([x, y])
    # 將權重傳入 gaussian_kde
    kernel = gaussian_kde(coords, bw_method='silverman', weights=w)
    
    positions = np.vstack([lon.ravel(), lat.ravel()])
    z = np.reshape(kernel(positions), lon.shape)
    return z


slope_cluster = np.zeros(5)
intercept_cluster = np.zeros(5)

for i in range(5):
    ratio = cluster[i] / cluster_sum
    slope_cluster[i], intercept_cluster[i], r_value, p_value, std_err = stats.linregress(time, ratio)


delta_ratio = slope_cluster * (2024 - 1979) 

d_total = np.sum(data_diff * delta_ratio[:, None, None], axis=0)


# ======= 繪圖 =======
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 5))

# 地圖格式
ax.coastlines(resolution='50m', linewidth=1)
ax.set_xticks(linspace(-180, 180, 37), crs=ccrs.PlateCarree())
ax.set_yticks(linspace(-90, 90, 19), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.set_extent([100,180,0,50],crs = ccrs.PlateCarree())

z_increase = compute_kde(lon_grid, lat_grid, d_total > 0, weights=d_total)
z_decrease = -compute_kde(lon_grid, lat_grid, d_total < 0, weights=np.abs(d_total))
z_combined = z_increase + z_decrease

norm = TwoSlopeNorm(vmin=-0.001,vcenter=0,vmax=0.001)
custom_bwr = mcolors.LinearSegmentedColormap.from_list(
'custom_bwr', [(0, 'blue'), (0.48, 'white'), (0.52, 'white'), (1, 'red')]
)
ct = ax.pcolormesh(
lon_grid, lat_grid, z_combined,
cmap=custom_bwr,
norm=norm,
transform=ccrs.PlateCarree(),
alpha=0.7
)

plt.colorbar(ct, ax=ax, ticks=np.linspace(-0.001,0.001,11), orientation='vertical', label='KDE')



lon = np.linspace(100,180,321)
lat = np.linspace(50,0,201)

levels = np.linspace(0.1,1,10)
linestyles = ['dashed' if lev < 0 else 'solid' for lev in levels]

# 加上黑色等高線
contour_lines = ax.contour(
    lon_grid, lat_grid, z_combined,
    levels=np.linspace(-0.001,0.001,11),
    colors='black',
    linewidths=0.4,
    transform=ccrs.PlateCarree()
)
ax.clabel(contour_lines, inline=True, fontsize=7, fmt="%.4f")

plt.title(f'TC track density contributed by WNPSH type change',fontweight='bold')
plt.tight_layout()
plt.savefig(f'Figure6.png')
plt.show()
