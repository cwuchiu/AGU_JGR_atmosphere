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


def compute_track_density(save_lon_subset, save_lat_subset, abc,
                          lon_min=100, lon_max=180, lat_min=0, lat_max=50, dlon=1.0, dlat=1.0):
    iii=0
    kkk=0
    lon_bins = np.arange(lon_min, lon_max + dlon, dlon)
    lat_bins = np.arange(lat_min, lat_max + dlat, dlat)
    density = np.zeros((abc, len(lat_bins)-1, len(lon_bins)-1), dtype=np.float64)

    for year_lons, year_lats in zip(save_lon_subset, save_lat_subset):
        for ty_lons, ty_lats in zip(year_lons, year_lats):  # 每個颱風
            ty_lons = np.array(ty_lons)
            ty_lats = np.array(ty_lats)
            valid = (~np.isnan(ty_lons)) & (~np.isnan(ty_lats))
            ty_lons = ty_lons[valid]
            ty_lats = ty_lats[valid]

            lon_idx = np.digitize(ty_lons, lon_bins) - 1
            lat_idx = np.digitize(ty_lats, lat_bins) - 1

            valid = (lon_idx >= 0) & (lon_idx < 80) & \
                    (lat_idx >= 0) & (lat_idx <50)
            lon_idx = lon_idx[valid]
            lat_idx = lat_idx[valid]

            for i, j in zip(lat_idx, lon_idx):
                density[iii, i, j] += 1
            
            kkk+=1
            
            if kkk==int(len(save_lon_subset[iii])):
                countt=0
                print(kkk)
                kkk=0
                for aaa in range(len(save_lon_subset[iii])):
                    countt+=len(save_lon_subset[iii][aaa])
                
                density[iii, :, :] = density[iii, :, :]/countt
                
                iii+=1
            else:    
                pass
                #kkk+=1
            
    return density, lon_bins, lat_bins




label = ['1979-2024']
for i in range(1):
    yr_num = 46
    # 計算前期 (1997~2010)
    density1, lon_bins, lat_bins = compute_track_density(save_lon[46-yr_num:], save_lat[46-yr_num:],yr_num)

    
    lon_bins = np.linspace(100, 180, 81)  # 示例
    lat_bins = np.linspace(0, 50, 51)
    
    lon_grid, lat_grid = np.meshgrid(lon_bins[:-1], lat_bins[:-1])  # shape: (20, 32)
    
    # 計算差值 (正值: 增加, 負值: 減少)
    time = np.linspace(1979,2024,yr_num)
    time_mean = np.mean(time)
    d1_mean = np.mean(density1,axis=0)
    numerator_direct = np.sum((time[:, None, None] - time_mean) * (density1 - d1_mean[None, :, :]), axis=0)
    denominator_direct = np.sum((time[:, None, None] - time_mean)**2, axis=0)

    data_diff = numerator_direct / denominator_direct
    
    
# =============================================================================
#     # KDE 計算函數
#     def compute_kde(lon, lat, mask):
#         x = lon[mask]
#         y = lat[mask]
#         if len(x) == 0:
#             return np.zeros_like(lon)
#         coords = np.vstack([x, y])
#         kernel = gaussian_kde(coords,bw_method='silverman')
#         positions = np.vstack([lon.ravel(), lat.ravel()])
#         z = np.reshape(kernel(positions), lon.shape)
#         return z
# =============================================================================
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
    # 分別計算增加與減少的 KDE
    z_increase = compute_kde(lon_grid, lat_grid, data_diff > 0, weights= data_diff)
    z_decrease = -compute_kde(lon_grid, lat_grid,  data_diff < 0, weights=np.abs(data_diff))
    
    # 合併成單一矩陣 (增加為正，減少為負)
    z_combined = z_increase + z_decrease
    
    # ======= 繪圖 =======
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 5))
    
    # 地圖格式
    ax.coastlines(resolution='50m', linewidth=1)
    ax.set_xticks(linspace(-180, 180, 37), crs=ccrs.PlateCarree())
    ax.set_yticks(linspace(-90, 90, 19), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_extent([100, 180, 0, 50])
    
    # 設定顏色範圍，零值置中
    #vmax = np.max(np.abs(z_combined))
    norm = TwoSlopeNorm(vmin=-0.001, vcenter=0, vmax=0.001)
    
    # 繪製藍-白-紅 KDE
    custom_bwr = LinearSegmentedColormap.from_list(
    'custom_bwr', [(0, 'blue'), (0.48, 'white'), (0.52, 'white'), (1, 'red')]
    )
    # 繪製藍-白-紅 KDE
    ct = ax.pcolormesh(
    lon_grid, lat_grid, z_combined,
    cmap=custom_bwr,
    norm=norm,
    transform=ccrs.PlateCarree(),
    alpha=0.7
    )
    
    # 加上黑色等高線
    contour_lines = ax.contour(
        lon_grid, lat_grid, z_combined,
        levels=np.linspace(-0.001,0.001,11),
        colors='black',
        linewidths=0.4,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.4f")
    
    # 標題與 colorbar
    if yr_num==28:
        ax.set_title('Kernel density of TC Track Change (Increase: Red, Decrease: Blue) \n 1997-2024', fontsize=14,fontweight='bold')
    else:
        ax.set_title('Kernel density of TC Track Change (Increase: Red, Decrease: Blue) \n '+label[i], fontsize=14,fontweight='bold')
    cbar = plt.colorbar(ct, ax=ax, ticks=np.linspace(-0.001,0.001,11),
                    orientation='vertical')

    cbar.set_label('KDE ' + label[i], fontsize=14,fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    ax.set_ylim(0, 50)
    plt.savefig('Figure1.png', dpi=400)
    plt.show()