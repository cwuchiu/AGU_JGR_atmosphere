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
#%%
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
#%%
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


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ==========================================
# 1. 資料轉換函數
# ==========================================
def extract_relative_tracks(save_lon, save_lat, year_start_idx, year_end_idx):
    """
    將指定年份區間的颱風軌跡轉換為相對座標 (相對於生成點)
    """
    rel_lons = [] # 儲存所有相對經度 (平面化)
    rel_lats = [] # 儲存所有相對緯度
    
    # 遍歷指定的年份
    for y in range(year_start_idx, year_end_idx):
        year_data_lon = save_lon[y]
        year_data_lat = save_lat[y]
        
        # 遍歷該年的每顆颱風
        for t in range(len(year_data_lon)):
            track_lon = np.array(year_data_lon[t])
            track_lat = np.array(year_data_lat[t])
            
            # 過濾掉空資料或過短的資料
            if len(track_lon) < 2:
                continue
                
            # 1. 抓出生成點 (Genesis Point) - 假設是第一個點
            # 需確保第一個點不是 NaN
            valid_idx = np.where(~np.isnan(track_lon) & ~np.isnan(track_lat))[0]
            if len(valid_idx) == 0:
                continue
                
            gen_lon = track_lon[valid_idx[0]]
            gen_lat = track_lat[valid_idx[0]]
            
            # 2. 計算相對座標 (Relative Coordinates)
            # dx = 當前經度 - 生成經度
            # dy = 當前緯度 - 生成緯度
            dx = track_lon - gen_lon
            dy = track_lat - gen_lat
            
            # 3. 存入列表 (過濾掉 NaN)
            mask = ~np.isnan(dx) & ~np.isnan(dy)
            rel_lons.extend(dx[mask])
            rel_lats.extend(dy[mask])
            
    return np.array(rel_lons), np.array(rel_lats)

# ==========================================
# 2. 計算密度圖 (2D Histogram)
# ==========================================
def compute_density(rel_x, rel_y, bins=50, range_x=(-30, 10), range_y=(-10, 20), sigma=1.5):
    """
    計算二維直方圖並進行高斯平滑
    range_x: 相對經度範圍 (例如生成點往西20度到往東40度)
    range_y: 相對緯度範圍
    sigma: 平滑係數 (讓圖好看一點)
    """
    # 計算 2D Histogram
    H, xedges, yedges = np.histogram2d(
        rel_x, rel_y, 
        bins=bins, 
        range=[range_x, range_y],
        density=True # 正規化，方便比較
    )
    
    # 轉置 H 以符合 imshow 的繪圖習慣 (x, y) -> (row, col)
    H = H.T
    
    # 高斯平滑 (Gaussian Smoothing) - 模擬 KDE 效果
    H_smooth = gaussian_filter(H, sigma=sigma)
    
    # 取得網格中心點 (用於 contourf)
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(xc, yc)
    
    return X, Y, H_smooth

# ==========================================
# 3. 主程式執行
# ==========================================
# 假設 split_year 是分隔點 (例如第 23 年)
split_idx = 23
total_years = 46

# 提取 Early Period (前 23 年)
rx_early, ry_early = extract_relative_tracks(save_lon, save_lat, 0, split_idx)

# 提取 Late Period (後 23 年)
rx_late, ry_late = extract_relative_tracks(save_lon, save_lat, split_idx, total_years)

# 計算密度
# 設定範圍：假設颱風大多往西往北，設定 x: -40~20 (往西為負), y: -10~50
# 注意：經度往西通常是減，所以相對經度若是負值代表往西行
bins = 60
x_range = (-30, 10)  # 生成點以西 40 度 ~ 以東 20 度
y_range = (-10, 20)  # 生成點以南 10 度 ~ 以北 50 度

X_early, Y_early, Z_early = compute_density(rx_early, ry_early, bins=bins, range_x=x_range, range_y=y_range)
X_late, Y_late, Z_late = compute_density(rx_late, ry_late, bins=bins, range_x=x_range, range_y=y_range)

# ==========================================
# 4. 繪圖比較
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(10, 10),layout='constrained',
    subplot_kw={'projection': ccrs.PlateCarree()})
levels = np.linspace(0, 0.02, 11)

# Early Plot
cf1 = axes[0].contourf(X_early, Y_early, Z_early, levels=levels, cmap='hot_r')
axes[0].set_title('(a) 1979-2001',fontweight='bold',fontsize=16)
axes[0].set_xlabel('Relative Longitude',fontsize=16)
axes[0].set_ylabel('Relative Latitude',fontsize=16)
axes[0].tick_params(axis='both', labelsize=16)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].set_extent([-20, 5, -5, 10])
# 標示生成點 (原點)
axes[0].plot(0, 0, 'bX', markersize=10, label='Genesis')

axes[0].set_xticks(np.arange(-20, 5+1, 10), crs=ccrs.PlateCarree())
axes[0].set_yticks(np.arange(-5, 10+1, 10), crs=ccrs.PlateCarree())
#axes[0].xaxis.set_major_formatter(LongitudeFormatter())
#axes[0].yaxis.set_major_formatter(LatitudeFormatter())

# Late Plot
cf2 = axes[1].contourf(X_late, Y_late, Z_late, levels=levels, cmap='hot_r')
axes[1].set_title('(b) 2002-2024',fontweight='bold',fontsize=16)
axes[1].set_xlabel('Relative Longitude',fontsize=16)
axes[1].tick_params(axis='both', labelsize=16)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].plot(0, 0, 'bX', markersize=10)
axes[1].set_extent([-20, 5, -5, 10])

axes[1].set_xticks(np.arange(-20, 5+1, 10), crs=ccrs.PlateCarree())
axes[1].set_yticks(np.arange(-5, 10+1, 10), crs=ccrs.PlateCarree())
#axes[1].xaxis.set_major_formatter(LongitudeFormatter())
#axes[1].yaxis.set_major_formatter(LatitudeFormatter())

# Difference Plot (Late - Early)
# 需重新計算統一網格的差異，這裡簡單用相減 (若 histogram 參數完全一樣，矩陣大小會一樣)
Z_diff = Z_late - Z_early
limit = np.max(np.abs(Z_diff)) * 0.8
levels_diff = np.linspace(-limit, limit, 6)



cbar = plt.colorbar(cf2, ax=axes[:2], location='bottom', label='Track Density',ticks=np.linspace(0,0.02,11),pad=0.01,aspect=40,shrink=0.6)
cbar.set_label('Track density', fontsize=16)
cbar.ax.tick_params(labelsize=14)
fig.suptitle('Track density relative to genesis location',fontweight='bold',fontsize=20)
plt.savefig('FigureS4.png',dpi=400)

plt.show()