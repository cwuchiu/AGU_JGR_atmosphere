#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:32:44 2026

@author: cwuchiu
"""

from numpy import where, linspace,nan,meshgrid, array, squeeze, where, pi, stack,zeros, loadtxt, arcsin, mean, concatenate, nanmin, nanmean, nanmax, array
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
import matplotlib.lines as lines
from scipy.stats import linregress
from scipy.sparse import diags, kron, identity, csr_matrix
from scipy.sparse.linalg import spsolve
from netCDF4 import Dataset
import matplotlib.patches as mpatches
import xarray as xr
from tqdm import trange
from numpy.fft import rfftn, irfftn, fftfreq, rfftfreq
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colors as colors
import scipy.stats as st
from scipy.stats import gaussian_kde
from matplotlib.colors import TwoSlopeNorm, LogNorm
import pywt
from scipy import signal
from scipy.stats import chi2


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
    

unique_times_set = set()

for matrix in save_time:
    # 假設 matrix 是二維的 (例如由列和行組成)
    for row in matrix:
        for time_str in row:
            unique_times_set.add(time_str) 
            # 如果你的 matrix 已經是一維的，就把上面那層 for row 拿掉，直接 for time_str in matrix

# 將集合轉回 list 並排序
result_list = sorted(list(unique_times_set))

print(f"處理後的長度: {len(result_list)}")
formatted_list = [int(t[0:4] + t[5:7] + t[8:10] + t[11:13]) for t in result_list]
formatted_matrix = np.array(formatted_list)

strong = nc.Dataset('TC_time_1979_2024_noET_34kt.nc','w',format='NETCDF4')

# Dimensions
strong.createDimension('time', len(result_list))
strong.createVariable('time','i8', ('time'))
strong.variables['time'][:] = formatted_matrix

strong.close()


all_times_list = []

for matrix in save_time:
    for row in matrix:
        for time_str in row:
            all_times_list.append(time_str)

print(f"處理後的長度: {len(all_times_list)}")
formatted_list = [int(t[0:4] + t[5:7] + t[8:10] + t[11:13]) for t in all_times_list]
formatted_matrix = np.array(formatted_list)


save_s = []   # 結構會跟 save_lon 完全一樣

for y in range(len(save_lon)):   # 每一年
    year_s = []
    for tc in range(len(save_lon[y])):   # 每顆颱風
        n = len(save_lon[y][tc])
        if n <= 1:
            s_tc = [0.0]   # 避免除以 0
        else:
            s_tc = np.linspace(0, 1, n)
        year_s.append(list(s_tc))
    save_s.append(year_s)

# 1. 攤平（已完成）
all_times_list = [
    t for m in save_time
      for row in m
      for t in row
]

all_lon_list = [
    t for m in save_lon
      for row in m
      for t in row
]

all_lat_list = [
    t for m in save_lat
      for row in m
      for t in row
]

all_s_list = [
    s for y in save_s
      for tc in y
      for s in tc
]

# 2. 轉成 YYYYMMDDHH（保留重複）
lo_hourly = np.array([
    t
    for t in all_lon_list
])

times_hourly = np.array([
    int(t[0:4] + t[5:7] + t[8:10] + t[11:13])
    for t in all_times_list
])

la_hourly = np.array([
    t
    for t in all_lat_list
])

s_hourly = np.array(all_s_list)

# 1. 取得「時間排序索引」
sort_idx = np.argsort(times_hourly)

# 2. 用同一組 index 重排所有變數
times_hourly_sorted = times_hourly[sort_idx]
lo_hourly_sorted    = lo_hourly[sort_idx]
la_hourly_sorted    = la_hourly[sort_idx]
s_hourly_sorted = s_hourly[sort_idx]

print(np.all(np.diff(times_hourly_sorted) >= 0))  # 應該是 True
print(len(times_hourly_sorted),
      len(lo_hourly_sorted),
      len(la_hourly_sorted),
      len(s_hourly_sorted))  # 三者必須一樣

strong = nc.Dataset('TC_time_1979_2024_not_set_noET_34kt.nc','w',format='NETCDF4')

# Dimensions
strong.createDimension('time', len(all_times_list))
strong.createVariable('time','i8', ('time'))
strong.createVariable('lon','f4', ('time'))
strong.createVariable('lat','f4', ('time'))
strong.createVariable('lifetime','f4', ('time'))

strong.variables['time'][:] = times_hourly_sorted
strong.variables['lon'][:] = lo_hourly_sorted
strong.variables['lat'][:] = la_hourly_sorted
strong.variables['lifetime'][:] = s_hourly_sorted


strong.close()