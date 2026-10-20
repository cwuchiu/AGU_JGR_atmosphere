#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:26:26 2026

@author: cwuchiu
"""
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from scipy import stats

# ==========================================
# 1. 讀取資料與計算年平均
# ==========================================
ncfile = nc.Dataset('/Dellwork4/cwuchiu/SHAI_km2_HGT_Lon_Lat_1979_2024_monthly_hourly_adjust_mean_2004.nc')
weighted_lon = np.array(ncfile.variables['Lon_monthly'])
west_lon = np.array(ncfile.variables['west_monthly'])
north_lat = np.array(ncfile.variables['north_monthly'])
ncfile.close()

ncfile = nc.Dataset('/Dellwork4/cwuchiu/WNPSH_ridge_lat_1979_2024.nc')
lat_yr = np.array(ncfile.variables['ridge_lat_yearly'])
ncfile.close()

lon_yr = np.zeros(46)
west_yr = np.zeros(46)
north_yr = np.zeros(46)
for i in range(46):
    lon_yr[i] = np.nanmean(weighted_lon[6*i:6*(i+1)], axis=0)
    west_yr[i] = np.nanmean(west_lon[6*i:6*(i+1)], axis=0)
    north_yr[i] = np.nanmean(north_lat[6*i:6*(i+1)], axis=0)

time = np.linspace(1979, 2024, 46)

# ==========================================
# 2. 畫圖設定 (2x2 子圖)
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(18, 12), sharex=True)

LABEL_SIZE = 18
TICK_SIZE = 16
TITLE_SIZE = 20
LEGEND_SIZE = 15

# ------------------------------------------
# [0, 0] 左上 (a)：Centroid Longitude
# ------------------------------------------
ax = axs[0, 0]
ax.plot(time, lon_yr, 'o-', color='navy', linewidth=2, markersize=6)

mask_lon = ~np.isnan(lon_yr)
if np.sum(mask_lon) > 1:
    slope, intercept, r_val, p_val, std_err = stats.linregress(time[mask_lon], lon_yr[mask_lon])
    trend_line = slope * time[mask_lon] + intercept
    trend_text = f'Trend: {slope*10:.2f}°E / decade (p={p_val:.2e})'
    ax.plot(time[mask_lon], trend_line, '--', color='crimson', label=trend_text, linewidth=2)

ax.set_title('(a) Centroid Longitude (1979-2024)', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_ylabel('Longitude (°E)', fontsize=LABEL_SIZE)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.set_xlim(1979, 2024)

# ------------------------------------------
# [0, 1] 右上 (b)：Ridge Latitude
# ------------------------------------------
ax = axs[0, 1]
ax.plot(time, lat_yr, 'o-', color='darkgreen', linewidth=2, markersize=6)

mask_lat = ~np.isnan(lat_yr)
if np.sum(mask_lat) > 1:
    slope, intercept, r_val, p_val, std_err = stats.linregress(time[mask_lat], lat_yr[mask_lat])
    trend_line = slope * time[mask_lat] + intercept
    trend_text = f'Trend: {slope*10:.2f}°N / decade (p={p_val:.2e})'
    ax.plot(time[mask_lat], trend_line, '--', color='crimson', label=trend_text, linewidth=2)

ax.set_title('(b) Ridge Latitude (1979-2024)', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_ylabel('Latitude (°N)', fontsize=LABEL_SIZE)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.set_xlim(1979, 2024)

# ------------------------------------------
# [1, 0] 左下 (c)：Western Edge Longitude
# ------------------------------------------
ax = axs[1, 0]
ax.plot(time, west_yr, 'o-', color='navy', linewidth=2, markersize=6)

mask_west = ~np.isnan(west_yr)
if np.sum(mask_west) > 1:
    slope, intercept, r_val, p_val, std_err = stats.linregress(time[mask_west], west_yr[mask_west])
    trend_line = slope * time[mask_west] + intercept
    trend_text = f'Trend: {slope*10:.2f}°E / decade (p={p_val:.2e})'
    ax.plot(time[mask_west], trend_line, '--', color='crimson', label=trend_text, linewidth=2)

ax.set_title('(c) Western Edge Longitude (1979-2024)', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_xlabel('Year', fontsize=LABEL_SIZE)
ax.set_ylabel('Longitude (°E)', fontsize=LABEL_SIZE)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.set_xticks(np.linspace(1979, 2024, 10))
ax.set_xlim(1979, 2024)

# ------------------------------------------
# [1, 1] 右下 (d)：Northern Edge Latitude
# ------------------------------------------
ax = axs[1, 1]
ax.plot(time, north_yr, 'o-', color='darkgreen', linewidth=2, markersize=6)

mask_north = ~np.isnan(north_yr)
if np.sum(mask_north) > 1:
    slope, intercept, r_val, p_val, std_err = stats.linregress(time[mask_north], north_yr[mask_north])
    trend_line = slope * time[mask_north] + intercept
    trend_text = f'Trend: {slope*10:.2f}°N / decade (p={p_val:.2e})'
    ax.plot(time[mask_north], trend_line, '--', color='crimson', label=trend_text, linewidth=2)

ax.set_title('(d) Northern Edge Latitude (1979-2024)', fontsize=TITLE_SIZE, fontweight='bold')
ax.set_xlabel('Year', fontsize=LABEL_SIZE)
ax.set_ylabel('Latitude (°N)', fontsize=LABEL_SIZE)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=LEGEND_SIZE, loc='upper right')
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.set_xticks(np.linspace(1979, 2024, 10))
ax.set_xlim(1979, 2024)

# ==========================================
# 3. 排版與儲存
# ==========================================
plt.tight_layout()
plt.savefig('Figure4.png', dpi=400, bbox_inches='tight')
plt.show()