#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:40:31 2026

@author: cwuchiu
"""
#%%
from numpy import where, linspace,nan,meshgrid, array, squeeze, where, pi, stack,zeros, loadtxt, arcsin, concatenate, nanmin, nanmean, nanmax, array
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
from windspharm.xarray import VectorWind
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
from datetime import datetime
import os
#%%
data_dir = '/Dellwork4/cwuchiu/UV_TC_noET_34kt/'
out_file = '/Dellwork4/cwuchiu/TC_time_VWS_noET_34kt.nc'
start_year = 1979        # <<< 若沒有舊檔，從哪一年開始建

visualize_count = 3  # 設定要畫幾張圖來檢查 (前 N 張)

# =========================
# 初始化輸出檔案 (檢查續算)
# =========================
if os.path.exists(out_file):
    ds_out = nc.Dataset(out_file, 'a')
    tvar = ds_out.variables['time']
    if tvar.size > 0:
        try:
            exist_times = nc.num2date(tvar[:], tvar.units)
            if isinstance(exist_times, np.ndarray):
                last_time = max(exist_times)
            else:
                last_time = exist_times
        except:
            last_time = datetime(start_year, 1, 1, 0)
        print('>>> Existing file found, last time =', last_time)
    else:
        last_time = datetime(start_year, 1, 1, 0)
        print('>>> Existing file found but empty, start from', last_time)
else:
    ds_out = None
    last_time = datetime(start_year, 1, 1, 0)
    print('>>> No existing file, start from', last_time)

# =========================
# 掃描新檔案
# =========================
files = []
file_times = []

for f in sorted(os.listdir(data_dir)):
    if not f.endswith('.nc'):
        continue
    try:
        t = datetime.strptime(f[:10], '%Y%m%d%H')
    except:
        continue

    if t <= last_time:
        continue

    files.append(f)
    file_times.append(t)

print(f'>>> {len(files)} new files to process')

# =========================
# 主迴圈
# =========================
plot_counter = 0

for fname, ftime in zip(files, file_times):

    print(f'Processing: {fname} ...', end='\r')
    path = os.path.join(data_dir, fname)
    
    try:
        ds = nc.Dataset(path)

        # 讀取維度
        lat = ds.variables['latitude'][:]
        lon = ds.variables['longitude'][:]
        pres = ds.variables['pressure_level'][:]
        
        # 找出 850 hPa 和 200 hPa 的 index
        idx_850 = np.where(pres == 850)[0]
        idx_200 = np.where(pres == 200)[0]
        
        if len(idx_850) == 0 or len(idx_200) == 0:
            print(f'\nSkip {fname}: Missing 850 or 200 hPa level')
            ds.close()
            continue
            
        idx_850 = idx_850[0]
        idx_200 = idx_200[0]

        # 讀取 U 和 V 變數
        u_var = ds.variables['u']
        v_var = ds.variables['v']
        
        # 提取資料 (處理 time 維度)
        if u_var.ndim == 4: # (time, level, lat, lon)
            u_850 = u_var[0, idx_850, :, :]
            u_200 = u_var[0, idx_200, :, :]
            v_850 = v_var[0, idx_850, :, :]
            v_200 = v_var[0, idx_200, :, :]
        elif u_var.ndim == 3: # (level, lat, lon)
            u_850 = u_var[idx_850, :, :]
            u_200 = u_var[idx_200, :, :]
            v_850 = v_var[idx_850, :, :]
            v_200 = v_var[idx_200, :, :]
        else:
            ds.close()
            continue

        ds.close() # 讀取完畢先關檔

        # ==========================================
        # 計算垂直風切 (VWS)
        # 定義: U_shear = U200 - U850, V_shear = V200 - V850
        # ==========================================
        
        # 處理 Masked Array (將 mask 填為 NaN 以避免計算錯誤)
        if np.ma.is_masked(u_850): u_850 = u_850.filled(np.nan)
        if np.ma.is_masked(u_200): u_200 = u_200.filled(np.nan)
        if np.ma.is_masked(v_850): v_850 = v_850.filled(np.nan)
        if np.ma.is_masked(v_200): v_200 = v_200.filled(np.nan)

        u_shear = u_200 - u_850
        v_shear = v_200 - v_850
        
        # 計算風切大小 (Magnitude)
        vws_mag = np.sqrt(u_shear**2 + v_shear**2)

        # ==========================================
        # 視覺化驗證 (前 N 張)
        # ==========================================
        if plot_counter < visualize_count:
            try:
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
                
                # 繪製 VWS Magnitude
                # 風切通常 0~30 m/s，大於 20 通常不利颱風發展
                mesh = ax.pcolormesh(lon, lat, vws_mag, 
                                     cmap='viridis', vmin=0, vmax=30,
                                     transform=ccrs.PlateCarree())
                
                ax.coastlines(resolution='50m', color='white', linewidth=0.8)
                ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='white')
                ax.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
                
                cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
                cbar.set_label('Vertical Wind Shear (m/s)')
                
                plt.title(f'VWS (200-850 hPa): {ftime}')
                
                png_name = f'Check_VWS_{ftime.strftime("%Y%m%d%H")}.png'
                plt.savefig(png_name, dpi=120, bbox_inches='tight')
                print(f'\n[Plot] Saved: {png_name}')
                plt.close()
                
                plot_counter += 1
            except Exception as e:
                print(f'\n[Plot Error] {e}')

        # ==========================================
        # 寫入 NetCDF
        # ==========================================
        if ds_out is None:
            ds_out = nc.Dataset(out_file, 'w')
            
            # 定義維度: (time, lat, lon)
            ds_out.createDimension('time', None)
            ds_out.createDimension('lat', len(lat))
            ds_out.createDimension('lon', len(lon))

            # 定義座標變數
            tvar = ds_out.createVariable('time', 'f8', ('time',))
            tvar.units = 'hours since 1970-01-01 00:00:00'
            tvar.calendar = 'standard'
            
            ds_out.createVariable('lat', 'f4', ('lat',))[:] = lat
            ds_out.createVariable('lon', 'f4', ('lon',))[:] = lon
            
            # 定義 VWS 變數 (儲存分量與大小，以便後續分析方向)
            # zlib=True 開啟壓縮，因為 3D 陣列很大
            v_u_shr = ds_out.createVariable('u_shear', 'f4', ('time','lat','lon'), zlib=True)
            v_u_shr.long_name = "Zonal Vertical Wind Shear (U200-U850)"
            v_u_shr.units = "m/s"
            
            v_v_shr = ds_out.createVariable('v_shear', 'f4', ('time','lat','lon'), zlib=True)
            v_v_shr.long_name = "Meridional Vertical Wind Shear (V200-V850)"
            v_v_shr.units = "m/s"
            
            v_mag = ds_out.createVariable('vws', 'f4', ('time','lat','lon'), zlib=True)
            v_mag.long_name = "Vertical Wind Shear Magnitude"
            v_mag.units = "m/s"
            
            print('\n>>> Output file created')

        nt = ds_out.dimensions['time'].size
        ds_out.variables['time'][nt] = nc.date2num(ftime, ds_out.variables['time'].units)
        
        # 寫入數據
        ds_out.variables['u_shear'][nt, :, :] = u_shear
        ds_out.variables['v_shear'][nt, :, :] = v_shear
        ds_out.variables['vws'][nt, :, :] = vws_mag
        
        # 定期存檔
        if nt % 50 == 0:
            ds_out.sync()

    except Exception as e:
        print(f'\nError processing {fname}: {e}')
        continue

# =========================
# 結束
# =========================
if ds_out is not None:
    ds_out.close()

print('\n>>> Done.')

#%%
filename = '/Dellwork4/cwuchiu/TC_time_VWS_noET_34kt.nc'
start_year = 1979
end_year = 2024
years = np.arange(start_year, end_year + 1)
n_years = len(years)

print(f"Calculating Annual Mean Trend for VWS ({start_year}-{end_year})...")

# ==========================================
# 1. 讀取時間並建立索引
# ==========================================
ds = nc.Dataset(filename)
time_var = ds.variables['time']
lat = ds.variables['lat'][:]
lon = ds.variables['lon'][:]
n_lat = len(lat)
n_lon = len(lon)

# 轉換時間格式 (數值 -> datetime 物件)
print("Parsing dates...")
# num2date 會回傳 datetime 物件列表
all_dates = nc.num2date(time_var[:], time_var.units)

# 提取所有資料的年份 (建立年份索引表)
all_years_index = np.array([d.year for d in all_dates])

# ==========================================
# 2. 計算每一年的年平均 (Annual Mean)
# ==========================================
# 建立一個 (46, lat, lon) 的陣列來存年平均
# 這個大小約為 46 * 201 * 321 * 4 bytes ~= 12 MB，非常小
annual_vws = np.full((n_years, n_lat, n_lon), np.nan)

vws_var = ds.variables['vws']

print("Computing annual means...")
for i, yr in enumerate(years):
    # 找出屬於該年份的 index 位置
    # np.where 回傳的是 tuple, 取 [0]
    indices = np.where(all_years_index == yr)[0]
    
    if len(indices) == 0:
        print(f"Warning: No data for year {yr}")
        continue
    
    # 優化讀取：找出該年的起始與結束 index，使用 slice 讀取 (比散亂讀取快)
    idx_start = indices[0]
    idx_end = indices[-1] + 1
    
    # [I/O] 讀取該年的所有資料
    data_year = vws_var[idx_start:idx_end, :, :]
    
    # 處理 Masked Array (如果有缺值填 NaN)
    if np.ma.is_masked(data_year):
        data_year = data_year.filled(np.nan)
        
    # [Math] 計算年平均 (忽略 NaN)
    # axis=0 代表對時間軸平均
    annual_vws[i, :, :] = np.nanmean(data_year, axis=0)
    
    print(f"Year {yr} processed. (Shape: {data_year.shape})", end='\r')

ds.close()
print(f"\nAnnual means calculated. Shape: {annual_vws.shape}")

# ==========================================
# 3. 計算長期線性趨勢 (Vectorized Linear Regression)
# ==========================================
print("Computing linear trend...")

# 定義 X 軸 (時間: 0, 1, 2, ... 45)
x = np.arange(n_years)
# 定義 Y 軸 (年平均資料)
y = annual_vws

# 為了向量化運算，利用廣播機制 (Broadcasting)
# 公式: Slope = sum((x - x_bar) * (y - y_bar)) / sum((x - x_bar)^2)

x_mean = np.mean(x)
y_mean = np.nanmean(y, axis=0) # (lat, lon)

# 分子: Covariance
# x[:, None, None] 將形狀轉為 (46, 1, 1) 以便跟 (46, lat, lon) 相乘
numerator = np.nansum((x[:, None, None] - x_mean) * (y - y_mean), axis=0)

# 分母: Variance of x
denominator = np.sum((x - x_mean)**2)

# 斜率 (Trend)
trend_map = numerator / denominator
#%%
# 單位轉換: 目前算出來是 "m/s per year"
# 如果想要 "m/s per decade"，可以乘 10
trend_map_decade = trend_map * 46

print("Calculation done.")

# ==========================================
# 4. 繪圖檢查
# ==========================================
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# 設定顏色範圍 (視數值大小調整)
# 假設趨勢大概在 +/- 0.5 m/s per decade 左右
limit = np.nanpercentile(np.abs(trend_map_decade), 98) # 自動抓 98% 分位數當範圍
if np.isnan(limit): limit = 1.0

pcm = ax.pcolormesh(lon, lat, trend_map_decade, 
                    cmap='RdBu_r', 
                    vmin=-limit, vmax=limit,
                    transform=ccrs.PlateCarree())

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
x_extent = linspace(-180,180,37)
y_extent = linspace(-90,90,19)
ax.set_xticks(x_extent, crs=ccrs.PlateCarree())
ax.set_yticks(y_extent, crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

ax.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())

cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.8)
cbar.set_label('VWS Trend (ms$^{-1}$ 46yr$^{-1}$)')

plt.title(f'Vertical Wind Shear Trend ({start_year}-{end_year})',fontsize=16,fontweight='bold')
#plt.savefig('VWS_trend_1979-2024.png')
plt.tight_layout()
plt.savefig('Fig5-6.png',dpi=400)
plt.show()

# ==========================================
# (選用) 儲存結果到 NetCDF 方便後續使用
# ==========================================
# output_nc = '/Dellwork4/cwuchiu/VWS_Trend_1979_2024.nc'
# ds_save = nc.Dataset(output_nc, 'w')
# ds_save.createDimension('lat', n_lat)
# ds_save.createDimension('lon', n_lon)
# ds_save.createVariable('lat', 'f4', ('lat',))[:] = lat
# ds_save.createVariable('lon', 'f4', ('lon',))[:] = lon
# v_trend = ds_save.createVariable('vws_trend', 'f4', ('lat','lon'))
# v_trend.units = "m/s per year"
# v_trend[:] = trend_map
# ds_save.close()
# print("Trend saved.")


#%%
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cftime
from scipy import stats

# ==============================
# 1. 參數與讀檔設定
# ==============================
filename = '/Dellwork4/cwuchiu/TC_time_VWS_noET_34kt.nc'
vws_var_name = 'vws'  # 請確認 nc 檔內的垂直風切變數名稱 (例如 'vws', 'shear', 'wspd_diff' 等)

start_year = 1979
end_year = 2024
target_years = np.arange(start_year, end_year + 1)

# 設定空間範圍
target_lon = (100, 180)
target_lat = (0, 30)

# 開啟檔案
ds = nc.Dataset(filename)
lat = ds.variables['lat'][:]
lon = ds.variables['lon'][:]
time_var = ds.variables['time']
vws_var = ds.variables[vws_var_name]

lon,lat = np.meshgrid(lon,lat)
vws_lon = np.zeros(41682)
vws_lat = np.zeros(41682)


for i in trange(41682):
    mask = (lat<=30) & (vws_var[i]<=10)
    vws_lon[i] = np.nanmean(lon[mask])
    vws_lat[i] = np.nanmean(lat[mask])
#%%
time = np.linspace(1979,2024,41682)

slope1, intercept1, r_value1, p_value1, std_err1 = linregress(time, vws_lon)
plt.plot(time,vws_lon,'b',label='')
plt.plot(time, time*slope1+intercept1, 'r--')
plt.text(2000,130,f"p-value = {p_value1:.2e}",fontsize=16)
plt.title(f"favor genesis region mean longitude (VWS<=10m/s)")
plt.xlim(1979,2024)
plt.xticks(np.linspace(1979,2024,10))
plt.xlabel('year')
plt.ylabel('longitude')
plt.savefig('vws_lon_hourly.png')
plt.show()

slope1, intercept1, r_value1, p_value1, std_err1 = linregress(time, vws_lat)
plt.plot(time,vws_lat,'b',label='')
plt.plot(time, time*slope1+intercept1, 'r--')
plt.text(2000,15,f"p-value = {p_value1:.2e}",fontsize=16)
plt.title(f"favor genesis region mean latitude (VWS<=10m/s)")
plt.xlim(1979,2024)
plt.xticks(np.linspace(1979,2024,10))
plt.xlabel('year')
plt.ylabel('latitude')
plt.savefig('vws_lat_hourly.png')
plt.show()

ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_1979_2024_noET_34kt.nc')
tt = np.array(ncfile.variables['time'])
time_tc = [str(x) for x in tt]
ncfile.close()

vws_lon_yr = np.zeros(46)
vws_lat_yr = np.zeros(46)

yr_case_number = np.zeros(46)

check_time = 1979
for i in range(len(time_tc)):
    if int(time_tc[i][0:4])==check_time:
        vws_lon_yr[int(check_time-1979)] += vws_lon[i]
        vws_lat_yr[int(check_time-1979)] += vws_lat[i]
        yr_case_number[int(check_time-1979)] +=1
    else:
        check_time+=1
        vws_lon_yr[int(check_time-1979)] += vws_lon[i]
        vws_lat_yr[int(check_time-1979)] += vws_lat[i]
        yr_case_number[int(check_time-1979)] +=1

vws_lon_yr = vws_lon_yr/yr_case_number[:]
vws_lat_yr = vws_lat_yr/yr_case_number[:]
#%%
time = np.linspace(1979,2024,46)
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(time, vws_lon_yr)
plt.plot(time,vws_lon_yr, marker='o', linestyle='-')
plt.plot(time, time*slope1+intercept1, 'r--')
plt.text(2000,146,f"p-value = {p_value1:.2e}",fontsize=16)
plt.title(f"favor genesis region mean longitude (VWS<=10m/s)")
plt.xlim(1979,2024)
plt.xticks(np.linspace(1979,2024,10))
plt.xlabel('year')
plt.ylabel('longitude')
plt.savefig('vws_lon_yr.png')
plt.show()

slope1, intercept1, r_value1, p_value1, std_err1 = linregress(time, vws_lat_yr)
plt.plot(time,vws_lat_yr, marker='o', linestyle='-')
plt.plot(time, time*slope1+intercept1, 'r--')
plt.text(2000,13.5,f"p-value = {p_value1:.2e}",fontsize=16)
plt.title(f"favor genesis region mean latitude (VWS<=10m/s)")
plt.xlim(1979,2024)
plt.xticks(np.linspace(1979,2024,10))
plt.xlabel('year')
plt.ylabel('latitude')
plt.savefig('vws_lat_yr.png')
plt.show()

#%%
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

time = np.linspace(1979, 2024, 46)

fig, axs = plt.subplots(2, 1, figsize=(12, 10))  # 2 行 1 列 subplot

# ----------------------
# 上方：經度
# ----------------------
slope_lon, intercept_lon, r_value_lon, p_value_lon, std_err_lon = linregress(time, vws_lon_yr)
axs[0].plot(time, vws_lon_yr, marker='o', linestyle='-', label='Mean longitude')
axs[0].plot(time, time*slope_lon + intercept_lon, 'r--', label=f'Linear fit')
axs[0].text(2000, 146, f"p-value = {p_value_lon:.2e}", fontsize=14)
axs[0].set_title("Favor genesis region mean longitude (VWS <= 10 m/s)", fontweight='bold',fontsize=16)
axs[0].set_xlim(1979, 2024)
axs[0].set_xticks(np.linspace(1979, 2024, 10))
axs[0].set_ylabel('Longitude')

# ----------------------
# 下方：緯度
# ----------------------
slope_lat, intercept_lat, r_value_lat, p_value_lat, std_err_lat = linregress(time, vws_lat_yr)
axs[1].plot(time, vws_lat_yr, marker='o', linestyle='-', label='Mean latitude')
axs[1].plot(time, time*slope_lat + intercept_lat, 'r--', label=f'Linear fit')
axs[1].text(2000, 13.5, f"p-value = {p_value_lat:.2e}", fontsize=14)
axs[1].set_title("Favor genesis region mean latitude (VWS <= 10 m/s)", fontweight='bold',fontsize=16)
axs[1].set_xlim(1979, 2024)
axs[1].set_xticks(np.linspace(1979, 2024, 10))
axs[1].set_xlabel('Year')
axs[1].set_ylabel('Latitude')

plt.tight_layout()
plt.savefig('vws_lon_lat_yr.png', dpi=300)
plt.show()



