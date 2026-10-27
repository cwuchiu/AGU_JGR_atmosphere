#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:40:31 2026

@author: cwuchiu
"""

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
