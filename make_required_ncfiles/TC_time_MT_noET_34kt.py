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
from scipy import stats, signal
#%%
data_dir = '/Dellwork4/cwuchiu/UV_TC_noET_34kt/'
out_file = '/Dellwork4/cwuchiu/TC_time_MT_noET_34kt.nc'
start_year = 1979        # <<< 若沒有舊檔，從哪一年開始建

visualize_count = 3  # 設定要畫幾張圖來檢查 (前 N 張)

def get_monsoon_trough_center_out(u_850, lats, lons, 
                                  anchor_lon_range=(110, 150), # <--- 放寬範圍
                                  search_lat_max=40,           # <--- 稍微放寬上限
                                  max_jump=6.0):
    """
    修正版中心擴散追蹤法:
    1. 處理 Masked Array 問題。
    2. 放寬 Anchor 搜尋範圍 (110E-150E)，避免季風槽西退時抓不到。
    """
    n_lon = len(lons)
    mt_lats = np.full(n_lon, np.nan)

    # ==========================================
    # Fix 1: 確保數據是純 Numpy Array (處理 MaskedArray)
    # ==========================================
    if np.ma.is_masked(u_850):
        u_850 = u_850.filled(np.nan) # 將 mask 填為 NaN
    
    # 轉為 np.array 以防萬一
    u_850 = np.array(u_850)

    # 1. 準備資料: 確保緯度由小到大 (南->北)
    if lats[0] > lats[-1]: 
        process_lats = lats[::-1]
        process_u = u_850[::-1, :]
    else:
        process_lats = lats
        process_u = u_850

    # 限制搜尋的最大緯度
    lat_indices = np.where(process_lats <= search_lat_max)[0]
    if len(lat_indices) < 2: return mt_lats
    
    subset_lats = process_lats[lat_indices]
    subset_u = process_u[lat_indices, :]

    # ==========================================
    # Helper: 找出某個經度上所有的 U=0 候選點
    # ==========================================
    def get_candidates(lon_idx):
        col_u = subset_u[:, lon_idx]
        
        # 忽略全 NaN 的列
        if np.isnan(col_u).all():
            return np.array([])

        # 尋找 西風(>0) 轉 東風(<0)
        # 這裡要小心 NaN 的比較，使用 np.nan_to_num 或忽略警告
        with np.errstate(invalid='ignore'): 
            # 找出非 NaN 的區域進行比較
            valid_mask = ~np.isnan(col_u)
            if np.sum(valid_mask) < 2: return np.array([])
            
            # 只在有效數據區尋找
            # 這裡簡化邏輯：直接比對，NaN 比較結果會是 False，不影響
            crossings = np.where((col_u[:-1] > 0) & (col_u[1:] < 0))[0]
        
        cands = []
        for idx in crossings:
            u1 = col_u[idx]
            u2 = col_u[idx+1]
            y1 = subset_lats[idx]
            y2 = subset_lats[idx+1]
            
            # 防呆: 避免除以 0
            if abs(u2 - u1) < 1e-5:
                zero_lat = y1
            else:
                zero_lat = y1 + (0 - u1) / (u2 - u1) * (y2 - y1)
            
            cands.append(zero_lat)
        return np.array(cands)

    # ==========================================
    # Step 1: 尋找 Anchor (起點) - 範圍放寬
    # ==========================================
    anchor_indices = np.where((lons >= anchor_lon_range[0]) & (lons <= anchor_lon_range[1]))[0]
    
    if len(anchor_indices) == 0: return mt_lats

    best_anchor_lon_idx = -1
    best_anchor_lat = np.nan
    min_diff_climo = 999
    
    # 掃描整個大範圍，尋找最符合 "15N" 特徵的切變點
    for idx in anchor_indices:
        cands = get_candidates(idx)
        if len(cands) == 0: continue
        
        # 找最接近 15N 的候選人
        diffs = np.abs(cands - 15.0)
        local_min_idx = np.argmin(diffs)
        
        # 額外檢查: 該點必須在合理的熱帶範圍 (例如 5N~30N) 才當作 Anchor
        # 避免抓到太北的鋒面當作種子
        candidate_lat = cands[local_min_idx]
        if candidate_lat < 5 or candidate_lat > 35:
            continue

        if diffs[local_min_idx] < min_diff_climo:
            min_diff_climo = diffs[local_min_idx]
            best_anchor_lat = candidate_lat
            best_anchor_lon_idx = idx

    # 如果還是找不到，回傳空值 (但通常範圍放寬後應該找得到)
    if np.isnan(best_anchor_lat):
        # Debug 訊息: 只有在完全找不到時才印出，避免洗版
        # print(f"Warning: No anchor found in {anchor_lon_range}")
        return mt_lats
    
    # 填入起點
    mt_lats[best_anchor_lon_idx] = best_anchor_lat

    # ==========================================
    # Step 2: 向東追蹤
    # ==========================================
    prev_lat = best_anchor_lat
    for i in range(best_anchor_lon_idx + 1, n_lon):
        cands = get_candidates(i)
        if len(cands) == 0: break 
            
        dists = np.abs(cands - prev_lat)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        
        if min_dist > max_jump: break 
        
        mt_lats[i] = cands[min_dist_idx]
        prev_lat = mt_lats[i]

    # ==========================================
    # Step 3: 向西追蹤
    # ==========================================
    prev_lat = best_anchor_lat
    for i in range(best_anchor_lon_idx - 1, -1, -1):
        if lons[i] < 100: break # 不要太西邊

        cands = get_candidates(i)
        if len(cands) == 0: break
            
        dists = np.abs(cands - prev_lat)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        
        if min_dist > max_jump: break
            
        mt_lats[i] = cands[min_dist_idx]
        prev_lat = mt_lats[i]

    return mt_lats

# =========================
# 初始化輸出檔案
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
                last_time = exist_times # 單一時間的情況
        except:
             # fallback if num2date fails or array is weird
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
plot_counter = 0  # 繪圖計數器

for fname, ftime in zip(files, file_times):

    print(f'Processing: {fname} ...', end='\r')
    path = os.path.join(data_dir, fname)
    
    try:
        ds = nc.Dataset(path)

        # 讀取變數
        lat = ds.variables['latitude'][:]
        lon = ds.variables['longitude'][:]
        pres = ds.variables['pressure_level'][:]
        
        # 找 850 hPa
        idx_850 = np.where(pres == 850)[0]
        if len(idx_850) == 0:
            ds.close()
            continue
        idx_850 = idx_850[0]

        u_var = ds.variables['u']
        
        # 處理維度 (是否有 time 維度)
        if u_var.ndim == 4:
            u_850_data = u_var[0, idx_850, :, :]
        elif u_var.ndim == 3:
            u_850_data = u_var[idx_850, :, :]
        else:
            ds.close()
            continue

        ds.close() # 讀完就可以先關輸入檔

        # --- 核心計算 ---
        mt_lats = get_monsoon_trough_center_out(u_850_data, lat, lon)

        # --- 視覺化驗證 (前 N 張) ---
        if plot_counter < visualize_count:
            try:
                fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
                
                # 設定顏色: 紅=西風(>0), 藍=東風(<0), 白=0
                norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
                
                # 畫風場底圖
                pcm = ax.pcolormesh(lon, lat, u_850_data, 
                                    cmap='RdBu_r', norm=norm, 
                                    transform=ccrs.PlateCarree())
                
                # 畫計算出的軸線 (螢光綠)
                valid_mask = ~np.isnan(mt_lats)
                ax.plot(lon[valid_mask], mt_lats[valid_mask], 
                        color='#00FF00', linewidth=2.5, 
                        label='Monsoon Trough (U=0)',
                        transform=ccrs.PlateCarree())

                # 地圖設定
                ax.coastlines(resolution='50m', color='k')
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
                
                # Colorbar
                cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
                cbar.set_label('U850 (m/s) [Red: Westerly, Blue: Easterly]')
                
                plt.title(f'Check Monsoon Trough: {ftime}')
                plt.legend(loc='upper right')
                
                # 存圖
                png_name = f'Check_MT_{ftime.strftime("%Y%m%d%H")}.png'
                plt.savefig(png_name, dpi=120, bbox_inches='tight')
                print(f'\n[Plot] Saved verification plot: {png_name}')
                plt.close() # 關閉圖表釋放記憶體
                
                plot_counter += 1

            except Exception as e:
                print(f'\n[Plot Error] Could not plot {fname}: {e}')

        # --- 寫入 NetCDF ---
        if ds_out is None:
            ds_out = nc.Dataset(out_file, 'w')
            ds_out.createDimension('time', None)
            ds_out.createDimension('lon', len(lon))

            tvar = ds_out.createVariable('time', 'f8', ('time',))
            tvar.units = 'hours since 1970-01-01 00:00:00'
            tvar.calendar = 'standard'
            
            ds_out.createVariable('lon', 'f4', ('lon',))[:] = lon
            
            vmt = ds_out.createVariable('mt_lat', 'f4', ('time','lon'), zlib=True)
            vmt.long_name = "Monsoon Trough Latitude (U850=0)"
            vmt.missing_value = np.nan
            print('\n>>> Output file created')

        nt = ds_out.dimensions['time'].size
        ds_out.variables['time'][nt] = nc.date2num(ftime, ds_out.variables['time'].units)
        ds_out.variables['mt_lat'][nt, :] = mt_lats
        
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
