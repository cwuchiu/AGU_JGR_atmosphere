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

import sys
from multiprocessing import Pool
from tqdm import tqdm

# =============================================================================
# 1. 核心參數與防護鎖定 (避免 BLAS/OpenMP 線程風暴)
# =============================================================================
N_WORKERS = 32  # 平行核心數

# 設定環境變數：強制底層數學庫使用單執行緒
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_WORKERS)

# Linux 物理核心鎖定
try:
    target_cpus = set(range(N_WORKERS))
    os.sched_setaffinity(0, target_cpus)
    print(f"[系統確認] 已由 Linux Kernel 物理鎖定在 CPU 0 ~ {N_WORKERS-1} 號核心！")
except (AttributeError, OSError):
    pass

# 設定路徑與參數
data_dir = '/Dellwork4/cwuchiu/UV_TC_noET_34kt/'
out_file = '/Dellwork4/cwuchiu/TC_time_vorticity_noET_34kt.nc'
start_year = 1979
visualize_count = 3  # 檢查前 N 張圖

# =============================================================================
# 2. 核心計算函數: 相對渦度 (Relative Vorticity)
# =============================================================================
def compute_vorticity(u, v, lat, lon):
    """
    計算球面坐標系下的相對渦度 (s^-1)
    """
    R = 6.371e6  # 地球半徑 (m)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # 1. 計算 dv/dx (沿經度方向微分)
    dlon = lon_rad[1] - lon_rad[0]
    dv_dlon = np.gradient(v, dlon, axis=1)
    dv_dx = dv_dlon / (R * np.cos(lat_rad[:, None]))
    
    # 2. 計算 du/dy (沿緯度方向微分，含球面修正項)
    dlat = lat_rad[1] - lat_rad[0]
    u_cos_lat = u * np.cos(lat_rad[:, None])
    d_u_cos_lat_dlat = np.gradient(u_cos_lat, dlat, axis=0)
    term2 = d_u_cos_lat_dlat / (R * np.cos(lat_rad[:, None]))
    
    return dv_dx - term2

# =============================================================================
# 3. Worker 進程初始化與工作函數
# =============================================================================
def _init_worker():
    """確保子進程徹底維持單執行緒"""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass

def _process_single_file(task):
    """
    單一 Worker 讀取單檔並計算渦度
    task: (fname, ftime, data_dir)
    """
    fname, ftime, d_dir = task
    path = os.path.join(d_dir, fname)
    
    try:
        with nc.Dataset(path, 'r') as ds:
            lat = ds.variables['latitude'][:]
            lon = ds.variables['longitude'][:]
            pres = ds.variables['pressure_level'][:]
            
            idx_850 = np.where(pres == 850)[0]
            idx_500 = np.where(pres == 500)[0]
            
            if len(idx_850) == 0 or len(idx_500) == 0:
                return (ftime, None, None, None, None, f"Skip: Missing 850 or 500 hPa level in {fname}")
                
            idx_850 = idx_850[0]
            idx_500 = idx_500[0]

            u_var = ds.variables['u']
            v_var = ds.variables['v']
            
            # 維度處理
            if u_var.ndim == 4:
                u_850 = u_var[0, idx_850, :, :]
                u_500 = u_var[0, idx_500, :, :]
                v_850 = v_var[0, idx_850, :, :]
                v_500 = v_var[0, idx_500, :, :]
            elif u_var.ndim == 3:
                u_850 = u_var[idx_850, :, :]
                u_500 = u_var[idx_500, :, :]
                v_850 = v_var[idx_850, :, :]
                v_500 = v_var[idx_500, :, :]
            else:
                return (ftime, None, None, None, None, f"Skip: Unsupported dimensions in {fname}")

        # Masked Array 處理
        if np.ma.is_masked(u_850): u_850 = u_850.filled(np.nan)
        if np.ma.is_masked(u_500): u_500 = u_500.filled(np.nan)
        if np.ma.is_masked(v_850): v_850 = v_850.filled(np.nan)
        if np.ma.is_masked(v_500): v_500 = v_500.filled(np.nan)

        # 平行運算核心：計算渦度
        vort_850 = compute_vorticity(u_850, v_850, lat, lon).astype(np.float32)
        vort_500 = compute_vorticity(u_500, v_500, lat, lon).astype(np.float32)

        return (ftime, vort_850, vort_500, lat, lon, None)

    except Exception as e:
        return (ftime, None, None, None, None, f"Error in {fname}: {str(e)}")

# =============================================================================
# 4. 主流程控制
# =============================================================================
if __name__ == '__main__':
    # 延遲引入繪圖庫，避免子進程 GUI backend 衝突
    import matplotlib
    matplotlib.use('Agg')
    try:
        has_cartopy = True
    except ImportError:
        has_cartopy = False
        print("[警告] 未偵測到 Cartopy，將略過地圖視覺化驗證")

    # -------------------------------------------------------------------------
    # 檢查續算與找出已存在的最後時間戳
    # -------------------------------------------------------------------------
    last_time = datetime(start_year, 1, 1, 0)
    if os.path.exists(out_file):
        try:
            with nc.Dataset(out_file, 'r') as ds_check:
                if 'time' in ds_check.variables and ds_check.variables['time'].size > 0:
                    tvar = ds_check.variables['time']
                    exist_times = nc.num2date(tvar[:], tvar.units)
                    last_time = max(exist_times) if isinstance(exist_times, np.ndarray) else exist_times
                    print(f'>>> Existing output file found. Last processed time: {last_time}')
        except Exception as e:
            print(f'>>> Failed to read existing file: {e}. Start from {last_time}')
    else:
        print(f'>>> No existing file found. Will create new output starting from {last_time}')

    # -------------------------------------------------------------------------
    # 掃描並篩選待處理檔案
    # -------------------------------------------------------------------------
    tasks = []
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.nc'):
            continue
        try:
            t = datetime.strptime(f[:10], '%Y%m%d%H')
        except ValueError:
            continue

        if t <= last_time:
            continue

        tasks.append((f, t, data_dir))

    print(f'>>> Found {len(tasks)} new files to process.')
    if not tasks:
        print('>>> All files are up to date. Exiting.')
        sys.exit(0)

    # -------------------------------------------------------------------------
    # 平行運算與主執行緒循序寫入
    # -------------------------------------------------------------------------
    ds_out = None
    plot_counter = 0

    try:
        # 開啟 Pool
        with Pool(processes=min(N_WORKERS, len(tasks)), initializer=_init_worker) as pool:
            # 使用 imap 保持時間順序，串流處理輸出以防記憶體溢出
            worker_results = pool.imap(_process_single_file, tasks, chunksize=1)

            for ftime, vort_850, vort_500, lat, lon, err_msg in tqdm(worker_results, total=len(tasks), desc="Calculating Vorticity"):
                
                if err_msg is not None:
                    print(f"\n[Worker Warn] {err_msg}")
                    continue

                # 第一次遇到成功資料時，初始化寫入檔
                if ds_out is None:
                    mode = 'a' if os.path.exists(out_file) else 'w'
                    ds_out = nc.Dataset(out_file, mode)

                    if 'lat' not in ds_out.dimensions:
                        ds_out.createDimension('time', None)
                        ds_out.createDimension('lat', len(lat))
                        ds_out.createDimension('lon', len(lon))

                        tvar = ds_out.createVariable('time', 'f8', ('time',))
                        tvar.units = 'hours since 1970-01-01 00:00:00'
                        tvar.calendar = 'standard'

                        ds_out.createVariable('lat', 'f4', ('lat',))[:] = lat
                        ds_out.createVariable('lon', 'f4', ('lon',))[:] = lon

                        v_vort850 = ds_out.createVariable('vort_850', 'f4', ('time', 'lat', 'lon'), zlib=True)
                        v_vort850.long_name = "Relative Vorticity at 850 hPa"
                        v_vort850.units = "s^-1"

                        v_vort500 = ds_out.createVariable('vort_500', 'f4', ('time', 'lat', 'lon'), zlib=True)
                        v_vort500.long_name = "Relative Vorticity at 500 hPa"
                        v_vort500.units = "s^-1"
                        print('\n>>> Output file initialized.')

                # -------------------------------------------------------------
                # 視覺化驗證 (前 N 張圖由主進程輸出)
                # -------------------------------------------------------------
                if has_cartopy and plot_counter < visualize_count:
                    try:
                        fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
                        limit = 5e-5
                        mesh = ax.pcolormesh(lon, lat, vort_850,
                                             cmap='RdBu_r', vmin=-limit, vmax=limit,
                                             transform=ccrs.PlateCarree())
                        ax.coastlines(resolution='50m', color='black', linewidth=0.8)
                        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
                        ax.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
                        
                        cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
                        cbar.set_label('Rel. Vorticity (s$^{-1}$)')
                        plt.title(f'850 hPa Vorticity: {ftime}')
                        
                        png_name = f'Check_Vort_{ftime.strftime("%Y%m%d%H")}.png'
                        plt.savefig(png_name, dpi=120, bbox_inches='tight')
                        plt.close()
                        plot_counter += 1
                    except Exception as e:
                        print(f"\n[Plot Error] {e}")

                # -------------------------------------------------------------
                # 寫入 NetCDF 檔案 (嚴格按時間順序 Append)
                # -------------------------------------------------------------
                nt = ds_out.dimensions['time'].size
                ds_out.variables['time'][nt] = nc.date2num(ftime, ds_out.variables['time'].units)
                ds_out.variables['vort_850'][nt, :, :] = vort_850
                ds_out.variables['vort_500'][nt, :, :] = vort_500

                # 定期刷新快取至磁碟
                if (nt + 1) % 50 == 0:
                    ds_out.sync()

    finally:
        if ds_out is not None:
            ds_out.close()
            print('\n>>> Output NetCDF file safely closed.')

    print('>>> All tasks successfully completed.')
