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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:57:04 2026

@author: cwuchiu
"""

import os
import sys
from datetime import datetime
import numpy as np
import netCDF4 as nc
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
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
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

#%%
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================
# 設定
# ==========================================
filename = '/Dellwork4/cwuchiu/TC_time_vorticity_noET_34kt.nc'  # 修改讀取檔案
start_year = 1979
end_year = 2024
years = np.arange(start_year, end_year + 1)
n_years = len(years)

print(f"Calculating Annual Mean Trend for Vorticity ({start_year}-{end_year})...")

# ==========================================
# 1. 讀取時間並建立索引
# ==========================================
ds = nc.Dataset(filename)
time_var = ds.variables['time']
lat = ds.variables['lat'][:]
lon = ds.variables['lon'][:]
n_lat = len(lat)
n_lon = len(lon)

# 讀取兩個渦度變數
vort850_var = ds.variables['vort_850']
vort500_var = ds.variables['vort_500']

# 轉換時間格式
print("Parsing dates...")
all_dates = nc.num2date(time_var[:], time_var.units)
all_years_index = np.array([d.year for d in all_dates])

# ==========================================
# 2. 計算每一年的年平均 (Annual Mean)
# ==========================================
# 準備兩個陣列存年平均
annual_v850 = np.full((n_years, n_lat, n_lon), np.nan)
annual_v500 = np.full((n_years, n_lat, n_lon), np.nan)

print("Computing annual means...")
for i, yr in enumerate(years):
    # 找出屬於該年份的 index
    indices = np.where(all_years_index == yr)[0]
    
    if len(indices) == 0:
        print(f"Warning: No data for year {yr}")
        continue
    
    # Slice 讀取優化
    idx_start = indices[0]
    idx_end = indices[-1] + 1
    
    # --- 處理 850 hPa ---
    data_850 = vort850_var[idx_start:idx_end, :, :]
    if np.ma.is_masked(data_850):
        data_850 = data_850.filled(np.nan)
    annual_v850[i, :, :] = np.nanmean(data_850, axis=0)
    
    # --- 處理 500 hPa ---
    data_500 = vort500_var[idx_start:idx_end, :, :]
    if np.ma.is_masked(data_500):
        data_500 = data_500.filled(np.nan)
    annual_v500[i, :, :] = np.nanmean(data_500, axis=0)
    
    print(f"Year {yr} processed.", end='\r')

ds.close()
print(f"\nAnnual means calculated.")

# ==========================================
# 3. 計算長期線性趨勢 (Vectorized)
# ==========================================
# 為了避免重複代碼，定義一個簡單的計算函數
def compute_linear_trend(annual_data):
    """
    Input: (n_years, lat, lon)
    Output: trend_per_decade (lat, lon)
    """
    x = np.arange(n_years)
    y = annual_data
    
    x_mean = np.mean(x)
    y_mean = np.nanmean(y, axis=0)

    # Covariance
    numerator = np.nansum((x[:, None, None] - x_mean) * (y - y_mean), axis=0)
    # Variance
    denominator = np.sum((x - x_mean)**2)
    
    trend_per_year = numerator / denominator
    return trend_per_year * 46  # 轉為每10年 (per decade)

print("Computing linear trends for 850 hPa...")
trend_850 = compute_linear_trend(annual_v850)

print("Computing linear trends for 500 hPa...")
trend_500 = compute_linear_trend(annual_v500)

print("Calculation done.")

# ==========================================
# 4. 繪圖檢查 (雙圖並排)
# ==========================================
#%%
fig, axes = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# 定義繪圖函數
def plot_trend_map(ax, data, title):
    # 自動抓取適合的顏色範圍 (98% 分位數)
    limit = np.nanpercentile(np.abs(data), 98)
    if np.isnan(limit) or limit == 0: limit = 1e-6

    # RdBu_r: 紅色為正(氣旋式增強)，藍色為負(反氣旋式/渦度減弱)
    pcm = ax.pcolormesh(lon, lat, data, 
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
    
    # 加上 Colorbar
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.7, pad=0.03)
    # 這裡使用 scientific notation 因為渦度數值很小
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label('Trend (s$^{-1}$ 46yr$^{-1}$)')
    
    ax.set_title(title, fontsize=20,fontweight='bold')
    

# 繪製兩張圖
plot_trend_map(axes, trend_850, f'850 hPa Vorticity Trend ({start_year}-{end_year})')
#plot_trend_map(axes[1], trend_500, f'500 hPa Vorticity Trend ({start_year}-{end_year})')

plt.tight_layout()
plt.savefig('Fig5-5.png',dpi=400)
plt.show()

# ==========================================
# (選用) 儲存結果
# ==========================================
# output_nc = '/Dellwork4/cwuchiu/Vorticity_Trend_1979_2024.nc'
# ds_save = nc.Dataset(output_nc, 'w')
# ds_save.createDimension('lat', n_lat)
# ds_save.createDimension('lon', n_lon)
# ds_save.createVariable('lat', 'f4', ('lat',))[:] = lat
# ds_save.createVariable('lon', 'f4', ('lon',))[:] = lon
#
# v850_t = ds_save.createVariable('trend_vort850', 'f4', ('lat','lon'))
# v850_t.units = "s^-1 per decade"
# v850_t[:] = trend_850
#
# v500_t = ds_save.createVariable('trend_vort500', 'f4', ('lat','lon'))
# v500_t.units = "s^-1 per decade"
# v500_t[:] = trend_500
#
# ds_save.close()
# print("Trend data saved.")

#%%
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy import stats

# ==========================================
# 設定檔案路徑
# ==========================================
file_vws = '/Dellwork4/cwuchiu/TC_time_VWS_noET_34kt.nc'
file_vort = '/Dellwork4/cwuchiu/TC_time_vorticity_noET_34kt.nc'
file_land = '/Dellwork4/cwuchiu/land.nc'  # 新增 Land-Sea Mask 檔案

start_year = 1979
end_year = 2024
years = np.arange(start_year, end_year + 1)
n_years = len(years)

# ==========================================
# 1. 讀取基礎資訊 (座標與時間)
# ==========================================
# 讀取 VWS 檔取得座標資訊
ds_vws = nc.Dataset(file_vws)

# 先讀取變數物件
time_var_obj = ds_vws.variables['time']

# 在關檔前，先把數值和單位讀到記憶體變數中
time_values = time_var_obj[:]  
time_units = time_var_obj.units

lat = ds_vws.variables['lat'][:]
lon = ds_vws.variables['lon'][:]

ds_vws.close()

# 建立 2D 網格
LON_GRID, LAT_GRID = np.meshgrid(lon, lat)

# 解析時間
all_dates = nc.num2date(time_values, time_units)
all_years_index = np.array([d.year for d in all_dates])

# ==========================================
# 1.5. 建立靜態空間遮罩 (Spatial Mask)
# 條件: 純海洋 (LSM == 0) 且 緯度在 0-30度之間
# ==========================================
print("Loading Land-Sea Mask and creating spatial filters...")
ds_land = nc.Dataset(file_land)
# lsm shape is (valid_time, lat, lon) -> (1, 201, 321)
# 我們取 index 0 降維成 (201, 321)
lsm_data = ds_land.variables['lsm'][0, :, :] 

# 條件 A: 必須是海洋。GRIB的lsm介於0-1，<0.5 通常代表主要為海洋，這裡嚴格取等於0
# 轉為 int 比較保險，或是直接 lsm_data == 0
ocean_mask = (np.round(lsm_data) == 0) 

# 條件 B: 緯度介於 0 到 30 度
lat_mask = (LAT_GRID >= 0) & (LAT_GRID <= 30)

# 合併條件 A & B -> (201, 321) 的布林矩陣
valid_spatial_mask = ocean_mask & lat_mask

ds_land.close()
print(f"Spatial mask created. Active ocean grid points (0-30N): {np.sum(valid_spatial_mask)}")

# ==========================================
# 2. 準備儲存陣列
# ==========================================
# 儲存每年的平均重心: [0]=Lat, [1]=Lon
center_850 = np.full((n_years, 2), np.nan) # (Year, Lat/Lon)
center_500 = np.full((n_years, 2), np.nan)

# ==========================================
# 3. 主迴圈：逐年計算
# ==========================================
print(f"Start processing Center of Gravity ({start_year}-{end_year})...")

# 開啟檔案準備讀取
nc_vws = nc.Dataset(file_vws)
nc_vort = nc.Dataset(file_vort)
var_vws = nc_vws.variables['vws']
var_vort850 = nc_vort.variables['vort_850']
var_vort500 = nc_vort.variables['vort_500']

for i, yr in enumerate(years):
    # 找出該年的 indices
    indices = np.where(all_years_index == yr)[0]
    
    if len(indices) == 0:
        print(f"Warning: No data for year {yr}")
        continue
        
    idx_start = indices[0]
    idx_end = indices[-1] + 1
    
    # --- 讀取該年數據 ---
    # Shape: (Time, Lat, Lon)
    vws_data = var_vws[idx_start:idx_end, :, :]
    v850_data = var_vort850[idx_start:idx_end, :, :]
    v500_data = var_vort500[idx_start:idx_end, :, :]
    
    # 處理 Masked Array
    if np.ma.is_masked(vws_data): vws_data = vws_data.filled(np.nan)
    if np.ma.is_masked(v850_data): v850_data = v850_data.filled(np.nan)
    if np.ma.is_masked(v500_data): v500_data = v500_data.filled(np.nan)
    
    # --- 定義計算重心的函數 ---
    # 新增 static_mask 參數
    def calculate_weighted_center(vws, vort, lat_grid, lon_grid, static_mask):
        """
        計算一整年每個時間點的重心
        vws, vort shape: (Time, Lat, Lon)
        static_mask shape: (Lat, Lon)
        """
        # 1. 建立遮罩 (Mask): 
        # (VWS <= 10) AND (Vort > 0) AND (Ocean) AND (0 <= Lat <= 30)
        # static_mask[None, :, :] 負責廣播到所有 Time 維度
        condition = (vws <= 10) & (vort > 0) & static_mask[None, :, :]
        
        # 2. 計算權重 (Weights)
        # 只有符合條件的地方有權重，權重值為渦度大小
        weights = np.where(condition, vort, 0)
        
        # 3. 計算每個時間點的加權總和
        # axis=(1, 2) 代表對 lat, lon 軸加總，保留 Time 軸
        sum_weights = np.sum(weights, axis=(1, 2))
        
        # 避免除以 0 (若某個時間點完全沒有符合條件的區域)
        valid_times = sum_weights > 0
        
        if np.sum(valid_times) == 0:
            return np.nan, np.nan
        
        # 擴張 Grid 以符合 Time 維度 (Broadcasting)
        weighted_lat_sum = np.sum(weights * lat_grid[None, :, :], axis=(1, 2))
        weighted_lon_sum = np.sum(weights * lon_grid[None, :, :], axis=(1, 2))
        
        # 4. 計算每個時間點的重心
        center_lats_t = weighted_lat_sum[valid_times] / sum_weights[valid_times]
        center_lons_t = weighted_lon_sum[valid_times] / sum_weights[valid_times]
        
        # 5. 取該年所有有效時間點的平均
        year_mean_lat = np.mean(center_lats_t)
        year_mean_lon = np.mean(center_lons_t)
        
        return year_mean_lat, year_mean_lon

    # --- 計算 850 hPa 重心 (傳入 valid_spatial_mask) ---
    lat_850, lon_850 = calculate_weighted_center(vws_data, v850_data, LAT_GRID, LON_GRID, valid_spatial_mask)
    center_850[i, 0] = lat_850
    center_850[i, 1] = lon_850
    
    # --- 計算 500 hPa 重心 (傳入 valid_spatial_mask) ---
    lat_500, lon_500 = calculate_weighted_center(vws_data, v500_data, LAT_GRID, LON_GRID, valid_spatial_mask)
    center_500[i, 0] = lat_500
    center_500[i, 1] = lon_500
    
    print(f"Year {yr} done. (850hPa: {lat_850:.1f}N, {lon_850:.1f}E)", end='\r')

nc_vws.close()
nc_vort.close()
print("\nCalculation finished.")
#%%
# ==========================================
# 4. 繪圖 (四張子圖)
# ==========================================
def plot_trend_subplot(ax, x, y, title, ylabel, color):
    # 繪製折線
    ax.plot(x, y, color=color, marker='o', markersize=4, linestyle='-', alpha=0.6)
    
    # 計算並繪製趨勢線
    # 過濾 NaN
    valid = ~np.isnan(y)
    if np.sum(valid) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])
        trend_line = slope * x + intercept
        ax.plot(x, trend_line, color='black', linestyle='--', linewidth=2, 
                label=f'Trend: {slope*10:.2f} deg/decade\n(p={p_value:.3f})')
        ax.legend(loc='best')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xlim(start_year, end_year)



# 建立畫布
fig, axes = plt.subplots(2, figsize=(12, 8), sharex=True)

# 經度 (Longitude)
plot_trend_subplot(axes[0], years, center_850[:, 1], 
                   '(a) 850 hPa Weighted Center - Longitude', 'Longitude (°E)', 'blue')

# 緯度 (Latitude)
plot_trend_subplot(axes[1], years, center_850[:, 0], 
                   '(b) 850 hPa Weighted Center - Latitude', 'Latitude (°N)', 'green')

plt.xlabel('Year')
#plt.tight_layout()
#plt.savefig('Genesis_Center_Shift_1979-2024.png')
plt.savefig('Fig5-2.png')
plt.show()

fig, axes = plt.subplots(2, figsize=(10, 8), sharex=True)

# 經度 (Longitude)
plot_trend_subplot(axes[0], years, center_500[:, 1], 
                   '500 hPa Weighted Center - Longitude', 'Longitude (°E)', 'blue')

# 緯度 (Latitude)
plot_trend_subplot(axes[1], years, center_500[:, 0], 
                   '500 hPa Weighted Center - Latitude', 'Latitude (°N)', 'blue')

plt.xlabel('Year')
#plt.tight_layout()
#plt.savefig('Genesis_Center_Shift_500_1979-2024.png')
plt.show()
#%%
# 印出簡單統計
print("\n=== Trend Summary (per decade) ===")
s_lat850, _, _, p_lat850, _ = stats.linregress(years[~np.isnan(center_850[:,0])], center_850[~np.isnan(center_850[:,0]), 0])
s_lon850, _, _, p_lon850, _ = stats.linregress(years[~np.isnan(center_850[:,1])], center_850[~np.isnan(center_850[:,1]), 1])
print(f"850 hPa: Lat {s_lat850*10:.2f} deg/dec (p={p_lat850:.3f}), Lon {s_lon850*10:.2f} deg/dec (p={p_lon850:.3f})")

s_lat500, _, _, p_lat500, _ = stats.linregress(years[~np.isnan(center_500[:,0])], center_500[~np.isnan(center_500[:,0]), 0])
s_lon500, _, _, p_lon500, _ = stats.linregress(years[~np.isnan(center_500[:,1])], center_500[~np.isnan(center_500[:,1]), 1])
print(f"500 hPa: Lat {s_lat500*10:.2f} deg/dec (p={p_lat500:.3f}), Lon {s_lon500*10:.2f} deg/dec (p={p_lon500:.3f})")
ncfile.close()
#%%
# the northest and southest ridge
ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_WNPSH_weighted_ridge_lat.nc')
ridge = np.array(ncfile.variables['hgt_adjust_ridge_lat'])
ncfile.close()

# 讀取時間並處理重複
ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_1979_2024_not_set.nc')
tt = np.array(ncfile.variables['time'])
time_tc = [str(x) for x in tt]
ncfile.close()

time_arr = np.array(time_tc)
_, unique_indices = np.unique(time_arr, return_index=True)
unique_indices = np.sort(unique_indices)
for i in range(unique_indices.shape[0]):
    unique_indices[i] = i

# 取得乾淨的時間與 Ridge
time_clean = time_arr[unique_indices]
ridge_clean = ridge[unique_indices]


nc_vort = nc.Dataset('/Dellwork4/cwuchiu/TC_time_vorticity.nc')
vort_var = nc_vort.variables['vort_850'] 
lat = nc_vort.variables['lat'][:]
lon = nc_vort.variables['lon'][:]


valid_mask = ~np.isnan(ridge_clean)
ridge_valid = ridge_clean[valid_mask]
indices_valid = unique_indices[valid_mask] # 這是對應回原始 NetCDF 檔案的索引

print(f"原始不重複時間點: {len(ridge_clean)}")
print(f"扣除 NaN 後有效時間點: {len(ridge_valid)}")

# 計算百分位數 (95% 為偏北，5% 為偏南)
p95 = np.percentile(ridge_valid, 95)
p05 = np.percentile(ridge_valid, 5)

print(f"Top 5% (Ridge偏北) 閾值: >= {p95:.2f} 度")
print(f"Bottom 5% (Ridge偏南) 閾值: <= {p05:.2f} 度")

# 找出符合條件的「原始檔案索引 (File Indices)」
# 這樣我們可以直接去 NetCDF 抓資料，不用把整顆 6萬步的 vort 讀進 RAM
idx_high = indices_valid[ridge_valid >= p95]
idx_low  = indices_valid[ridge_valid <= p05]

print(f"High Case 樣本數: {len(idx_high)}")
print(f"Low Case 樣本數: {len(idx_low)}")

# ==========================================
# 4. 計算合成平均 (Composite Mean)
# ==========================================
# 這裡使用 NetCDF 的 fancy indexing 直接讀取特定時間步
# 如果記憶體夠大 (64G以上)，也可以先讀全部再切，但這樣寫比較保險

print("Reading Vorticity data for High Composite...")
# 讀取並直接計算平均 (axis=0) 以節省記憶體
vort_high_comp = np.mean(vort_var[idx_high], axis=0)

print("Reading Vorticity data for Low Composite...")
vort_low_comp = np.mean(vort_var[idx_low], axis=0)

# 計算差異 (High - Low)
vort_diff = vort_high_comp - vort_low_comp

nc_vort.close()

# ==========================================
# 5. 繪圖
# ==========================================
def plot_composite(ax, data, title, cmap='RdBu_r', levels=None):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
    
    # 設置經緯度標籤
    ax.set_xticks(np.arange(100, 181, 20), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 51, 10), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    cf = ax.contourf(lon, lat, data, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=12, fontweight='bold')
    return cf

# 設定繪圖畫布
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})

# 設定 colorbar 的範圍 (視您的數據大小調整，渦度通常是 1e-5 量級)
# 這裡先自動抓最大值來對稱
max_val = max(np.abs(np.nanmax(vort_high_comp)), np.abs(np.nanmax(vort_low_comp)))
levels = np.linspace(-max_val, max_val, 21)

# 1. High Ridge (Northward)
cf1 = plot_composite(axes[0], vort_high_comp, f'Top 5% Ridge Lat (>= {p95:.1f}$^o$N)', cmap='RdBu_r', levels=levels)
plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.1, label='Vorticity ($s^{-1}$)')

# 2. Low Ridge (Southward)
cf2 = plot_composite(axes[1], vort_low_comp, f'Bottom 5% Ridge Lat (<= {p05:.1f}$^o$N)', cmap='RdBu_r', levels=levels)
plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.1, label='Vorticity ($s^{-1}$)')

# 3. Difference (High - Low)
# 差異圖通常範圍比較小，重新設定 levels
diff_max = np.abs(np.nanmax(vort_diff)) * 0.8
diff_levels = np.linspace(-diff_max, diff_max, 21)

cf3 = plot_composite(axes[2], vort_diff, 'Difference (High - Low)', cmap='PuOr_r', levels=diff_levels)
plt.colorbar(cf3, ax=axes[2], orientation='horizontal', pad=0.1, label='Diff Vorticity ($s^{-1}$)')

plt.suptitle('850hPa Vorticity Composite by WNPSH Ridge Latitude', fontsize=16)
plt.tight_layout()
plt.savefig('north_south_ridge_vorticity.png')
plt.show()

#%%
def calculate_centroid(vort_data, lon_1d, lat_1d):
    """
    計算渦度場的加權中心 (Centroid)
    """
    # 1. 建立 2D 網格
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
    
    # 2. 前處理：只取正渦度 (Positive Vorticity Only)
    # 我們假設要找的是氣旋式環流中心 (季風槽)
    # 如果渦度很小 (接近0)，也可設一個閾值濾掉雜訊，例如 1e-6
    vort_pos = np.copy(vort_data)
    vort_pos[vort_pos < 0] = 0 
    
    # 3. 計算面積權重 (Area Weighting)
    # 因為地球是球體，不同緯度的網格面積不同，需乘上 cos(lat)
    area_weights = np.cos(np.deg2rad(lat_2d))
    
    # 4. 總權重 = 渦度強度 * 網格面積
    # 這裡代表「質量」
    W = vort_pos * area_weights
    
    total_weight = np.sum(W)
    
    if total_weight == 0:
        return np.nan, np.nan
        
    # 5. 計算重心公式: Sum(X * W) / Sum(W)
    center_lon = np.sum(lon_2d * W) / total_weight
    center_lat = np.sum(lat_2d * W) / total_weight
    
    return center_lon, center_lat

# ==========================================
# 使用範例
# ==========================================

# 計算 High Ridge Case 的渦度中心
cen_lon_high, cen_lat_high = calculate_centroid(vort_high_comp, lon, lat)

# 計算 Low Ridge Case 的渦度中心
cen_lon_low, cen_lat_low = calculate_centroid(vort_low_comp, lon, lat)

print(f"High Ridge Case -> Vorticity Center: ({cen_lon_high:.2f}, {cen_lat_high:.2f})")
print(f"Low Ridge Case  -> Vorticity Center: ({cen_lon_low:.2f}, {cen_lat_low:.2f})")

# 驗證物理機制
if cen_lat_high > cen_lat_low:
    print("結果合理：副高北抬導致季風槽渦度中心北移。")
else:
    print("結果異常：需檢查渦度場分佈。")
        
        
        
# 設定繪圖畫布
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), subplot_kw={'projection': ccrs.PlateCarree()})

# 設定 colorbar 的範圍 (視您的數據大小調整，渦度通常是 1e-5 量級)
# 這裡先自動抓最大值來對稱
max_val = max(np.abs(np.nanmax(vort_high_comp)), np.abs(np.nanmax(vort_low_comp)))
levels = np.linspace(-max_val, max_val, 21)

# 1. High Ridge (Northward)
cf1 = plot_composite(axes[0], vort_high_comp, f'Top 5% Ridge Lat (>= {p95:.1f}$^o$N)\nVorticity Center: ({cen_lon_high:.2f}, {cen_lat_high:.2f})', cmap='RdBu_r', levels=levels)
plt.colorbar(cf1, ax=axes[0], orientation='horizontal', pad=0.1, label='Vorticity ($s^{-1}$)')

# 2. Low Ridge (Southward)
cf2 = plot_composite(axes[1], vort_low_comp, f'Bottom 5% Ridge Lat (<= {p05:.1f}$^o$N)\nVorticity Center: ({cen_lon_low:.2f}, {cen_lat_low:.2f})', cmap='RdBu_r', levels=levels)
plt.colorbar(cf2, ax=axes[1], orientation='horizontal', pad=0.1, label='Vorticity ($s^{-1}$)')

# 3. Difference (High - Low)
# 差異圖通常範圍比較小，重新設定 levels
diff_max = np.abs(np.nanmax(vort_diff)) * 0.8
diff_levels = np.linspace(-diff_max, diff_max, 21)

cf3 = plot_composite(axes[2], vort_diff, 'Difference (High - Low)', cmap='PuOr_r', levels=diff_levels)
plt.colorbar(cf3, ax=axes[2], orientation='horizontal', pad=0.1, label='Diff Vorticity ($s^{-1}$)')

plt.suptitle('850hPa Vorticity Composite by WNPSH Ridge Latitude', fontsize=16)
plt.tight_layout()
plt.savefig('north_south_ridge_vorticity.png')
plt.show()      
        
        
#%%
ncfile = nc.Dataset('/Dellwork4/cwuchiu/Cluster4_WNPSH_5880.nc')
which_cluster = np.array(ncfile.variables['which_cluster']) 
ncfile.close()



ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_1979_2024_not_set.nc')
tt = np.array(ncfile.variables['time'])
ncfile.close()




tt = np.array(tt)
which_cluster = np.array(which_cluster)

# 找 unique 時間的 index（保留第一次出現）
_, unique_indices = np.unique(tt, return_index=True)

# 依照原順序排序
unique_indices = np.sort(unique_indices)

# 去除重複
tt_unique = tt[unique_indices]
time_tc =[str(x) for x in tt_unique]

which_cluster_unique = which_cluster[unique_indices]


ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_vorticity.nc')

vo850 = np.array(ncfile.variables['vort_850'])

target_years= np.linspace(1979,2024,46)
# ================= 2. 資料處理與計算 =================

# 擷取每筆資料的「年份」 (從 YYYYMMDDHH 取前四個字元並轉為整數)
years_tc = np.array([int(t[:4]) for t in time_tc])

# 建立一個陣列儲存每年(46年)、各群(5群, index 0~4)的脊線緯度平均值
# 先全部填入 NaN，避免某年某群剛好沒資料時算成 0
mean_vo = np.full((46, 5, 201,321), np.nan)

# 開始計算每年、各群的平均值
for i, year in enumerate(target_years):
    for c in range(1, 5):  # 只算 Cluster 1 到 4
        # 條件：年份相符 & Cluster相符 & 脊線緯度不是 NaN
        condition = (years_tc == year) & (which_cluster_unique == c)
        
        # 如果該年該群有符合條件的資料，就取平均
        if np.any(condition):
            mean_vo[i, c,:,:] = np.mean(vo850[condition],axis=0)

#%%
# ================= 3. 繪圖 (Cluster 1~4 的內部脊線趨勢) =================

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

lon = np.linspace(100,180,321)
lat = np.linspace(50,0,201)

for c_idx in range(1, 5):  # c_idx 為 1, 2, 3, 4
    ax = axes[c_idx - 1]
    
    # 取出該 Cluster 的 46 年平均脊線緯度資料
    y_data = mean_vo[:, c_idx,:,:]
    
    time = np.linspace(1979,2024,46)
    time_mean = np.mean(time)
    vo_mean = np.mean(y_data,axis=0)
    numerator_direct = np.sum((time[:, None, None] - time_mean) * (y_data - vo_mean[None, :, :]), axis=0)
    denominator_direct = np.sum((time[:, None, None] - time_mean)**2, axis=0)

    dvo850dt = numerator_direct / denominator_direct
    norm = TwoSlopeNorm(vmin=-1e-9,vcenter=0,vmax=1e-9)
    custom_bwr = mcolors.LinearSegmentedColormap.from_list(
    'custom_bwr', [(0, 'blue'), (0.45, 'white'), (0.55, 'white'), (1, 'red')]
    )
    
    ct = ax.pcolormesh(
    lon,lat,dvo850dt,
    cmap=custom_bwr,
    norm=norm,
    alpha=1
    )

    plt.colorbar(ct, ax=ax, ticks=np.linspace(-1e-9,1e-9,11), orientation='vertical', label='850 hPa vorticity trend ($s^{-1} per year$)')
    
        
    # 圖表美化設定
    ax.set_title(f'Cluster {c_idx} - WNPSH Ridge Latitude Trend', fontsize=13, fontweight='bold')
    ax.set_ylabel('Ridge Latitude (°N)')
    if c_idx >= 3:
        ax.set_xlabel('Year')
    
    #ax.legend(loc='best', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('cluster_vortivity_trend.png')
plt.show()
      
#ncfile.close() 
#%%
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

lon = np.linspace(100,180,321)
lat = np.linspace(50,0,201)

lon_bins = np.linspace(100, 180, 33)  # 示例
lat_bins = np.linspace(0, 50, 21)

lon_grid, lat_grid = np.meshgrid(lon_bins[:-1], lat_bins[:-1])  # shape: (20, 32)



# KDE 計算函數
def compute_kde(lon, lat, mask):
    x = lon[mask]
    y = lat[mask]
    if len(x) == 0:
        return np.zeros_like(lon)
    coords = np.vstack([x, y])
    kernel = gaussian_kde(coords,bw_method='silverman')
    positions = np.vstack([lon.ravel(), lat.ravel()])
    z = np.reshape(kernel(positions), lon.shape)
    return z

for c_idx in range(1, 5):  # c_idx 為 1, 2, 3, 4
    ax = axes[c_idx - 1]
    
    # 取出該 Cluster 的 46 年平均脊線緯度資料
    y_data = mean_vo[:, c_idx,:,:]
    
    vo_diff = np.mean(y_data[23:46]-y_data[0:23],axis=0)
    
    vo_coarse = vo_diff[:200, :320] \
                .reshape(20, 10, 32, 10) \
                .mean(axis=(1, 3))
    vo_coarse = vo_coarse[::-1,:]
    
    z_increase = compute_kde(lon_grid, lat_grid, vo_coarse > 0)
    z_decrease = -compute_kde(lon_grid, lat_grid, vo_coarse < 0)  # 負號讓減少為負值
    z_combined = z_increase + z_decrease
    
    
    norm = TwoSlopeNorm(vmin=-5e-4,vcenter=0,vmax=5e-4)
    custom_bwr = mcolors.LinearSegmentedColormap.from_list(
    'custom_bwr', [(0, 'blue'), (0.45, 'white'), (0.55, 'white'), (1, 'red')]
    )
    
    ct = ax.pcolormesh(
    lon_grid,lat_grid,z_combined,
    cmap=custom_bwr,
    norm=norm,
    alpha=1
    )

    plt.colorbar(ct, ax=ax, ticks=np.linspace(-5e-4,5e-4,11), orientation='vertical', label='850 hPa vorticity trend ($s^{-1} per year$)')
    
        
    # 圖表美化設定
    ax.set_title(f'Cluster {c_idx}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Ridge Latitude (°N)')
    if c_idx >= 3:
        ax.set_xlabel('Year')
    
    #ax.legend(loc='best', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)
plt.suptitle('Vorticity difference (2002-2024 minus 1979-2001)',fontweight='bold',fontsize=16)
plt.tight_layout()
plt.savefig('cluster_vortivity_diff.png')
plt.show()