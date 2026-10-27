from numpy import where, linspace,nan,meshgrid, array, squeeze, where, min, pi, stack,zeros, loadtxt, arcsin, mean, concatenate, sum, nanmin, nanmean, nanmax, array
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

def Tvbar_fit_curve(year,month):
    nc_coef = nc.Dataset('/Dellwork4/cwuchiu/fit_curve_deg4_2004.nc')
    fit = array(nc_coef.variables['fit_curve'])
    m = (year-2004)*6+month-6
    Tvbar_diff = fit[m] - fit[0]
    nc_coef.close()
    return Tvbar_diff

ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_1979_2024_not_set_noET_34kt.nc')
tt = np.array(ncfile.variables['time'])
time_tc = [str(x) for x in tt]
ncfile.close()

import numpy as np
import netCDF4 as nc


# === 1. 設定路徑與網格 ===
input_file_path = '/Dellwork4/cwuchiu/Tvbar_hourly.nc'
output_file_path = '/Dellwork4/cwuchiu/Tvbar_hgt_WNPSH_2004_not_set_noET_34kt.nc'

lon = np.linspace(100, 180, 321)
lat = np.linspace(50, 0, 201)

# === 2. 建立輸出的 NetCDF 檔案 ===
total_timesteps = len(time_tc)  # 依據 time_tc 的時間步數 (例如 41682)

out_nc = nc.Dataset(output_file_path, 'w', format='NETCDF4')

# 建立維度
out_nc.createDimension('time', total_timesteps)
out_nc.createDimension('lat', len(lat))
out_nc.createDimension('lon', len(lon))

# 建立座標變數
var_lat = out_nc.createVariable('lat', 'f4', ('lat',))
var_lat.units = 'degrees_north'
var_lat.long_name = 'Latitude'
var_lat[:] = lat

var_lon = out_nc.createVariable('lon', 'f4', ('lon',))
var_lon.units = 'degrees_east'
var_lon.long_name = 'Longitude'
var_lon[:] = lon

# 建立時間標籤字串變數 (儲存 time_tc)
var_time = out_nc.createVariable('time', str, ('time',))
var_time.long_name = 'Time string (YYYYMMDDHH)'
var_time[:] = np.array(time_tc, dtype=object)

# 建立 500hPa 重力位高度變數 (開啟壓縮與分塊優化)
var_hgt = out_nc.createVariable(
    'hgt500_adjust', 'f4', ('time', 'lat', 'lon'),
    zlib=True, complevel=4, 
    chunksizes=(24, len(lat), len(lon))  # 每次存取以 1 天 (24小時) 為單位
)
var_hgt.units = 'gpm'
var_hgt.long_name = 'Adjusted 500hPa Geopotential Height'
var_hgt.description = '500hPa geopotential height adjusted with Tvbar curve for year >= 2004'

# === 3. 批次讀取、校正並寫入 ===
in_nc = nc.Dataset(input_file_path, 'r')
hgt500 = in_nc.variables['hgt500']

# 每月天數定義 (6~11月)
days_in_month = {6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30}

i_global = 0      # 原始 hgt500 的全局時間指標
ii_tc = 0         # time_tc 的指標
out_idx = 0       # 寫入 output_nc 的指標

print("Start processing and saving adjusted hgt500...")

for year in trange(1979, 2025):
    for month in range(6, 12):
        n_days = days_in_month[month]
        n_month_hours = n_days * 24
        
        # 1. 一次讀取整個月的資料 (Shape: [n_month_hours, 201, 321])
        hgt_chunk = hgt500[i_global : i_global + n_month_hours].astype(np.float32)
        
        # 2. 進行 2004 年之後的校正
        if year >= 2004:
            offset = float(Tvbar_fit_curve(year, month) * 20.3)
            hgt_chunk -= offset
            
        # 3. 建立該月每小時的時間戳字串對照表 (YYYYMMDDHH)
        cur_month_times = {
            f"{year}{month:02d}{d:02d}{h:02d}": idx
            for idx, (d, h) in enumerate(
                [(d, h) for d in range(1, n_days + 1) for h in range(24)]
            )
        }
        
        # 4. 比對 time_tc，找出屬於該月份的時間點
        matched_indices = []
        while ii_tc < total_timesteps:
            tc_str = str(time_tc[ii_tc])
            if tc_str in cur_month_times:
                matched_indices.append(cur_month_times[tc_str])
                ii_tc += 1
            else:
                break  # 該月份在 time_tc 中的時間點已比對完畢，換下個月
        
        # 5. 批次寫入 output nc 檔案
        if len(matched_indices) > 0:
            selected_hgt = hgt_chunk[matched_indices]
            var_hgt[out_idx : out_idx + len(matched_indices)] = selected_hgt
            out_idx += len(matched_indices)
            
        # 前進全局讀取指標
        i_global += n_month_hours

# 關閉檔案
in_nc.close()
out_nc.close()

print(f"\n[Done] Successfully saved {out_idx} hourly adjusted HGT fields to: {output_file_path}")