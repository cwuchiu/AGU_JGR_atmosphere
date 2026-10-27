#%%
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
#%%
def Tvbar_fit_curve(year,month):
    nc_coef = nc.Dataset('/Dellwork4/cwuchiu/fit_curve_deg4_2004.nc')
    fit = array(nc_coef.variables['fit_curve'])
    m = (year-2004)*6+month-6
    Tvbar_diff = fit[m] - fit[0]
    nc_coef.close()
    return Tvbar_diff
#%%
ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_1979_2024_not_set_noET_34kt.nc')
tt = np.array(ncfile.variables['time'])
time_tc = [str(x) for x in tt]
ncfile.close()
#%%
import numpy as np
import netCDF4 as nc
from tqdm import trange
import os

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

#%%
ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_WNPSH_1979-2024_2004_not_set_noET_34kt.nc')
east = array(ncfile.variables['east_monthly'])
west = array(ncfile.variables['west_monthly'])
north = array(ncfile.variables['north_monthly'])
south = array(ncfile.variables['south_monthly'])

Lat = array(ncfile.variables['latitude'])
Lon = array(ncfile.variables['longitude'])

ncfile.close()

WNP = [east,west,north,south]
SH = np.zeros((4, 276)) #E W N S
for sb in range(4):
    reshaped = np.zeros((46, 6))
    kkk=0
    for i in range(46):
        for j in range(6):
            tt = WNP[sb]
            reshaped[i,j] = tt[kkk]
            kkk+=1
            
    
    
    SeasonalCycle = np.nanmean(reshaped, axis=0)
    
    # 計算 anomaly：原始資料 - 平均季節循環（broadcasting）
    Anomaly_full = reshaped - SeasonalCycle  # shape = (10, 4320, 201, 321)
    
    anomaly = Anomaly_full.reshape(276)
    SH[sb] = anomaly


title = ['east edge anomaly','west edge anomaly','north edge anomaly','south edge anomaly']
time = np.linspace(1979,2024,276)

for i in range(4):
    mask = ~np.isnan(SH[i])
    save=SH[i]
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(time[mask], save[mask])
    fig, ax = plt.subplots(figsize=(10, 6))
    
    
    plt.plot(time,SH[i])
    plt.plot(time,time*slope1+intercept1,'r')
    
    
    plt.xlabel('time (yr)',fontsize = 18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if i==0:
        ax.text(0.02, 0.09, f"p-value={p_value1:.2e}",
        transform=ax.transAxes,  # 使用軸座標系統
        fontsize=18,
        va='top', ha='left',     # 對齊方式：上對齊、左對齊
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))  # 可選：加背景框讓文字更清晰

        #plt.text(2000,160,f"p-value={p_value1:.2e}",fontsize=18)
        plt.ylabel('Longitude',fontsize = 18)
    elif i==1:
        ax.text(0.02, 0.09, f"p-value={p_value1:.2e}",
        transform=ax.transAxes,  # 使用軸座標系統
        fontsize=18,
        va='top', ha='left',     # 對齊方式：上對齊、左對齊
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))  # 可選：加背景框讓文字更清晰
        plt.ylabel('Longitude',fontsize = 18)
    elif i==2:
        ax.text(0.02, 0.09, f"p-value={p_value1:.2e}",
        transform=ax.transAxes,  # 使用軸座標系統
        fontsize=18,
        va='top', ha='left',     # 對齊方式：上對齊、左對齊
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))  # 可選：加背景框讓文字更清晰
        plt.ylabel('Latitude',fontsize = 18)
    else:
        ax.text(0.02, 0.09, f"p-value={p_value1:.2e}",
        transform=ax.transAxes,  # 使用軸座標系統
        fontsize=18,
        va='top', ha='left',     # 對齊方式：上對齊、左對齊
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))  # 可選：加背景框讓文字更清晰
        plt.ylabel('Latitude',fontsize = 18)
    
    plt.title(title[i],fontsize=18)
    plt.savefig('WNPSH_shift_adjust_TC_1979-2024_ANOMALY_'+title[i]+'.png',dpi=400)
    plt.tight_layout()
    plt.show()
#%%
ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_WNPSH_1979-2024_2004_noET_34kt.nc')
hgt_adjust = array(ncfile.variables['HGT_monthly'])
ncfile.close()
hgt_adjust_yr = np.zeros((46,201,321))
for i in range(46):
    hgt_adjust_yr[i]=np.nanmean(hgt_adjust[6*i:6*(i+1)],axis=0)
time = np.linspace(1979,2024,46)
time_mean = np.mean(time)
hgt_adjust_mean = np.mean(hgt_adjust_yr,axis=0)
numerator_direct = np.sum((time[:, None, None] - time_mean) * (hgt_adjust_yr - hgt_adjust_mean[None, :, :]), axis=0)
denominator_direct = np.sum((time[:, None, None] - time_mean)**2, axis=0)

dhgt500dt = numerator_direct / denominator_direct

p_value = np.zeros((201,321))
for i in range(201):
    for j in range(321):
        _, _, _, p_value[i,j], _ = linregress(time, hgt_adjust_yr[:,i,j])

lon = np.linspace(100,180,321)
lat = np.linspace(0,50,201)[::-1]

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

levels = np.arange(-0.3, 1.51, 0.1)
norm = mcolors.TwoSlopeNorm(vmin=levels.min(), vcenter=0.0, vmax=levels.max())

plt.rcParams['hatch.color'] = 'gray'
ax.contourf(
    lon, lat, p_value,
    levels=[0,0.01],
    colors='none',            # 不填色，只顯示斜線
    hatches=['////'],  # 斜線樣式
)
cf = ax.contourf(
    lon, lat, dhgt500dt,
    levels=levels,
    cmap='RdBu_r',   # 或 'bwr'/'seismic'，皆為負藍正紅的發散色盤
    norm=norm,
    alpha=0.5,
    extend='both'    # 讓超出範圍的值在色棒末端以尖角標示
)
cs = ax.contour(lon, lat, dhgt500dt, levels=levels, colors='k', linewidths=0.8)
ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
ax.coastlines(resolution='50m', linewidth=1)
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.1, aspect=40)
cbar.set_label('hgt500 trend (gpm/year)')


ax.set_title('500hPa geopotential height trend (1979–2024 TC time)', fontsize=12)
plt.savefig('500z_adjust_TC_1979-2024.png',dpi=400)
plt.show()
