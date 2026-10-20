#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:26:26 2026

@author: cwuchiu
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as mcolors
from scipy import stats
import pymannkendall as mk
import time as time_lib
import netCDF4 as nc



# Figure 6
ncfile = nc.Dataset('/Dellwork4/cwuchiu/SHAI_km2_HGT_Lon_Lat_1979_2024_monthly_hourly_adjust_mean_2004.nc')
hgt_adjust = np.array(ncfile.variables['HGT_monthly'])
ncfile.close()

year_num = 46
hgt_adjust_yr = np.zeros((year_num, 201, 321))

for i in range(year_num):
    if i>=31:
        hgt_adjust_yr[i]=np.nanmean(hgt_adjust[6*i:6*(i+1)],axis=0)/9.81*9.80
    else:
        hgt_adjust_yr[i]=np.nanmean(hgt_adjust[6*i:6*(i+1)],axis=0)

time = np.linspace(2024 - year_num + 1, 2024, year_num)
lon = np.linspace(100, 180, 321)
lat = np.linspace(0, 50, 201)[::-1]

# ==========================================
# 2. 計算 Theil-Sen 趨勢與 Mann-Kendall 檢定
# ==========================================
dhgt500dt_sen = np.zeros((201, 321))
p_value_mk = np.zeros((201, 321))

print("開始計算 Theil-Sen 趨勢與 Mann-Kendall 檢定... (約需1-3分鐘)")
start_time = time_lib.time()

for i in range(201):
    # 每處理 50 條緯線顯示一次進度，避免以為程式當機
    if i % 50 == 0:
        print(f"處理進度: 第 {i}/201 列緯度...")
        
    for j in range(321):
        ts_data = hgt_adjust_yr[:, i, j]
        
        # 1. Theil-Sen Estimator 估算斜率 (Trend)
        # res[0] 是斜率, res[1] 是截距, res[2]/res[3] 是上下界
        res = stats.theilslopes(ts_data, time, alpha=0.95)
        dhgt500dt_sen[i, j] = res[0]
        
        # 2. Mann-Kendall 檢定計算 P-value
        # mk_test.p 就是無母數檢定的 p-value
        mk_test = mk.original_test(ts_data)
        p_value_mk[i, j] = mk_test.p

ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_1979_2024_not_set_noET_34kt.nc')
tt = np.array(ncfile.variables['time'])
time_tc =[str(x) for x in tt]
ncfile.close()


ncfile = nc.Dataset('/Dellwork4/cwuchiu/Tvbar_hgt_WNPSH_2004_not_set_noET_34kt.nc')

hgt_adj = ncfile.variables['hgt500_adjust']

hgt = np.zeros((276,201,321))
month_count = np.zeros(276)




for i in range(tt.shape[0]):
    month = int((int(time_tc[i][0:4])-1979)*6+int(time_tc[i][4:6])-6)
    hgt[month]+= hgt_adj[i]
    month_count[month]+=1
hgt_monthly = hgt/month_count[:,None,None]



year_num = 46
hgt_adjust_yr_TC = np.zeros((year_num, 201, 321))

for i in range(year_num):
    if i>=31:
        hgt_adjust_yr_TC[i]=np.nanmean(hgt_monthly[6*i:6*(i+1)],axis=0)/9.81*9.80
    else:
        hgt_adjust_yr_TC[i]=np.nanmean(hgt_monthly[6*i:6*(i+1)],axis=0)


time = np.linspace(2024 - year_num + 1, 2024, year_num)
lon = np.linspace(100, 180, 321)
lat = np.linspace(0, 50, 201)[::-1]

# ==========================================
# 2. 計算 Theil-Sen 趨勢與 Mann-Kendall 檢定
# ==========================================
dhgt500dt_sen_TC = np.zeros((201, 321))
p_value_mk_TC = np.zeros((201, 321))

print("開始計算 Theil-Sen 趨勢與 Mann-Kendall 檢定... (約需1-3分鐘)")
start_time = time_lib.time()

for i in range(201):
    # 每處理 50 條緯線顯示一次進度，避免以為程式當機
    if i % 50 == 0:
        print(f"處理進度: 第 {i}/201 列緯度...")
        
    for j in range(321):
        ts_data = hgt_adjust_yr_TC[:, i, j]
        
        # 1. Theil-Sen Estimator 估算斜率 (Trend)
        # res[0] 是斜率, res[1] 是截距, res[2]/res[3] 是上下界
        res = stats.theilslopes(ts_data, time, alpha=0.95)
        dhgt500dt_sen_TC[i, j] = res[0]
        
        # 2. Mann-Kendall 檢定計算 P-value
        # mk_test.p 就是無母數檢定的 p-value
        mk_test = mk.original_test(ts_data)
        p_value_mk_TC[i, j] = mk_test.p

print(f"計算完成！耗時: {time_lib.time() - start_time:.2f} 秒")


# ==========================================
# 1. 建立 2x1 畫布與子圖，並設定投影方式
# ==========================================
# 💡 修改重點 1：加入 layout='constrained'，讓系統自動完美計算排版與間距
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), 
                         subplot_kw={'projection': ccrs.PlateCarree()},
                         layout='constrained')

x_extent = np.linspace(-180, 180, 37)
y_extent = np.linspace(-90, 90, 19)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

# 設定等值線與發散顏色盤
levels = np.arange(-0.2, 1.01, 0.1)
norm = mcolors.TwoSlopeNorm(vmin=levels.min(), vcenter=0.0, vmax=levels.max())
plt.rcParams['hatch.color'] = 'gray'

# ==========================================
# 2. 將兩張圖的資料與標題整理成 List，方便用迴圈繪製
# ==========================================
plot_data = [
    {
        'data': dhgt500dt_sen_TC, 
        'p_value': p_value_mk_TC, 
        'title': f'(b) 500 hPa Geopotential Height Trend ({int(time[0])}–{int(time[-1])} TC time)'
    },
    {
        'data': dhgt500dt_sen, 
        'p_value': p_value_mk, 
        'title': f'(a) 500 hPa Geopotential Height Trend ({int(time[0])}–{int(time[-1])})'
    }
][::-1]

# ==========================================
# 3. 逐一繪製子圖
# ==========================================
for i, ax in enumerate(axes):
    # 地圖範圍與刻度
    ax.set_xticks(x_extent, crs=ccrs.PlateCarree())
    ax.set_yticks(y_extent, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
    
    # --- 第一層：繪製 Theil-Sen 趨勢填色 ---
    cf = ax.contourf(
        lon, lat, plot_data[i]['data'],
        levels=levels,
        cmap='RdBu_r', 
        norm=norm,
        alpha=0.9,
        extend='both',
        transform=ccrs.PlateCarree()
    )
    
    # --- 第二層：繪製等值線邊界 ---
    cs = ax.contour(
        lon, lat, plot_data[i]['data'], 
        levels=levels, 
        colors='k', 
        linewidths=0.8,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(cs, inline=True, fontsize=10, fmt="%.2f")
    
    # --- 第三層：繪製 Mann-Kendall 顯著性檢定網底 (p < 0.05) ---
    ax.contourf(
        lon, lat, plot_data[i]['p_value'],
        levels=[0, 0.05],     
        colors='none', 
        hatches=['////'], 
        transform=ccrs.PlateCarree()
    )
    
    # 子圖的外觀調整
    ax.tick_params(axis='both', labelsize=14)
    ax.coastlines(resolution='50m', linewidth=1, alpha=0.8)
    ax.set_title(plot_data[i]['title'], fontsize=16, fontweight='bold', y=1.03)

# ==========================================
# 4. 加入共用 Colorbar 並儲存
# ==========================================
# 💡 修改重點 2：將 orientation 改為 location='bottom'，不需要手動設 pad，引擎會自動處理
cbar = fig.colorbar(cf, ax=axes, location='bottom', aspect=40, shrink=0.5)
cbar.set_label('500 hPa geopotential height trend (gpm $yr^{-1}$)', fontsize=16)
cbar.ax.tick_params(labelsize=14)

# 💡 修改重點 3：刪除 plt.subplots_adjust(hspace=0.25)
# 因為 layout='constrained' 會自動依據標題和刻度的文字大小，騰出剛好不會重疊的空間，手動 adjust 反而會破壞它

plt.savefig('FigureS3.png', dpi=400, bbox_inches='tight')
plt.show()
