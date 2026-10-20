#%% ==================== 1. 模組匯入 (Imports) ====================
import math
from collections import Counter
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import scipy.stats as st
from cartopy import crs as ccrs, feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from numpy import (
    arcsin, array, linspace, loadtxt, meshgrid, nan, pi, squeeze, stack,
    where, zeros
)
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, linregress, ttest_ind

#==================== 2. 讀取 NetCDF 資料 ====================
# 讀取 WNPSH 與 Cluster 基本資訊
ncfile = nc.Dataset('/Dellwork4/cwuchiu/Cluster4_WNPSH_centroid_Ridge_SHAI_noET_34kt.nc')
WNPSH_possibility = np.array(ncfile.variables['hgt_adjust_cluster']) 
cluster_case_number = np.array(ncfile.variables['cluster_case_number']) 
which_cluster = np.array(ncfile.variables['which_cluster']) 
ncfile.close()

# 讀取 TC 軌跡與時間資料
ncfile = nc.Dataset('/Dellwork4/cwuchiu/TC_time_1979_2024_not_set_noET_34kt.nc')
tt = np.array(ncfile.variables['time'])
time_tc = [str(x) for x in tt]
ll = np.array(ncfile.variables['lon'])
la = np.array(ncfile.variables['lat'])
ncfile.close()

# ==================== 3. 統計各年份與月份的 Cluster 數量 ====================
cluster = np.zeros((5, 46))
cluster_month = np.zeros((5, 6))

for i in range(len(time_tc)):
    year_idx = int(time_tc[i][0:4]) - 1979
    month_idx = int(time_tc[i][4:6]) - 6
    cluster[int(which_cluster[i]), year_idx] += 1
    cluster_month[int(which_cluster[i]), month_idx] += 1

cluster_sum = np.sum(cluster, axis=0)
time = np.linspace(1979, 2024, 46)

# ==================== 4. 整理軌跡經緯度點位 ====================
# (1) 依年份與群別分類
pos_lon = [[[] for _ in range(5)] for _ in range(46)]
pos_lat = [[[] for _ in range(5)] for _ in range(46)]

check_time = 1979
for i in range(len(time_tc)):
    year_cur = int(time_tc[i][0:4])
    if year_cur == check_time:
        pos_lon[year_cur - 1979][int(which_cluster[i])].append(ll[i])
        pos_lat[year_cur - 1979][int(which_cluster[i])].append(la[i])
    else:
        check_time += 1
        pos_lon[year_cur - 1979][int(which_cluster[i])].append(ll[i])
        pos_lat[year_cur - 1979][int(which_cluster[i])].append(la[i])

# (2) 依各群總體分類 (用於繪製各群 46 年整體的軌跡密度)
pplo = [[], [], [], [], []]
ppla = [[], [], [], [], []]

for i in range(len(time_tc)):
    pplo[int(which_cluster[i])].append(ll[i])
    ppla[int(which_cluster[i])].append(la[i])

# (3) 全體颱風點位合併
pos_lon_all = []
pos_lat_all = []
for i in range(len(pplo)):
    pos_lon_all.extend(pplo[i])
    pos_lat_all.extend(ppla[i])
pos_lon_all = [pos_lon_all]
pos_lat_all = [pos_lat_all]

# ==================== 5. 軌跡密度計算函數與執行 ====================
def compute_track_density(save_lon, save_lat,
                          lon_min=100, lon_max=180,
                          lat_min=0, lat_max=50,
                          dlon=1.0, dlat=1.0):
    lon_bins = np.arange(lon_min, lon_max + dlon, dlon)
    lat_bins = np.arange(lat_min, lat_max + dlat, dlat)

    n_cluster = len(save_lon)
    density_all = np.zeros((n_cluster, len(lat_bins)-1, len(lon_bins)-1), dtype=np.float64)
    save_total_point = np.zeros(n_cluster)

    for c in range(n_cluster):
        density = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
        total_points = 0

        for ty_lons, ty_lats in zip(save_lon[c], save_lat[c]):
            ty_lons = np.array(ty_lons)
            ty_lats = np.array(ty_lats)

            valid = (~np.isnan(ty_lons)) & (~np.isnan(ty_lats))
            ty_lons = ty_lons[valid]
            ty_lats = ty_lats[valid]
            total_points += len(ty_lons)

            lon_idx = np.digitize(ty_lons, lon_bins) - 1
            lat_idx = np.digitize(ty_lats, lat_bins) - 1

            valid = (lon_idx >= 0) & (lon_idx < len(lon_bins)-1) & \
                    (lat_idx >= 0) & (lat_idx < len(lat_bins)-1)
            lon_idx = lon_idx[valid]
            lat_idx = lat_idx[valid]

            for j, k in zip(lat_idx, lon_idx):
                density[j, k] += 1

        if total_points > 0:
            density /= total_points
        save_total_point[c] = total_points
        density_all[c] = density

    return density_all, lon_bins, lat_bins

# 計算每年各群密度 (形狀: 46, 5, 20, 32)
density1 = np.zeros((46, 5, 50, 80))
density1_new = np.zeros((46, 5, 50, 80))

for i in range(46):
    density1[i, :, :, :], lon_bins, lat_bins = compute_track_density(pos_lon[i], pos_lat[i])

for i in range(cluster.shape[0]):
    for j in range(cluster.shape[1]):
        density1_new[j, i, :, :] = density1[j, i, :, :] * cluster[i, j]

# 計算全體合併密度與各群總體氣候密度 (形狀: 5, 20, 32)
density2, _, _ = compute_track_density(pos_lon_all, pos_lat_all)
density_cluster, _, _ = compute_track_density(pplo, ppla)

# 定義 2D 網格
lon_bins = np.linspace(100, 180, 81)
lat_bins = np.linspace(0, 50, 51)
lon_grid, lat_grid = np.meshgrid(lon_bins[:-1], lat_bins[:-1])  # (20, 32)
#%%
# ==================== 7. 合併 2x4 子圖繪製 (Cluster 1 ~ 4) ====================
# =================== 1. 畫布與網格設定 ====================
# 高度調至 10.5，給中間的 Colorbar 與刻度預留充裕空間
fig = plt.figure(figsize=(24, 10.5))

# hspace 提高至 0.60，確保中間有足夠縱向距離放置 Colorbar
gs = gridspec.GridSpec(
    2, 4, figure=fig,
    top=0.91, bottom=0.08, left=0.07, right=0.96,
    hspace=0.60, wspace=0.25
)

lon_contour = np.linspace(100, 180, 321)
lat_contour = np.linspace(50, 0, 201)
norm = TwoSlopeNorm(vmin=0, vcenter=0.0010, vmax=0.0020)
color_list = ['r', 'g', 'b', 'k', 'purple', 'orange']

ct = None

# 編號順序：上排 (a)-(d)，下排 (e)-(h)
title_top = ['(a)', '(b)', '(c)', '(d)']
title_bot = ['(e)', '(f)', '(g)', '(h)']

for col, i in enumerate(range(1, 5)):
    # ------------------------------------------------------------------
    # 上排 (Row 0)：Track Density & WNPSH 發生頻率 (Cartopy 地圖投影)
    # ------------------------------------------------------------------
    ax_top = fig.add_subplot(gs[0, col], projection=ccrs.PlateCarree())
    
    ax_top.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
    ax_top.coastlines(resolution='50m', linewidth=0.8)
    ax_top.add_feature(cfeature.BORDERS, linewidth=0.5)

    ax_top.set_xticks(np.arange(100, 181, 20), crs=ccrs.PlateCarree())
    ax_top.set_yticks(np.arange(0, 51, 10), crs=ccrs.PlateCarree())
    ax_top.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax_top.yaxis.set_major_formatter(LatitudeFormatter())
    
    # 經緯度刻度字體大小
    ax_top.tick_params(axis='both', labelsize=14)
    
    # 只有最左側子圖顯示緯度刻度標籤
    if col > 0:
        ax_top.set_yticklabels([])

    # 繪製 TC 軌跡密度 (Pcolormesh)
    ct = ax_top.pcolormesh(
        lon_grid, lat_grid, density_cluster[i],
        cmap='Reds',
        norm=norm,
        transform=ccrs.PlateCarree(),
        alpha=1
    )

    # 繪製 WNPSH 發生機率等值線 (Contour)
    contour_lines = ax_top.contour(
        lon_contour, lat_contour, WNPSH_possibility[i - 1],
        levels=np.linspace(0, 1, 11),
        linewidths=0.8,
        transform=ccrs.PlateCarree()
    )
    ax_top.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

    # 上排子圖標題 (a) ~ (d)
    ax_top.set_title(
        f'{title_top[i-1]} Cluster {i} WNPSH & TC Density\nCase number = {int(cluster_case_number[i-1])}',
        fontweight='bold', 
        fontsize=15,
        pad=10
    )

    # ------------------------------------------------------------------
    # 下排 (Row 1)：Cluster 比例時間變化趨勢 (標準平面座標)
    # ------------------------------------------------------------------
    ax_bot = fig.add_subplot(gs[1, col])
    
    ratio = cluster[i] / cluster_sum
    slope_val, intercept_val, r_val, p_val, std_err_val = linregress(time, ratio)
    
    # 繪製真實比例折線
    ax_bot.plot(time, ratio, color=color_list[i], linewidth=1.8, marker='o', markersize=3)
    
    # 趨勢線與 Label
    trend_label = f"Slope: {slope_val*10:.3f} decade$^{{-1}}$\np-value: {p_val:.2e}"
    ax_bot.plot(time, time * slope_val + intercept_val, color='crimson', linestyle='--', linewidth=1.5, zorder=999, label=trend_label)
    
    # Legend
    ax_bot.legend(loc='upper right', fontsize=11, framealpha=0.8)

    # 下排子圖標題 (e) ~ (h) 與軸標籤
    ax_bot.set_title(f"{title_bot[i-1]} Cluster {i}", fontweight='bold', fontsize=16, pad=10)
    ax_bot.set_xlabel("Year", fontsize=15)
    ax_bot.grid(True, linestyle=":", alpha=0.6)
    
    # 刻度字體大小
    ax_bot.tick_params(axis='both', labelsize=14)
    
    if col == 0:
        ax_bot.set_ylabel("Occurrence frequency", fontsize=15)

# ==================== 2. 上排地圖的共用 Colorbar (置於兩排之間) ====================
# [left, bottom, width, height]
# bottom 設在 0.515：上方避開地圖的經度文字，下方避開下排標題
cbar_ax = fig.add_axes([0.10, 0.515, 0.80, 0.020])
cbar = fig.colorbar(ct, cax=cbar_ax, orientation='horizontal', ticks=np.linspace(0, 0.002, 5))

# 設定 Colorbar 刻度與標籤
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Normalized Track Density', fontsize=15, fontweight='bold', labelpad=6)

# ==================== 3. 整體大標題 ====================
# 置於畫布頂部
# =============================================================================
# fig.text(
#     0.51, 0.97, 
#     'WNPSH Pattern & Tropical Cyclone Track Density with Occurrence Frequency', 
#     ha='center', va='center', 
#     fontsize=20, fontweight='bold'
# )
# =============================================================================

# ==================== 4. 儲存與顯示 ====================
plt.savefig('Figure5.png', dpi=400, bbox_inches='tight')
plt.show()