#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 18:40:54 2026

@author: cwuchiu
"""

import os
import sys
import time as time_lib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy import stats
from scipy.stats import linregress
import pymannkendall as mk
import netCDF4 as nc
from multiprocessing import Pool
from tqdm import tqdm

# =============================================================================
# 1. 核心參數與執行緒防護鎖定 (避免 BLAS / MKL 引起執行緒爆炸)
# =============================================================================
N_WORKERS = 32

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_WORKERS)

# Linux 核心層級物理隔離：僅允許 CPU 0 到 N_WORKERS-1
try:
    target_cpus = set(range(N_WORKERS))
    os.sched_setaffinity(0, target_cpus)
    print(f"[系統確認] 已由 Linux Kernel 物理鎖定在 CPU 0 ~ {N_WORKERS-1} 號核心！")
except (AttributeError, OSError):
    pass


# =============================================================================
# 2. Worker 函數與初始化器 (必須放在全域層級)
# =============================================================================
def _init_worker():
    """確保每個子進程內部的 BLAS / OpenMP 函式庫嚴格使用單執行緒"""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass


def _worker_calc_theil_mk_row(args):
    """
    以單一緯度列 (321 個網格) 為任務單元，計算 Theil-Sen 斜率與 Mann-Kendall 檢定
    """
    row_idx, row_ts_data, time_arr = args
    n_lon = row_ts_data.shape[1]
    
    trend_row = np.zeros(n_lon)
    p_val_row = np.zeros(n_lon)
    
    for j in range(n_lon):
        ts = row_ts_data[:, j]
        # 1. Theil-Sen 斜率
        res = stats.theilslopes(ts, time_arr, alpha=0.95)
        trend_row[j] = res[0]
        
        # 2. Mann-Kendall 檢定
        mk_test = mk.original_test(ts)
        p_val_row[j] = mk_test.p
        
    return row_idx, trend_row, p_val_row


# =============================================================================
# 3. 主運算流程
# =============================================================================
if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # PART 1: 讀取與計算 500 hPa 位勢高度趨勢 (Figure 6)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("讀取 500 hPa 位勢高度資料中...")
    file_hgt = '/Dellwork4/cwuchiu/SHAI_km2_HGT_Lon_Lat_1979_2024_monthly_hourly_adjust_mean_2004.nc'
    ncfile_hgt = nc.Dataset(file_hgt)
    hgt_adjust = np.array(ncfile_hgt.variables['HGT_monthly'])
    ncfile_hgt.close()

    year_num = 46
    hgt_adjust_yr = np.zeros((year_num, 201, 321))

    for i in range(year_num):
        if i >= 31:
            hgt_adjust_yr[i] = np.nanmean(hgt_adjust[6*i:6*(i+1)], axis=0) / 9.81 * 9.80
        else:
            hgt_adjust_yr[i] = np.nanmean(hgt_adjust[6*i:6*(i+1)], axis=0)

    time_hgt = np.linspace(2024 - year_num + 1, 2024, year_num)
    lon_hgt = np.linspace(100, 180, 321)
    lat_hgt = np.linspace(0, 50, 201)[::-1]

    # --- 平行化計算 Theil-Sen 趨勢與 Mann-Kendall 檢定 ---
    dhgt500dt_sen = np.zeros((201, 321))
    p_value_mk = np.zeros((201, 321))

    print(f"啟動多核心平行計算 Theil-Sen & Mann-Kendall (共 201 列緯度，調用 {N_WORKERS} 核心)...")
    start_time = time_lib.time()

    # 打包任務：每一任務為一條完整緯度列 (46, 321)
    tasks = [(i, hgt_adjust_yr[:, i, :], time_hgt) for i in range(201)]

    with Pool(processes=min(N_WORKERS, len(tasks)), initializer=_init_worker) as pool:
        for row_idx, trend_row, p_val_row in tqdm(
            pool.imap_unordered(_worker_calc_theil_mk_row, tasks),
            total=len(tasks),
            desc="Theil-Sen & MK Progress"
        ):
            dhgt500dt_sen[row_idx, :] = trend_row
            p_value_mk[row_idx, :] = p_val_row

    print(f"500 hPa 趨勢計算完成！耗時: {time_lib.time() - start_time:.2f} 秒\n")

    # -------------------------------------------------------------------------
    # PART 2: 讀取與計算導引氣流 (Steering Flow) 變化 (Figure 7)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("讀取導引氣流 (Steering Flow) 資料中...")
    file_steer = '/Dellwork4/cwuchiu/Monthly_steering_1979_2025.nc'
    ncfile_steer = nc.Dataset(file_steer)
    U = np.array(ncfile_steer.variables['U_steer'])
    V = np.array(ncfile_steer.variables['V_steer'])
    lat_steer = np.array(ncfile_steer.variables['latitude'])
    lon_steer = np.array(ncfile_steer.variables['longitude'])
    ncfile_steer.close()

    U_yr = np.zeros((46, 201, 321))
    V_yr = np.zeros((46, 201, 321))
    for i in range(46):
        U_yr[i] = np.nanmean(U[6*i:6*(i+1)], axis=0)
        V_yr[i] = np.nanmean(V[6*i:6*(i+1)], axis=0)

    time_steer = np.linspace(1979, 2024, 46)
    time_s_mean = np.mean(time_steer)

    # 計算 dU/dt 線性趨勢斜率
    U_mean = np.mean(U_yr, axis=0)
    numerator_direct = np.sum((time_steer[:, None, None] - time_s_mean) * (U_yr - U_mean[None, :, :]), axis=0)
    denominator_direct = np.sum((time_steer[:, None, None] - time_s_mean)**2, axis=0)
    dUdt = numerator_direct / denominator_direct

    # 計算 dV/dt 線性趨勢斜率
    V_mean = np.mean(V_yr, axis=0)
    numerator_direct2 = np.sum((time_steer[:, None, None] - time_s_mean) * (V_yr - V_mean[None, :, :]), axis=0)
    denominator_direct2 = np.sum((time_steer[:, None, None] - time_s_mean)**2, axis=0)
    dVdt = numerator_direct2 / denominator_direct2

    # 保留原始 p-value 計算過程
    p_value = np.zeros((201, 321))
    p_value2 = np.zeros((201, 321))
    for i in range(201):
        for j in range(321):
            _, _, _, p_value[i, j], _ = linregress(time_steer, U_yr[:, i, j])
            _, _, _, p_value2[i, j], _ = linregress(time_steer, V_yr[:, i, j])

    print("導引氣流計算完成！\n")
## =========================================================================
    # 4. 繪製 1x2 子圖 (右圖：風速 Shading + 風向固定箭頭；雙垂直 Colorbar)
    # =========================================================================
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    print("=" * 60)
    print("開始繪製 1x2 組合圖...")

    # 計算 46 年累積導引氣流變化 (U, V) 與合風速變化 (Speed)
    U_change = dUdt * 10
    V_change = dVdt * 10
    wspd_change = np.hypot(U_change, V_change)  # 等同於 sqrt(U^2 + V^2)

    # 抽取 Quiver 格點 (每 10 點取 1 點)
    lon_q = lon_steer[::10]
    lat_q = lat_steer[::10]
    u_q = U_change[::10, ::10]
    v_q = V_change[::10, ::10]
    spd_q = wspd_change[::10, ::10]

    # --- 關鍵：將風向量正規化為單位向量 (箭頭長度固定，僅代表風向) ---
    with np.errstate(divide='ignore', invalid='ignore'):
        u_norm = np.where(spd_q > 0.05, u_q / spd_q, 0.0)
        v_norm = np.where(spd_q > 0.05, v_q / spd_q, 0.0)

    # 建立 1x2 子圖
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        figsize=(22, 7.5), 
        subplot_kw={'projection': ccrs.PlateCarree()},
        gridspec_kw={'wspace': 0.28}
    )

    x_extent = np.linspace(-180, 180, 37)
    y_extent = np.linspace(-90, 90, 19)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    # -------------------------------------------------------------------------
    # 左子圖 (ax1): 500 hPa 位勢高度趨勢
    # -------------------------------------------------------------------------
    ax1.set_xticks(x_extent, crs=ccrs.PlateCarree())
    ax1.set_yticks(y_extent, crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    ax1.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())

    levels_hgt = np.arange(-0.2, 1.01, 0.1)
    norm_hgt = mcolors.TwoSlopeNorm(vmin=levels_hgt.min(), vcenter=0.0, vmax=levels_hgt.max())

    # 填色層
    cf1 = ax1.contourf(
        lon_hgt, lat_hgt, dhgt500dt_sen,
        levels=levels_hgt,
        cmap='RdBu_r', 
        norm=norm_hgt,
        alpha=0.9,
        extend='both',
        transform=ccrs.PlateCarree()
    )

    # 等值線
    cs1 = ax1.contour(
        lon_hgt, lat_hgt, dhgt500dt_sen, 
        levels=levels_hgt, 
        colors='k', 
        linewidths=0.8,
        transform=ccrs.PlateCarree()
    )
    ax1.clabel(cs1, inline=True, fontsize=10, fmt="%.2f")

    # 顯著性檢定網底
    plt.rcParams['hatch.color'] = 'gray'
    ax1.contourf(
        lon_hgt, lat_hgt, p_value_mk,
        levels=[0, 0.05],
        colors='none', 
        hatches=['////'], 
        transform=ccrs.PlateCarree()
    )

    ax1.tick_params(axis='both', labelsize=14)
    ax1.coastlines(resolution='50m', linewidth=1, alpha=0.8)
    ax1.set_title(
        f'(a) 500 hPa Geopotential Height Trend ({int(time_hgt[0])}–{int(time_hgt[-1])})', 
        fontsize=16, fontweight='bold', y=1.03
    )

    # 左圖垂直 Colorbar
    cax1 = inset_axes(
        ax1,
        width="3.2%",
        height="100%",
        loc='lower left',
        bbox_to_anchor=(1.03, 0.0, 1, 1),
        bbox_transform=ax1.transAxes,
        borderpad=0
    )
    cbar1 = fig.colorbar(cf1, cax=cax1, orientation='vertical')
    cbar1.set_label('500 hPa geopotential height trend (gpm $yr^{-1}$)', fontsize=14, labelpad=10)
    cbar1.ax.tick_params(labelsize=14)

    # -------------------------------------------------------------------------
    # 右子圖 (ax2): 導引氣流 (風速 Shading + 風向 Unit Quiver)
    # -------------------------------------------------------------------------
    ax2.set_xticks(x_extent, crs=ccrs.PlateCarree())
    ax2.set_yticks(y_extent, crs=ccrs.PlateCarree())
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)
    ax2.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())

    # 1. 底層 Shading：風速變化量 (Wind Speed Change)
    # 可依資料分布調整 levels 上下限與步長，此處預設 0 ~ 3 m/s
    levels_spd = np.arange(0, 0.71, 0.1)
    cf2 = ax2.contourf(
        lon_steer, lat_steer, wspd_change,
        levels=levels_spd,
        cmap='YlOrRd',     # 風速常用色階盤 (若喜歡純藍色系也可改為 'Blues')
        extend='max',
        alpha=0.60,
        transform=ccrs.PlateCarree()
    )

    # 2. 上層 Quiver：單位長度向量 (僅表示風向)
    # scale=0.6 使箭頭長度約占相鄰格點間距的一半，pivot='mid' 讓箭頭中心置中
    q = ax2.quiver(
        lon_q, lat_q,
        u_norm, v_norm,
        scale=0.6,        
        scale_units='xy',
        angles='xy', 
        pivot='mid',
        color='black',      # 黑色箭頭在淺彩色底圖上辨識度最高
        alpha=0.75,
        width=0.003,
        headwidth=3.5,
        headlength=4.5,
        transform=ccrs.PlateCarree()
    )

    ax2.tick_params(axis='both', labelsize=14)
    ax2.coastlines(resolution='50m', linewidth=1, alpha=0.8)
    ax2.set_title(
        '(b) 1979-2024 (300-850 hPa) Steering Flow change', 
        fontsize=16, fontweight='bold', y=1.03
    )

    # 右圖垂直 Colorbar (結構與左圖 100% 對稱，確保兩圖尺寸完全相同)
    cax2 = inset_axes(
        ax2,
        width="3.2%",
        height="100%",
        loc='lower left',
        bbox_to_anchor=(1.03, 0.0, 1, 1),
        bbox_transform=ax2.transAxes,
        borderpad=0
    )
    cbar2 = fig.colorbar(cf2, cax=cax2, orientation='vertical')
    cbar2.set_label('Steering flow speed change (m s$^{-1}$ decade$^{{-1}}$)', fontsize=14, labelpad=10)
    cbar2.ax.tick_params(labelsize=14)

    # -------------------------------------------------------------------------
    # 儲存與顯示
    # -------------------------------------------------------------------------
    output_filename = 'Figure3.png'
    plt.savefig(output_filename, dpi=400, bbox_inches='tight')
    print(f"圖片已成功產出並儲存至：{output_filename}")
    plt.show()