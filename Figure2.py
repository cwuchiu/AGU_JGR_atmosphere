#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:26:26 2026

@author: cwuchiu
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from multiprocessing import Pool
from scipy.stats import linregress
from tqdm import tqdm
import statsmodels.api as sm
from scipy.stats import linregress, t

# =============================================================================
# 1. 核心參數與環境變數鎖定 (鎖定 16 核心)
# =============================================================================
N_WORKERS = 32

# 環境變數設定：防止底層函式庫自動多執行緒造成 CPU 過載
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_WORKERS)

# Linux 核心層級物理隔離：僅允許 CPU 0 到 15
try:
    target_cpus = set(range(N_WORKERS))
    os.sched_setaffinity(0, target_cpus)
    print(f"[系統確認] 已由 Linux Kernel 物理鎖定在 CPU 0 ~ {N_WORKERS-1} 號核心！")
except AttributeError:
    pass

# =============================================================================
# 2. Worker 函數與初始化器 (全域層級)
# =============================================================================
def _init_worker():
    """確保每個子進程內部的 BLAS / OpenMP 函式庫嚴格使用單執行緒"""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass

def _process_monthly_slice(args):
    """
    單一月份的平行處理 Worker
    args: (idx, start_hr, end_hr, file_path, var_name, op_type)
    """
    idx, start_hr, end_hr, file_path, var_name, op_type = args
    
    # 限制執行緒數
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
        
    # 獨立在子進程開啟 NetCDF，防止多行程檔案衝突
    with nc.Dataset(file_path, 'r') as ds:
        data = ds.variables[var_name][start_hr:end_hr]
        if op_type == 'max':
            val = float(np.nanmax(data))
        elif op_type == 'mean':
            val = float(np.nanmean(data))
        else:
            raise ValueError(f"不支援的運算型態: {op_type}")
            
    return idx, val

def run_parallel_tasks(pool, file_path, var_name, time_slices, op_type, desc):
    """打包任務並透過 Pool 平行計算"""
    tasks = [(idx, s, e, file_path, var_name, op_type) for idx, s, e in time_slices]
    results = list(tqdm(pool.imap_unordered(_process_monthly_slice, tasks), 
                        total=len(tasks), desc=desc))
    # 依月份索引排序還原時序
    results.sort(key=lambda x: x[0])
    return np.array([r[1] for r in results])

# =============================================================================
# 3. 主執行流程
# =============================================================================
if __name__ == '__main__':
    # 時間與時長設定 (46 年 x 6 個月 = 276 個月份)
    month_hr = [720, 744, 744, 720, 744, 720]
    time = np.linspace(1979, 2024, 276)

    # 預先計算 276 個月份的切片起訖點
    time_slices = []
    count = 0
    for i in range(276):
        hr = month_hr[i % 6]
        time_slices.append((i, count, count + hr))
        count += hr

    print(f"\n[開始平行運算] 使用核心數: {N_WORKERS} (CPU 0~{N_WORKERS-1})")

    # 檔案路徑與變數名稱
    sv_file = '/Dellwork4/cwuchiu/shear_vorticity_hourly_new.nc'
    adj_file = '/Dellwork4/cwuchiu/Tvbar_hgt_WNPSH_2004_all_time.nc'
    raw_file = '/Dellwork4/cwuchiu/Tvbar_hourly.nc'

    # 開啟 Pool 平行處理三大資料集
    with Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        # (1) 計算 SV850 月平均
        sv850_m = run_parallel_tasks(
            pool, sv_file, 'SV850', time_slices, 
            op_type='mean', desc="[1/3] Parallel SV850 Mean"
        )
        
        # (2) 計算 Adjusted Hgt 空間最大值 (不扣除 mean)
        phi_max_adj = run_parallel_tasks(
            pool, adj_file, 'hgt500_adjust', time_slices, 
            op_type='max', desc="[2/3] Parallel Adjusted Hgt Max"
        )
        
        # (3) 計算 Original Hgt 空間最大值 (不扣除 mean)
        phi_max_raw = run_parallel_tasks(
            pool, raw_file, 'hgt500', time_slices, 
            op_type='max', desc="[3/3] Parallel Original Hgt Max"
        )
# ==================== 3.5 若原始資料為月資料，計算年平均 (Yearly Mean) ====================
# （如果你的變數已經是 1 年 1 個點的年資料，可跳過這段，直接使用你的年變數）
# 假設年份為 1979-2024，共 46 年，每 12 個月取平均：
years = np.arange(1979, 2025)

# 若資料為月資料且形狀為 (N_months,)，可 reshape 成 (年數, 6) 取平均：
sv850_yearly = sv850_m.reshape(-1, 6).mean(axis=1)
phi_adj_yearly = phi_max_adj.reshape(-1, 6).mean(axis=1)
phi_raw_yearly = phi_max_raw.reshape(-1, 6).mean(axis=1)

# 若你已經有算好的年平均變數，請確認變數名稱對應如下：
time_year = years  # 年份時間軸 (1979, 1980, ..., 2024)
sv_y = sv850_yearly  # 850 hPa shear vorticity 年平均
phi_adj_y = phi_adj_yearly  # Adjusted Phi_max 年平均
phi_raw_y = phi_raw_yearly  # Original Phi_max 年平均


# ==================== 4. 線性趨勢計算 (Linear Regression) ====================
slope_sv, intercept_sv, _, p_sv, _ = linregress(time_year, sv_y)
slope_adj, intercept_adj, _, p_adj, _ = linregress(time_year, phi_adj_y)
slope_raw, intercept_raw, _, p_raw, _ = linregress(time_year, phi_raw_y)




# ==================== 1. 自訂函數：考慮自相關的線性迴歸 (Santer et al. 2000) ====================
def linregress_eff(x, y):
  """計算線性趨勢，並根據殘差 Lag-1 自相關 (AR1) 校正有效自由度與 p-value。

  氣候學標準參考：Santer et al. (2000, JGR)
  """
  res = linregress(x, y)
  n = len(x)
  slope = res.slope
  intercept = res.intercept
  p_raw = res.pvalue

  # 1. 計算殘差 (Residuals)
  residuals = y - (slope * x + intercept)

  # 2. 計算殘差 Lag-1 自相關係數 (r1)
  e_mean = np.mean(residuals)
  r1_num = np.sum((residuals[:-1] - e_mean) * (residuals[1:] - e_mean))
  r1_den = np.sum((residuals - e_mean) ** 2)
  r1 = r1_num / r1_den if r1_den != 0 else 0.0

  # 3. 只有正自相關 (r1 > 0) 會虛增自由度，需校正；負自相關採保守原則不校正
  if r1 > 0:
    n_eff = n * (1.0 - r1) / (1.0 + r1)
    df_eff = max(n_eff - 2.0, 1.0)

    # 殘差自相關導致的標準誤膨脹係數
    se_eff = res.stderr * np.sqrt((1.0 + r1) / (1.0 - r1))
    t_stat = slope / se_eff
    # 依有效自由度重算雙尾 p-value
    p_eff = 2.0 * t.sf(np.abs(t_stat), df=df_eff)
  else:
    p_eff = p_raw
    r1 = max(r1, 0.0)

  return slope, intercept, p_eff, p_raw, r1


# ==================== 2. 年平均計算 (Yearly Mean) ====================
# 1979-2024 共 46 年，每年 6 個月 (6-11月) 取平均
time_year = np.arange(1979, 2025)

sv_y = sv850_m.reshape(-1, 6).mean(axis=1)
phi_adj_y = phi_max_adj.reshape(-1, 6).mean(axis=1)
phi_raw_y = phi_max_raw.reshape(-1, 6).mean(axis=1)

# ==================== 3. 趨勢與有效顯著性計算 ====================
# (a) WNPSHI
slope_sv, intercept_sv, p_sv_eff, p_sv_raw, r1_sv = linregress_eff(
    time_year, sv_y
)

# (b) Adjusted Phi_max
slope_adj, intercept_adj, p_adj_eff, p_adj_raw, r1_adj = linregress_eff(
    time_year, phi_adj_y
)

# (c) Original Phi_max (ERA5)
slope_raw, intercept_raw, p_raw_eff, p_raw_raw, r1_raw = linregress_eff(
    time_year, phi_raw_y
)

# 讀取 SHAI 資料
ncfile = nc.Dataset(
    "/Dellwork4/cwuchiu/SHAI_km2_HGT_Lon_Lat_1979_2024_monthly_hourly_adjust_mean_2004.nc"
)
SHAI_monthly = np.array(ncfile.variables["SHAI_monthly"])
ncfile.close()

ncfile = nc.Dataset(
    "/Dellwork4/cwuchiu/SHAI_km2_HGT_Lon_Lat_1979_2024_monthly_hourly_ERA5.nc"
)
SHAI_ERA5 = np.array(ncfile.variables["SHAI_monthly"])
ncfile.close()

SHAI_yr = SHAI_monthly.reshape(-1, 6).mean(axis=1)
SHAI_ERA5_yr = SHAI_ERA5.reshape(-1, 6).mean(axis=1)

# ----------------- (d) SHAI 分段趨勢計算 -----------------
# 1. 2004 年前遮罩 (1979–2004)
mask_pre = time_year <= 2004
time_pre = time_year[mask_pre]
shai_pre = SHAI_ERA5_yr[mask_pre]  # 2004 前 Adjusted 與 ERA5 完全相同

slope_pre, intercept_pre, p_pre_eff, p_pre_raw, r1_pre = linregress_eff(
    time_pre, shai_pre
)

# 2. 2004 年後遮罩 (2004–2024)
mask_post = time_year >= 2004
time_post = time_year[mask_post]
shai_post_adj = SHAI_yr[mask_post]
shai_post_era5 = SHAI_ERA5_yr[mask_post]

# 2004 後 Adjusted
(
    slope_post_adj,
    intercept_post_adj,
    p_post_adj_eff,
    p_post_adj_raw,
    r1_post_adj,
) = linregress_eff(time_post, shai_post_adj)

# 2004 後 ERA5
(
    slope_post_era5,
    intercept_post_era5,
    p_post_era5_eff,
    p_post_era5_raw,
    r1_post_era5,
) = linregress_eff(time_post, shai_post_era5)

# 在終端機印出完整統計表
print("=" * 85)
print(f"{'指標名稱':<25}{'斜率 (/dec)':<18}{'OLS p':<12}{'r1':<8}{'有效 p-eff'}")
print("-" * 85)
print(
    f"{'(a) WNPSHI':<25}{slope_sv * 10:<+18.2e}{p_sv_raw:<12.3e}{r1_sv:<8.2f}{p_sv_eff:<12.3e}"
)
print(
    f"{'(b) Phi_max (Adj)':<25}{slope_adj * 10:<+18.2f}{p_adj_raw:<12.3e}{r1_adj:<8.2f}{p_adj_eff:<12.3e}"
)
print(
    f"{'(c) Phi_max (ERA5)':<25}{slope_raw * 10:<+18.2f}{p_raw_raw:<12.3e}{r1_raw:<8.2f}{p_raw_eff:<12.3e}"
)
print(
    f"{'(d) SHAI (Pre-2004)':<25}{slope_pre * 10:<+18.2e}{p_pre_raw:<12.3e}{r1_pre:<8.2f}{p_pre_eff:<12.3e}"
)
print(
    f"{'(d) SHAI (Post-Adj)':<25}{slope_post_adj * 10:<+18.2e}{p_post_adj_raw:<12.3e}{r1_post_adj:<8.2f}{p_post_adj_eff:<12.3e}"
)
print(
    f"{'(d) SHAI (Post-ERA5)':<25}{slope_post_era5 * 10:<+18.2e}{p_post_era5_raw:<12.3e}{r1_post_era5:<8.2f}{p_post_era5_eff:<12.3e}"
)
print("=" * 85)


# ==================== 4. 繪圖 (4x1 Subplots) ====================
fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

# --- Subplot 1: (a) WNPSHI ---
axs[0].plot(
    time_year,
    sv_y,
    ".-",
    color="royalblue",
    linewidth=2,
    markersize=8,
    label=(
        f"Trend: {slope_sv * 10:+.2e} m s$^{{-1}}$ decade$^{{-1}}$ (p-value ="
        f" {p_sv_eff:.2f})"
    ),
)
axs[0].plot(
    time_year,
    time_year * slope_sv + intercept_sv,
    "--",
    color="crimson",
    linewidth=2,
)
axs[0].set_title(
    "(a) WNPSHI (850 hPa shear vorticity)", fontweight="bold", fontsize=16
)
axs[0].set_ylabel("Shear Vorticity (m s$^{-1}$)", fontsize=16)
axs[0].tick_params(axis="both", labelsize=14)
axs[0].legend(fontsize=12, loc="upper right")

# --- Subplot 2: (b) Adjusted Phi_max ---
axs[1].plot(
    time_year,
    phi_adj_y,
    ".-",
    color="darkgreen",
    linewidth=2,
    markersize=8,
    zorder=2,
)
axs[1].plot(
    time_year,
    time_year * slope_adj + intercept_adj,
    "--",
    color="crimson",
    linewidth=2,
    label=(
        f"Trend: {slope_adj * 10:+.2f} gpm decade$^{{-1}}$ (p-value ="
        f" {p_adj_eff:.3f})"
    ),
    zorder=3,
)
axs[1].set_title(
    r"(b) Central Intensity Index ($Z_{max}$) - Adjusted",
    fontweight="bold",
    fontsize=16,
)
axs[1].set_ylabel("Geopotential Height (gpm)", fontsize=16)
axs[1].tick_params(axis="both", labelsize=14)
axs[1].legend(fontsize=12, loc="upper right")

# --- Subplot 3: (c) Original Phi_max ---
axs[2].plot(
    time_year,
    phi_raw_y,
    ".-",
    color="darkgreen",
    linewidth=2,
    markersize=8,
    zorder=2,
)
axs[2].plot(
    time_year,
    time_year * slope_raw + intercept_raw,
    "--",
    color="crimson",
    linewidth=2,
    label=(
        f"Trend: {slope_raw * 10:+.2f} gpm decade$^{{-1}}$ (p-value ="
        f" {p_raw_eff:.2e})"
    ),
    zorder=3,
)
axs[2].set_title(
    r"(c) Central Intensity Index ($Z_{max}$) - ERA5",
    fontweight="bold",
    fontsize=16,
)
axs[2].set_ylabel("Geopotential Height (gpm)", fontsize=16)
axs[2].tick_params(axis="both", labelsize=14)
axs[2].legend(fontsize=12, loc="upper right")

# --- Subplot 4: (d) SHAI ---
# 1. 繪製兩條數據折線
axs[3].plot(
    time_year,
    SHAI_yr,
    ".-",
    color="royalblue",
    linewidth=2,
    markersize=8,
    zorder=2,
)
axs[3].plot(
    time_year,
    SHAI_ERA5_yr,
    ".-",
    color="crimson",
    linewidth=2,
    markersize=8,
    zorder=2,
)

# 2. 加入 2004 年分界垂直參考線
axs[3].axvline(2004, color="gray", linestyle=":", linewidth=1.5, alpha=0.8)

# 3. 繪製 2004 年前的共同趨勢線 (1979–2004)
axs[3].plot(
    time_pre,
    time_pre * slope_pre + intercept_pre,
    "--",
    color="dimgray",
    linewidth=2,
    zorder=3,
    label=(
        f"1979–2004: {slope_pre * 10:+.2e} km$^{{2}}$ decade$^{{-1}}$ (p-value ="
        f" {p_pre_eff:.2f})"
    ),
)

# 4. 繪製 2004 年後的兩條趨勢線 (2004–2024)
axs[3].plot(
    time_post,
    time_post * slope_post_adj + intercept_post_adj,
    "--",
    color="darkblue",
    linewidth=2.5,
    zorder=3,
    label=(
        f"2004–2024 (Adj): {slope_post_adj * 10:+.2e} km$^{{2}}$"
        f" decade$^{{-1}}$ (p-value = {p_post_adj_eff:.2f})"
    ),
)
axs[3].plot(
    time_post,
    time_post * slope_post_era5 + intercept_post_era5,
    "--",
    color="firebrick",
    linewidth=2.5,
    zorder=3,
    label=(
        f"2004–2024 (ERA5): {slope_post_era5 * 10:+.2e} km$^{{2}}$"
        f" decade$^{{-1}}$ (p-value = {p_post_era5_eff:.2e})"
    ),
)

axs[3].set_title(
    r"(d) SHAI",
    fontweight="bold",
    fontsize=16,
)
axs[3].set_xlim(1979, 2024)
axs[3].set_xticks(np.linspace(1979, 2024, 10))
axs[3].set_xlabel("Year", fontsize=16)
axs[3].set_ylabel("SHAI (km$^{2}$)", fontsize=16)
axs[3].tick_params(axis="both", labelsize=14)

# 圖例置於左上角，清晰展示三條趨勢線
axs[3].legend(fontsize=10.5, loc="upper left", framealpha=0.9)

plt.tight_layout()
plt.savefig("Figure2.png", dpi=400)
plt.show()