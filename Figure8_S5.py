#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 14:12:02 2026

@author: cwuchiu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Tropical Cyclone Trajectory Model with 16-Core Parallel Computing,
Beta Drift Effect (Wu and Wang 2004), Factor Separation, and Combined Visualization.
"""

# =============================================================================
# 0. 環境變數設定與 16 核心安全鎖定 (必須在載入 numpy/scipy 前設定)
# =============================================================================
import os

N_WORKERS = 16

# 限制底層數學庫僅使用單執行緒，避免多進程 * 多執行緒造成 CPU 超載
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_WORKERS)

# Linux Kernel 物理親和力綁定：鎖定 CPU 0 到 15 號核心
try:
    target_cpus = set(range(N_WORKERS))
    os.sched_setaffinity(0, target_cpus)
    print(f"[系統安全鎖定] 已物理鎖定在 CPU 0 ~ {N_WORKERS-1} 號核心！")
except AttributeError:
    pass

from multiprocessing import Pool
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from global_land_mask import globe
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import netCDF4 as nc
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from tqdm import tqdm, trange
import xarray as xr

# =============================================================================
# 1. 全域輔助函式與 Worker 函式 (供 multiprocessing 調用)
# =============================================================================
def _init_worker():
    """確保每個子進程內部的 BLAS / OpenMP 函式庫嚴格使用單執行緒"""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass


def _advect_chunk_worker(args):
    """
    單一 Worker 負責計算一小批 (Chunk) 颱風的積分軌跡 (含 Beta Drift)
    """
    (gen_chunk, u_field, v_field, lat_arr, lon_arr, 
     duration_days, dt_hours, max_land_hours, beta_speed, beta_dir) = args
    
    interp_u = RegularGridInterpolator((lat_arr, lon_arr), u_field, bounds_error=False, fill_value=0)
    interp_v = RegularGridInterpolator((lat_arr, lon_arr), v_field, bounds_error=False, fill_value=0)
    
    # 計算 Beta drift 分量 (Wu and Wang 2004: 303° 方向，3.0 m/s)
    # 303° (正北為 0° 順時針): 西風分量約 -2.52 m/s, 北風分量約 +1.63 m/s
    u_beta = beta_speed * np.sin(np.deg2rad(beta_dir))
    v_beta = beta_speed * np.cos(np.deg2rad(beta_dir))
    
    steps = int(duration_days * 24 / dt_hours)
    dt_sec = dt_hours * 3600
    R_earth = 6371000.0  # 地球半徑 (m)
    
    chunk_lon = []
    chunk_lat = []
    
    for start_lat, start_lon in gen_chunk:
        curr_lat, curr_lon = float(start_lat), float(start_lon)
        path_lat = [curr_lat]
        path_lon = [curr_lon]
        hours_on_land = 0
        
        for _ in range(steps):
            # 1. 取得環境駛流速度並疊加 Beta Drift
            u_steer = float(interp_u((curr_lat, curr_lon)))
            v_steer = float(interp_v((curr_lat, curr_lon)))
            
            u_total = u_steer + u_beta
            v_total = v_steer + v_beta
            
            # 2. 登陸消散機制
            if globe.is_land(curr_lat, curr_lon):
                hours_on_land += dt_hours
            else:
                hours_on_land = 0
                
            if hours_on_land >= max_land_hours:
                break
                
            # 3. 尤拉積分更新經緯度位置
            dlat = (v_total * dt_sec) / R_earth * (180.0 / np.pi)
            dlon = (u_total * dt_sec) / (R_earth * np.cos(np.deg2rad(curr_lat))) * (180.0 / np.pi)
            
            curr_lat += dlat
            curr_lon += dlon
            
            # 邊界判定 (超出預設範圍終止追蹤)
            if curr_lat < 0 or curr_lat > 50 or curr_lon < 100 or curr_lon > 180:
                break
                
            path_lat.append(curr_lat)
            path_lon.append(curr_lon)
            
        chunk_lon.extend(path_lon)
        chunk_lat.extend(path_lat)
        
    return chunk_lon, chunk_lat


def run_advection_parallel(genesis_list, u_field, v_field, lat_arr, lon_arr, 
                           duration_days=10, dt_hours=1, max_land_hours=48,
                           beta_speed=3.0, beta_dir=303.0, n_workers=N_WORKERS):
    """透過 16-Core Pool 平行運算軌跡模型"""
    if len(genesis_list) == 0:
        return np.array([]), np.array([])
        
    n_chunks = min(n_workers, len(genesis_list))
    chunks = np.array_split(genesis_list, n_chunks)
    
    tasks = [
        (chunk, u_field, v_field, lat_arr, lon_arr, 
         duration_days, dt_hours, max_land_hours, beta_speed, beta_dir)
        for chunk in chunks if len(chunk) > 0
    ]
    
    with Pool(processes=n_chunks, initializer=_init_worker) as pool:
        results = list(tqdm(pool.imap(_advect_chunk_worker, tasks), 
                            total=len(tasks), 
                            desc=f"Parallel Advection ({len(genesis_list)} TCs / {n_chunks} Cores)"))
    
    all_lons = []
    all_lats = []
    for sub_lon, sub_lat in results:
        all_lons.extend(sub_lon)
        all_lats.extend(sub_lat)
        
    return np.array(all_lons), np.array(all_lats)


def get_genesis_points(save_lon, save_lat, start_year_idx, end_year_idx):
    """提取指定年份區間的颱風生成位置"""
    gen_points = []
    for y in range(start_year_idx, end_year_idx):
        for t in range(len(save_lon[y])):
            track_lon = np.array(save_lon[y][t])
            track_lat = np.array(save_lat[y][t])
            if len(track_lon) > 0 and not np.isnan(track_lon[0]):
                gen_points.append((track_lat[0], track_lon[0]))
    return gen_points


def plot_streamlines(ax, u, v, gen_points, title, lon_grid, lat_grid):
    """繪製平均駛流向量與生成點"""
    x_extent = np.linspace(-180, 180, 37)
    y_extent = np.linspace(-90, 90, 19)

    ax.set_xticks(x_extent, crs=ccrs.PlateCarree())
    ax.set_yticks(y_extent, crs=ccrs.PlateCarree())

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='gray')
    ax.tick_params(axis='both', labelsize=14)

    q = ax.quiver(
        lon_grid[::10], lat_grid[::10],
        u[::10, ::10], v[::10, ::10],
        scale=1.5,
        scale_units='xy',
        angles='xy',
        color='royalblue',
        alpha=1.0
    )

    ax.quiverkey(
        q, X=0.9, Y=1.03, U=10,
        label='10 m/s',
        coordinates='axes',
        labelpos='E',
        fontproperties={'size': 14}
    )

    gen_lats = [p[0] for p in gen_points]
    gen_lons = [p[1] for p in gen_points]
    ax.scatter(gen_lons, gen_lats, s=12, c='red', alpha=0.4,
               transform=ccrs.PlateCarree(), label='Genesis')

    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    return q


def get_density_grid(lons, lats, bins=(80, 50), range_lon=(100, 180), range_lat=(0, 50), sigma=2):
    """計算軌跡網格密度場並進行高斯平滑"""
    H, _, _ = np.histogram2d(lons, lats, bins=bins, range=[range_lon, range_lat], density=True)
    H = H.T
    return gaussian_filter(H, sigma=sigma)


# =============================================================================
# 2. 主程式入口 (Main Workflow)
# =============================================================================
if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # A. 讀取 IBTrACS 最佳路徑資料 (1979 - 2024)
    # -------------------------------------------------------------------------
    print("\n[步驟 1/7] 讀取 IBTrACS 最佳路徑資料...")
    ibtracs_path = '/Dellwork5/cwuchiu/IBTrACS.WP.v04r01.nc'
    ncfile = nc.Dataset(ibtracs_path)
    ar = xr.open_dataset(ibtracs_path)

    lon_raw = np.array(ncfile.variables['usa_lon'])
    lat_raw = np.array(ncfile.variables['usa_lat'])
    year_raw = np.array(ncfile.variables['season'])
    number_raw = np.array(ncfile.variables['number'])
    usa_wind_raw = np.array(ncfile.variables['usa_wind'])

    nature = ar['nature'].data
    time_raw = ar['iso_time'].data
    basin = ar['basin'].data
    basin1 = basin[:, 0]
    track_type = ar['track_type'].data
    storm_dir = ar['storm_dir'].data
    ncfile.close()

    save_lon = []
    save_lat = []
    save_time = []
    save_wind = []

    # 標記 6 - 11 月 (JJASON)
    JJASON = np.zeros(time_raw.shape[0], dtype=int)
    for i in range(JJASON.shape[0]):
        if all(6 <= int(t[5:7]) <= 11 for t in time_raw[i, :] if t):
            JJASON[i] = 1

    os.makedirs('/Dellwork5/cwuchiu/SH/best_track_original', exist_ok=True)

    for j in range(1979, 2025):
        ab, ba, ccc, www = [], [], [], []
        max_wind = np.nanmax(usa_wind_raw, axis=1)
        
        if j == 2024:
            tcmask = np.where((year_raw == j) & (basin1 == b'WP') & 
                              (track_type == b'PROVISIONAL') & (JJASON == 1) & (max_wind >= 34))
        else:
            tcmask = np.where((year_raw == j) & (basin1 == b'WP') & 
                              (track_type == b'main') & (JJASON == 1) & (max_wind >= 34))

        lon_tc = lon_raw[tcmask]
        lat_tc = lat_raw[tcmask]
        time_tc = time_raw[tcmask]
        usa_wind_tc = usa_wind_raw[tcmask]
        nature_tc = nature[tcmask]

        for i in range(len(lon_tc)):
            mask_pt = ((lat_tc[i, :] != -9999) & (lon_tc[i, :] >= 100) & 
                       (lat_tc[i, :] >= 0) & (nature_tc[i, :] != b'ET'))
            ab.append(lon_tc[i, mask_pt])
            ba.append(lat_tc[i, mask_pt])
            ccc.append(time_tc[i, mask_pt])
            www.append(usa_wind_tc[i, mask_pt])

        save_lon.append(ab)
        save_lat.append(ba)
        save_time.append(ccc)
        save_wind.append(www)

    # 清除空軌跡
    for i in range(len(save_lon)):
        save_lon[i] = [track for track in save_lon[i] if len(track) > 0]
        save_lat[i] = [track for track in save_lat[i] if len(track) > 0]
        save_time[i] = [track for track in save_time[i] if len(track) > 0]
        save_wind[i] = [track for track in save_wind[i] if len(track) > 0]

    # 分離 Early (1979-2001, 23年) 與 Late (2002-2024, 23年) 的生成點
    split_idx = 23
    total_years = 46
    gen_early = get_genesis_points(save_lon, save_lat, 0, split_idx)
    gen_late  = get_genesis_points(save_lon, save_lat, split_idx, total_years)
    print(f"-> Early 期生成颱風數: {len(gen_early)}")
    print(f"-> Late  期生成颱風數: {len(gen_late)}")

    # -------------------------------------------------------------------------
    # B. 分批讀取大型 NetCDF 駛流資料並計算氣候平均
    # -------------------------------------------------------------------------
    print("\n[步驟 2/7] 分批處理大型駛流場資料...")
    nc_time = nc.Dataset('/Dellwork4/cwuchiu/TC_time_1979_2024_noET_34kt.nc')
    TCtime = np.array(nc_time.variables['time'])
    nc_time.close()

    years = np.array([int(str(t)[:4]) for t in TCtime])
    mask_early = (years >= 1979) & (years <= 2001)
    mask_late  = (years >= 2002) & (years <= 2024)

    n_early = np.sum(mask_early)
    n_late  = np.sum(mask_late)

    nc_steer = nc.Dataset('/Dellwork4/cwuchiu/TC_time_steer_noET_34kt.nc', 'r')
    U_var = nc_steer.variables['U_steer']
    V_var = nc_steer.variables['V_steer']
    lat_grid = nc_steer.variables['lat'][:]
    lon_grid = nc_steer.variables['lon'][:]

    _, nlat, nlon = U_var.shape
    sum_u_early = np.zeros((nlat, nlon), dtype=np.float64)
    sum_v_early = np.zeros((nlat, nlon), dtype=np.float64)
    sum_u_late  = np.zeros((nlat, nlon), dtype=np.float64)
    sum_v_late  = np.zeros((nlat, nlon), dtype=np.float64)

    batch_size = 2000
    total_steps = len(TCtime)

    for start_idx in trange(0, total_steps, batch_size, desc="Batch Averaging Steering"):
        end_idx = min(start_idx + batch_size, total_steps)
        b_mask_early = mask_early[start_idx:end_idx]
        b_mask_late  = mask_late[start_idx:end_idx]

        if not (np.any(b_mask_early) or np.any(b_mask_late)):
            continue

        u_chunk = U_var[start_idx:end_idx]
        v_chunk = V_var[start_idx:end_idx]

        if np.any(b_mask_early):
            sum_u_early += np.sum(u_chunk[b_mask_early], axis=0)
            sum_v_early += np.sum(v_chunk[b_mask_early], axis=0)

        if np.any(b_mask_late):
            sum_u_late += np.sum(u_chunk[b_mask_late], axis=0)
            sum_v_late += np.sum(v_chunk[b_mask_late], axis=0)

    nc_steer.close()

    mean_u_early = np.nan_to_num((sum_u_early / n_early).astype(np.float32), nan=0.0)
    mean_v_early = np.nan_to_num((sum_v_early / n_early).astype(np.float32), nan=0.0)
    mean_u_late  = np.nan_to_num((sum_u_late / n_late).astype(np.float32), nan=0.0)
    mean_v_late  = np.nan_to_num((sum_v_late / n_late).astype(np.float32), nan=0.0)

    # -------------------------------------------------------------------------
    # C. 繪製平均駛流場與生成位置 (Fig4-12)
    # -------------------------------------------------------------------------
    print("\n[步驟 3/7] 輸出平均駛流圖 (Fig4-12.png)...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    plot_streamlines(
        axes[0], mean_u_early, mean_v_early, gen_early,
        "(a) Early Period: Mean Steering flow & Genesis locations",
        lon_grid, lat_grid
    )
    plot_streamlines(
        axes[1], mean_u_late, mean_v_late, gen_late,
        "(b) Late Period: Mean Steering flow & Genesis locations",
        lon_grid, lat_grid
    )

    plt.tight_layout()
    plt.savefig('FigureS5.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # D. 16 核平行運算交叉實驗 (疊加 Wu and Wang 2004 Beta Effect: 303°, 3.0 m/s)
    # -------------------------------------------------------------------------
    print(f"\n[步驟 4/7] 啟動 16 核心平行運算軌跡模型 (疊加 Beta Drift: 303°, 3.0 m/s)...")

    # Exp 1: Late Genesis + Early Flow
    print(">> Running Exp 1 (Late Genesis + Early Steering Flow)...")
    sim_lon_exp1, sim_lat_exp1 = run_advection_parallel(
        gen_late, mean_u_early, mean_v_early, lat_grid, lon_grid,
        beta_speed=3.0, beta_dir=303.0
    )

    # Exp 2: Early Genesis + Late Flow
    print(">> Running Exp 2 (Early Genesis + Late Steering Flow)...")
    sim_lon_exp2, sim_lat_exp2 = run_advection_parallel(
        gen_early, mean_u_late, mean_v_late, lat_grid, lon_grid,
        beta_speed=3.0, beta_dir=303.0
    )

    # Target: Late Genesis + Late Flow
    print(">> Running Target (Late Genesis + Late Steering Flow)...")
    sim_lon_target, sim_lat_target = run_advection_parallel(
        gen_late, mean_u_late, mean_v_late, lat_grid, lon_grid,
        beta_speed=3.0, beta_dir=303.0
    )

    # Early Base: Early Genesis + Early Flow
    print(">> Running Early Base (Early Genesis + Early Steering Flow)...")
    sim_lon_early, sim_lat_early = run_advection_parallel(
        gen_early, mean_u_early, mean_v_early, lat_grid, lon_grid,
        beta_speed=3.0, beta_dir=303.0
    )

    # -------------------------------------------------------------------------
    # E. 計算空間密度分佈與圖形相關係數 (Pattern Correlation Coefficient, PCC)
    # -------------------------------------------------------------------------
    print("\n[步驟 5/7] 計算軌跡密度與相關係數 (PCC)...")
    dens_exp1   = get_density_grid(sim_lon_exp1, sim_lat_exp1)
    dens_exp2   = get_density_grid(sim_lon_exp2, sim_lat_exp2)
    dens_target = get_density_grid(sim_lon_target, sim_lat_target)
    dens_early  = get_density_grid(sim_lon_early, sim_lat_early)

    pcc_exp1, _ = pearsonr(dens_exp1.flatten(), dens_target.flatten())
    pcc_exp2, _ = pearsonr(dens_exp2.flatten(), dens_target.flatten())
    pcc_early, _ = pearsonr(dens_early.flatten(), dens_target.flatten())

    print("\n" + "=" * 45)
    print(f"Exp 1 (Genesis Shift) PCC with Target : {pcc_exp1:.3f}")
    print(f"Exp 2 (Steering Chg)  PCC with Target : {pcc_exp2:.3f}")
    print(f"Early Base Line       PCC with Target : {pcc_early:.3f}")
    if pcc_exp1 > pcc_exp2:
        print(">> 評估結論: Exp 1 PCC 較高 -> 生成位置變化為路徑改變主因！")
    else:
        print(">> 評估結論: Exp 2 PCC 較高 -> 環境駛流變化為路徑改變主因！")
    print("=" * 45)

    # -------------------------------------------------------------------------
    # F. 因素分離分解計算 (Factor Separation / Sensitivity Analysis)
    # -------------------------------------------------------------------------
    print("\n[步驟 6/7] 計算因素分離分解與貢獻百分比...")
    D_LL = dens_target
    D_LE = dens_exp1
    D_EL = dens_exp2
    D_EE = dens_early

    # 總變化量與各分量
    delta_total = D_LL - D_EE
    delta_G     = D_LE - D_EE  # Genesis shift effect
    delta_S     = D_EL - D_EE  # Steering change effect

    # 計算空間絕對貢獻百分比
    total_var = np.sum(np.abs(delta_G)) + np.sum(np.abs(delta_S))
    if total_var > 0:
        contrib_G = (np.sum(np.abs(delta_G)) / total_var) * 100.0
        contrib_S = (np.sum(np.abs(delta_S)) / total_var) * 100.0
    else:
        contrib_G, contrib_S = 0.0, 0.0

    print(f"-> 生成位置貢獻 (Genesis Shift)      : {contrib_G:.1f}%")
    print(f"-> 駛流場變化貢獻 (Steering Change)   : {contrib_S:.1f}%")

    # -------------------------------------------------------------------------
    # G. 繪製合併 12 欄 GridSpec 子圖 (上排 4 張 (a-d) + 下排 3 張 (e-g))
    # -------------------------------------------------------------------------
    print("\n[步驟 7/7] 繪製合併綜合成果圖 (Combined_Fig4-13_Fig4-14.png)...")

    
    # 建議將高度從 16 稍微降至 13~14，整體視覺比例會更扎實緊湊
    fig = plt.figure(figsize=(28, 14.5))

    # 拆分為兩個獨立的 GridSpec：
    # 上半部：top=0.95, bottom=0.53，hspace 設小 (0.05) 讓 Colorbar 緊貼上排圖
    gs_top = gridspec.GridSpec(
        nrows=2, ncols=12,
        height_ratios=[1, 0.06],
        top=0.95, bottom=0.53, left=0.05, right=0.96,
        hspace=0.20, wspace=0.4
    )

    # 下半部：top=0.46, bottom=0.06，與上半部留出適當間距 (0.53 - 0.46 = 0.07)
    gs_bot = gridspec.GridSpec(
        nrows=2, ncols=12,
        height_ratios=[1, 0.05],
        top=0.46, bottom=0.06, left=0.05, right=0.96,
        hspace=0.20, wspace=0.4
    )

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    xticks = np.arange(100, 181, 20)
    yticks = np.arange(0, 51, 10)

    # ---------------------------------------------------------------------
    # 上排 4 張圖：模擬 Track Density (a) ~ (d)
    # ---------------------------------------------------------------------
    x_top = np.linspace(100, 180, 80)
    y_top = np.linspace(0, 50, 50)
    X_top, Y_top = np.meshgrid(x_top, y_top)
    levels_top = np.linspace(0, 0.002, 11)

    top_data = [dens_target, dens_exp1, dens_exp2, dens_early]
    top_titles = [
        '(a) Late Genesis + Late Steering Flow',
        '(b) Late Genesis + Early Steering Flow',
        '(c) Early Genesis + Late Steering Flow',
        '(d) Early Genesis + Early Steering Flow'
    ]
    top_pccs = [None, pcc_exp1, pcc_exp2, pcc_early]
    top_col_slices = [(0, 3), (3, 6), (6, 9), (9, 12)]

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)

    cf_top = None
    for idx in range(4):
        c_start, c_end = top_col_slices[idx]
        # 使用 gs_top[0, ...]
        ax = fig.add_subplot(gs_top[0, c_start:c_end], projection=ccrs.PlateCarree())
        
        # 強制讓上排地圖靠底端對齊，消除底部留白
        ax.set_anchor('S')
        
        ax.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.8, color='gray')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        if idx > 0:
            ax.set_yticklabels([])

        cf_top = ax.contourf(
            X_top, Y_top, top_data[idx],
            levels=levels_top, cmap='hot_r',
            transform=ccrs.PlateCarree()
        )
        
        if top_pccs[idx] is not None:
            ax.text(
                0.95, 0.95, f'PCC = {top_pccs[idx]:.3f}',
                transform=ax.transAxes, fontsize=16, fontweight='bold',
                ha='right', va='top', bbox=bbox_props
            )

        ax.set_title(top_titles[idx], fontsize=16, fontweight='bold', pad=10)

    # 上排共用 Colorbar (使用 gs_top[1, 1:11])
    cbar_ax_top = fig.add_subplot(gs_top[1, 1:11])
    cbar_top = fig.colorbar(cf_top, cax=cbar_ax_top, orientation='horizontal', ticks=levels_top)
    cbar_top.set_label('Track Density', fontsize=16, fontweight='bold')
    cbar_top.ax.tick_params(labelsize=16)

    # ---------------------------------------------------------------------
    # 下排 3 張圖：貢獻度分析 / 變化量 (e) ~ (g)
    # ---------------------------------------------------------------------
    bot_data = [delta_total, delta_G, delta_S]
    bot_titles = [
        '(e) Total Change (Late - Early)',
        f'(f) Genesis Shift ({contrib_G:.1f}%)',
        f'(g) Steering Flow Change ({contrib_S:.1f}%)'
    ]
    bot_col_slices = [(0, 4), (4, 8), (8, 12)]

    norm_bot = TwoSlopeNorm(vmin=-0.0008, vcenter=0, vmax=0.0008)
    levels_bot = np.linspace(-0.0008, 0.0008, 17)
    ticks_bot = np.linspace(-0.0008, 0.0008, 9)

    ny, nx = bot_data[0].shape
    x_bot = np.linspace(100, 180, nx)
    y_bot = np.linspace(0, 50, ny)

    cf_bot = None
    for idx in range(3):
        c_start, c_end = bot_col_slices[idx]
        # 使用 gs_bot[0, ...]
        ax = fig.add_subplot(gs_bot[0, c_start:c_end], projection=ccrs.PlateCarree())
        ax.set_anchor('S')

        ax.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', linewidth=0.8, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(axis='both', which='major', labelsize=12)

        if idx > 0:
            ax.set_yticklabels([])

        cf_bot = ax.contourf(
            x_bot, y_bot, bot_data[idx],
            levels=levels_bot, cmap='RdBu_r', norm=norm_bot,
            extend='both', transform=ccrs.PlateCarree()
        )

        ax.set_title(bot_titles[idx], fontsize=16, fontweight='bold', pad=10)

    # 下排共用 Colorbar (使用 gs_bot[1, 1:11])
    cbar_ax_bot = fig.add_subplot(gs_bot[1, 1:11])
    cbar_bot = fig.colorbar(cf_bot, cax=cbar_ax_bot, orientation='horizontal', ticks=ticks_bot)
    cbar_bot.set_label('Density Change', fontsize=16, fontweight='bold')
    cbar_bot.ax.tick_params(labelsize=16)

    # ---------------------------------------------------------------------
    # 輸出儲存
    # ---------------------------------------------------------------------
    plt.savefig('Figure8.png', dpi=400, bbox_inches='tight')
    plt.show()