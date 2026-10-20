#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from numpy import array, linspace
from scipy.stats import gaussian_kde, linregress
import xarray as xr

# =============================================================================
# 1. 讀取 IBTrACS 資料
# =============================================================================
ncfile = nc.Dataset('/Dellwork5/cwuchiu/IBTrACS.WP.v04r01.nc')

ar = xr.open_dataset('/Dellwork5/cwuchiu/IBTrACS.WP.v04r01.nc')


lon = array(ncfile.variables['usa_lon'])
lat = array(ncfile.variables['usa_lat'])
year = array(ncfile.variables['season'])
number = array(ncfile.variables['number'])
usa_wind = array(ncfile.variables['usa_wind'])

source_usa = array(ncfile.variables['source_usa'])


nature = ar['nature'].data  # b'TS' = 熱帶, b'ET' = 溫帶, b'SS' = 亞熱帶
sid = ar['sid'].data
raw_time = ar['iso_time'].data
name = ar['name'].data
basin = ar['basin'].data
basin1 = basin[:,0]
track_type = ar['track_type'].data
storm_dir = ar['storm_dir'].data
ncfile.close()
# =============================================================================
# 2. 篩選 1979-2024 年 JJASON (6-11月) 的西北太平洋颱風路徑
# =============================================================================
save_lon = []
save_lat = []
save_time = []
save_wind = []

JJASON = np.zeros(raw_time.shape[0])
for i in range(JJASON.shape[0]):
    if all(6 <= int(t[5:7]) <= 11 for t in raw_time[i, :] if t):
        JJASON[i] = 1
    else:
        JJASON[i] = 0

for j in range(1979, 2025):
    ab = []
    ba = []
    ccc = []
    www = []
    max_wind = np.nanmax(usa_wind, axis=1)
    if j == 2024:
        tcmask = np.where(
            (year == j)
            & (basin1 == b"WP")
            & (track_type == b"PROVISIONAL")
            & (JJASON == 1)
            & (max_wind >= 34)
        )
    else:
        tcmask = np.where(
            (year == j)
            & (basin1 == b"WP")
            & (track_type == b"main")
            & (JJASON == 1)
            & (max_wind >= 34)
        )

    lon_tc = lon[tcmask]
    lat_tc = lat[tcmask]
    time_tc_track = raw_time[tcmask]
    usa_wind_tc = usa_wind[tcmask]
    nature_tc = nature[tcmask]
    for i in range(len(lon_tc)):
        valid_idx = (
            (lat_tc[i, :] != -9999)
            & (lon_tc[i, :] >= 100)
            & (lat_tc[i, :] >= 0)
            & (nature_tc[i, :] != b'ET')
        )
        tclat = lat_tc[i, valid_idx]
        tclon = lon_tc[i, valid_idx]
        tct = time_tc_track[i, valid_idx]
        tcwind = usa_wind_tc[i, valid_idx]

        ab.append(tclon)
        ba.append(tclat)
        ccc.append(tct)
        www.append(tcwind)

    save_lon.append(ab)
    save_lat.append(ba)
    save_time.append(ccc)
    save_wind.append(www)

# 去除空軌跡
for i in range(46):
    save_lon[i] = [track for track in save_lon[i] if len(track) > 0]
    save_lat[i] = [track for track in save_lat[i] if len(track) > 0]
    save_time[i] = [track for track in save_time[i] if len(track) > 0]
    save_wind[i] = [track for track in save_wind[i] if len(track) > 0]

# =============================================================================
# 3. 取得各年生成點 (第一點) 及年度平均經緯度
# =============================================================================
save_lon_first = [[] for _ in range(46)]
save_lat_first = [[] for _ in range(46)]
save_time_first = [[] for _ in range(46)]

save_lon_first_flat = []
save_lat_first_flat = []
save_time_first_flat = []

for i in range(len(save_lon)):
    for j in range(len(save_lon[i])):
        save_lon_first[i].append(save_lon[i][j][0])
        save_lat_first[i].append(save_lat[i][j][0])
        save_time_first[i].append(save_time[i][j][0])

        save_lon_first_flat.append(save_lon[i][j][0])
        save_lat_first_flat.append(save_lat[i][j][0])
        save_time_first_flat.append(save_time[i][j][0])

genesis_lon = np.zeros(46)
genesis_lat = np.zeros(46)
for i in range(46):
    genesis_lon[i] = np.nanmean(save_lon_first[i])
    genesis_lat[i] = np.nanmean(save_lat_first[i])

years = np.linspace(1979, 2024, 46)


# =============================================================================
# 4. 計算整體生成密度趨勢 (Genesis Density Trend, z1)
# =============================================================================
def compute_track_density(
    save_lon,
    save_lat,
    lon_min=100,
    lon_max=180,
    lat_min=0,
    lat_max=50,
    dlon=1.0,
    dlat=1.0,
):
    lon_bins = np.arange(lon_min, lon_max + dlon, dlon)
    lat_bins = np.arange(lat_min, lat_max + dlat, dlat)
    n_years = len(save_lon)

    density_all = np.zeros(
        (n_years, len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.float64
    )
    save_total_point = np.zeros(n_years)

    for y in range(n_years):
        density = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
        total_points = 0
        for i in range(len(save_lon[y])):
            ty_lons = np.atleast_1d(save_lon[y][i])
            ty_lats = np.atleast_1d(save_lat[y][i])
            total_points += 1

            lon_idx = np.digitize(ty_lons, lon_bins) - 1
            lat_idx = np.digitize(ty_lats, lat_bins) - 1
            valid = (
                (lon_idx >= 0)
                & (lon_idx < len(lon_bins) - 1)
                & (lat_idx >= 0)
                & (lat_idx < len(lat_bins) - 1)
            )

            lon_idx = lon_idx[valid]
            lat_idx = lat_idx[valid]
            for r, c in zip(lat_idx, lon_idx):
                density[r, c] += 1

        if total_points > 0:
            density /= total_points
        save_total_point[y] = total_points
        density_all[y] = density

    return density_all, lon_bins, lat_bins, save_total_point


def compute_kde(lon, lat, mask, weights):
    x = lon[mask]
    y = lat[mask]
    if len(x) == 0:
        return np.zeros_like(lon)
    w = weights[mask] if weights is not None else None
    coords = np.vstack([x, y])
    kernel = gaussian_kde(coords, bw_method="silverman", weights=w)
    positions = np.vstack([lon.ravel(), lat.ravel()])
    z = np.reshape(kernel(positions), lon.shape)
    return z


density1, lon_bins, lat_bins, _ = compute_track_density(
    save_lon_first, save_lat_first
)
lon_grid, lat_grid = np.meshgrid(lon_bins[:-1], lat_bins[:-1])

time_mean = np.mean(years)
d1_mean = np.mean(density1, axis=0)
numerator_direct = np.sum(
    (years[:, None, None] - time_mean) * (density1 - d1_mean[None, :, :]),
    axis=0,
)
denominator_direct = np.sum((years[:, None, None] - time_mean) ** 2, axis=0)
dd1dt = numerator_direct / denominator_direct

z_increase = compute_kde(lon_grid, lat_grid, dd1dt > 0, weights=dd1dt)
z_decrease = -compute_kde(
    lon_grid, lat_grid, dd1dt < 0, weights=np.abs(dd1dt)
)  # 負值代表減少
z1 = z_increase + z_decrease

# =============================================================================
# 5. 結合 WNPSH Cluster 計算型態變化的貢獻 (z2)
# =============================================================================
ncfile = nc.Dataset("/Dellwork4/cwuchiu/Cluster4_WNPSH_centroid_Ridge_SHAI_noET_34kt.nc")
which_cluster = np.array(ncfile.variables["which_cluster"])
ncfile.close()

ncfile = nc.Dataset("/Dellwork4/cwuchiu/TC_time_1979_2024_not_set_noET_34kt.nc")
tt = np.array(ncfile.variables["time"])
time_tc = [str(x) for x in tt]
ncfile.close()

save_time_str = np.array([t.decode("utf-8") for t in save_time_first_flat])
save_time_fmt = np.array([
    datetime.strptime(t, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H")
    for t in save_time_str
])

start_year = 1979
end_year = 2024
num_years = end_year - start_year + 1

genesis_cluster_lon = [[[] for _ in range(5)] for _ in range(num_years)]
genesis_cluster_lat = [[[] for _ in range(5)] for _ in range(num_years)]

for i in range(save_time_fmt.shape[0]):
    current_year = int(save_time_fmt[i][:4])
    if start_year <= current_year <= end_year:
        year_idx = current_year - start_year
        mask = np.isin(np.array(time_tc), save_time_fmt[i])
        indices_time_tc = np.where(mask)[0]
        for j in range(indices_time_tc.shape[0]):
            this_cluster = int(which_cluster[indices_time_tc[j]])
            genesis_cluster_lon[year_idx][this_cluster].append(
                save_lon_first_flat[i]
            )
            genesis_cluster_lat[year_idx][this_cluster].append(
                save_lat_first_flat[i]
            )


def compute_track_density_yearly(
    save_lon,
    save_lat,
    lon_min=100,
    lon_max=180,
    lat_min=0,
    lat_max=50,
    dlon=1.0,
    dlat=1.0,
):
    lon_bins = np.arange(lon_min, lon_max + dlon, dlon)
    lat_bins = np.arange(lat_min, lat_max + dlat, dlat)
    n_years = len(save_lon)
    n_cluster = len(save_lon[0])

    density_all = np.zeros(
        (n_years, n_cluster, len(lat_bins) - 1, len(lon_bins) - 1),
        dtype=np.float64,
    )

    for y in range(n_years):
        for c in range(n_cluster):
            density = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1))
            total_points = 0
            for ty_lons, ty_lats in zip(save_lon[y][c], save_lat[y][c]):
                ty_lons = np.atleast_1d(ty_lons)
                ty_lats = np.atleast_1d(ty_lats)
                valid = (~np.isnan(ty_lons)) & (~np.isnan(ty_lats))
                ty_lons = ty_lons[valid]
                ty_lats = ty_lats[valid]
                total_points += len(ty_lons)

                lon_idx = np.digitize(ty_lons, lon_bins) - 1
                lat_idx = np.digitize(ty_lats, lat_bins) - 1
                valid = (
                    (lon_idx >= 0)
                    & (lon_idx < len(lon_bins) - 1)
                    & (lat_idx >= 0)
                    & (lat_idx < len(lat_bins) - 1)
                )
                lon_idx = lon_idx[valid]
                lat_idx = lat_idx[valid]
                for r, c_idx in zip(lat_idx, lon_idx):
                    density[r, c_idx] += 1

            if total_points > 0:
                density /= total_points
            density_all[y, c] = density

    return density_all, lon_bins, lat_bins


density_cluster_yearly, _, _ = compute_track_density_yearly(
    genesis_cluster_lon, genesis_cluster_lat
)

for y in range(num_years):
    for c in range(5):
        sum_density = np.sum(density_cluster_yearly[y, c])
        if sum_density > 0:
            density_cluster_yearly[y, c] /= sum_density

density_mean = np.nanmean(density_cluster_yearly, axis=0)

# 計算各群頻率趨勢
cluster = np.zeros((5, 46))
for i in range(len(time_tc)):
    year_idx = int(time_tc[i][0:4]) - 1979
    cluster[int(which_cluster[i]), year_idx] += 1

cluster_sum = np.sum(cluster, axis=0)
slope_cluster = np.zeros(5)
for i in range(1,5):
    slope_cluster[i], _, _, _, _ = linregress(years, cluster[i] / cluster_sum)

cluster_early = slope_cluster[:5] * 1979
cluster_late = slope_cluster[:5] * 2024

d_total = np.sum(
    density_mean * cluster_late[:, None, None], axis=0
) - np.sum(density_mean * cluster_early[:, None, None], axis=0) / 46

z_increase_wnpsh = compute_kde(lon_grid, lat_grid, d_total > 0, weights=d_total)
z_decrease_wnpsh = -compute_kde(
    lon_grid, lat_grid, d_total < 0, weights=np.abs(d_total)
)
z2 = z_increase_wnpsh + z_decrease_wnpsh


# =============================================================================
# 6. 整合繪圖：上排 3 張 (a)(b)(c)，下排 2 張 (d)(e)
# =============================================================================
norm = TwoSlopeNorm(vmin=-0.001, vcenter=0, vmax=0.001)
custom_bwr = mcolors.LinearSegmentedColormap.from_list(
    "custom_bwr", [(0, "blue"), (0.48, "white"), (0.52, "white"), (1, "red")]
)
contour_levels = np.linspace(-0.001, 0.001, 11)

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    }
)

fig = plt.figure(figsize=(18, 12))
# 使用 2x6 格線配置
gs = gridspec.GridSpec(
    2, 6, figure=fig, height_ratios=[1, 1.15], hspace=0.32, wspace=0.55
)

# ---------------- (a) Genesis Longitude ----------------
ax1 = fig.add_subplot(gs[0, 0:2])
slope_a, intercept_a, _, p_val_a, _ = linregress(years, genesis_lon)
trend_a = slope_a * years + intercept_a
ax1.plot(
    years,
    trend_a,
    "--",
    color="crimson",
    linewidth=2,
    label=f"Trend: {slope_a*10:.2f}° decade$^{{-1}}$ (p={p_val_a:.2e})",
)
ax1.plot(
    years, genesis_lon, "o-", color="royalblue", linewidth=1.8, markersize=4
)
ax1.set_title("(a) Genesis Longitude", fontweight="bold", pad=10)
ax1.set_xlabel("Year")
ax1.set_ylabel("Longitude (°E)")
ax1.set_xticks(np.arange(1980, 2025, 10))
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper left")

# ---------------- (b) Genesis Latitude ----------------
ax2 = fig.add_subplot(gs[0, 2:4])
slope_b, intercept_b, _, p_val_b, _ = linregress(years, genesis_lat)
trend_b = slope_b * years + intercept_b
ax2.plot(
    years,
    trend_b,
    "--",
    color="crimson",
    linewidth=2,
    label=f"Trend: {slope_b*10:.2f}° decade$^{{-1}}$ (p={p_val_b:.2e})",
)
ax2.plot(
    years, genesis_lat, "o-", color="darkgreen", linewidth=1.8, markersize=4
)
ax2.set_title("(b) Genesis Latitude", fontweight="bold", pad=10)
ax2.set_xlabel("Year")
ax2.set_ylabel("Latitude (°N)")
ax2.set_xticks(np.arange(1980, 2025, 10))
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left")

# ---------------- (c) Annual Mean Genesis Location ----------------
ax3 = fig.add_subplot(gs[0, 4:6], projection=ccrs.PlateCarree())
ax3.add_feature(cfeature.COASTLINE, linewidth=0.8, color="gray")
ax3.set_extent([125, 155, 7.5, 22.5], crs=ccrs.PlateCarree())

gl3 = ax3.gridlines(
    draw_labels=True, linestyle="--", alpha=0.5, crs=ccrs.PlateCarree()
)
gl3.top_labels = False
gl3.right_labels = False
gl3.xlabel_style = {"size": 11}
gl3.ylabel_style = {"size": 11}
ax3.xaxis.set_major_formatter(LongitudeFormatter())
ax3.yaxis.set_major_formatter(LatitudeFormatter())

sc = ax3.scatter(
    genesis_lon,
    genesis_lat,
    c=years,
    cmap="seismic",
    s=45,
    edgecolor="k",
    linewidths=0.5,
    transform=ccrs.PlateCarree(),
    zorder=3,
)

# 繪製移動趨勢箭頭
lon_start, lon_end = (
    slope_a * years[0] + intercept_a,
    slope_a * years[-1] + intercept_a,
)
lat_start, lat_end = (
    slope_b * years[0] + intercept_b,
    slope_b * years[-1] + intercept_b,
)
ax3.annotate(
    "",
    xy=(lon_end, lat_end),
    xytext=(lon_start, lat_start),
    arrowprops=dict(
        facecolor="red", edgecolor="black", width=2, headwidth=7, alpha=0.85
    ),
    transform=ccrs.PlateCarree(),
    zorder=4,
)
ax3.set_title("(c) Annual Mean Genesis Location", fontweight="bold", pad=10)
ax3.set_aspect("auto")

cbar_c = plt.colorbar(
    sc,
    ax=ax3,
    orientation="horizontal",
    fraction=0.06,
    pad=0.22,
    aspect=30,
    ticks=np.linspace(1979, 2024, 10),
)
cbar_c.set_label("Year", fontsize=11)
cbar_c.ax.tick_params(labelsize=10)

# ---------------- (d) Genesis Density Trend (1979-2024) ----------------
ax4 = fig.add_subplot(gs[1, 0:3], projection=ccrs.PlateCarree())
ax4.coastlines(resolution="50m", linewidth=0.8)
ax4.add_feature(cfeature.BORDERS, linewidth=0.5)
ax4.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())

gl4 = ax4.gridlines(
    draw_labels=True, linestyle="--", alpha=0.5, crs=ccrs.PlateCarree()
)
gl4.top_labels = False
gl4.right_labels = False
gl4.xlabel_style = {"size": 11}
gl4.ylabel_style = {"size": 11}
ax4.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax4.yaxis.set_major_formatter(LatitudeFormatter())

ct4 = ax4.pcolormesh(
    lon_grid,
    lat_grid,
    z1,
    cmap=custom_bwr,
    norm=norm,
    transform=ccrs.PlateCarree(),
    alpha=0.9,
)
contour_lines4 = ax4.contour(
    lon_grid,
    lat_grid,
    z1,
    levels=contour_levels,
    colors="black",
    linewidths=0.4,
    transform=ccrs.PlateCarree(),
)
ax4.clabel(contour_lines4, inline=True, fontsize=8, fmt="%.4f")
ax4.set_title(
    "(d) Genesis Density Trend (1979-2024)", fontweight="bold", pad=10
)
ax4.set_aspect("auto")

# ---------------- (e) Genesis Density Contributed by WNPSH Type Change ----------------
ax5 = fig.add_subplot(gs[1, 3:6], projection=ccrs.PlateCarree())
ax5.coastlines(resolution="50m", linewidth=0.8)
ax5.add_feature(cfeature.BORDERS, linewidth=0.5)
ax5.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())

gl5 = ax5.gridlines(
    draw_labels=True, linestyle="--", alpha=0.5, crs=ccrs.PlateCarree()
)
gl5.top_labels = False
gl5.right_labels = False
gl5.xlabel_style = {"size": 11}
gl5.ylabel_style = {"size": 11}
ax5.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax5.yaxis.set_major_formatter(LatitudeFormatter())

ct5 = ax5.pcolormesh(
    lon_grid,
    lat_grid,
    z2,
    cmap=custom_bwr,
    norm=norm,
    transform=ccrs.PlateCarree(),
    alpha=0.9,
)
contour_lines5 = ax5.contour(
    lon_grid,
    lat_grid,
    z2,
    levels=contour_levels,
    colors="black",
    linewidths=0.4,
    transform=ccrs.PlateCarree(),
)
ax5.clabel(contour_lines5, inline=True, fontsize=8, fmt="%.4f")
ax5.set_title(
    "(e) Genesis Density Contributed by WNPSH Type Change",
    fontweight="bold",
    pad=10,
)
ax5.set_aspect("auto")

# ---------------- 下排 (d) 和 (e) 共用置底 Colorbar ----------------
cbar_bottom = fig.colorbar(
    ct5,
    ax=[ax4, ax5],
    orientation="horizontal",
    ticks=contour_levels,
    location="bottom",
    aspect=45,
    shrink=0.9,
    pad=0.10,
)
cbar_bottom.set_label("Genesis KDE", fontsize=14, labelpad=8)
cbar_bottom.ax.tick_params(labelsize=11)

# ================= 儲存與顯示 =================
plt.savefig("Figure7.png", dpi=400, bbox_inches="tight")
plt.show()