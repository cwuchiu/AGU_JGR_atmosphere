import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.stats import linregress
from tqdm import trange

# ==========================================
# 0. 全域基礎參數設定
# ==========================================
start_year = 1979
end_year = 2024
years = np.arange(start_year, end_year + 1)
n_years = len(years)

# 通用線性趨勢計算函式 (向量化加速，算整段期間 46 年的總變化趨勢)
def compute_linear_trend_map(annual_data):
    x = np.arange(n_years)
    y = annual_data
    x_mean = np.mean(x)
    y_mean = np.nanmean(y, axis=0)
    numerator = np.nansum((x[:, None, None] - x_mean) * (y - y_mean), axis=0)
    denominator = np.sum((x - x_mean)**2)
    return (numerator / denominator) * 10 #decadal

# ==========================================
# 1. 計算 Vorticity (850 hPa) 年平均與趨勢
# ==========================================
print("--> Processing Vorticity Data...")
vort_file = '/Dellwork4/cwuchiu/TC_time_vorticity_noET_34kt.nc'
ds_vort = nc.Dataset(vort_file)
time_var_vort = ds_vort.variables['time']
lat_map = ds_vort.variables['lat'][:]
lon_map = ds_vort.variables['lon'][:]
vort850_var = ds_vort.variables['vort_850']

vort_dates = nc.num2date(time_var_vort[:], time_var_vort.units)
vort_years = np.array([d.year for d in vort_dates])

annual_v850 = np.full((n_years, len(lat_map), len(lon_map)), np.nan)
for i, yr in enumerate(years):
    indices = np.where(vort_years == yr)[0]
    if len(indices) == 0: continue
    data_850 = vort850_var[indices[0]:indices[-1] + 1, :, :]
    if np.ma.is_masked(data_850):
        data_850 = data_850.filled(np.nan)
    annual_v850[i, :, :] = np.nanmean(data_850, axis=0)
ds_vort.close()

trend_850 = compute_linear_trend_map(annual_v850)

# ==========================================
# 2. 計算 VWS 年平均與趨勢
# ==========================================
print("--> Processing VWS Data...")
vws_file = '/Dellwork4/cwuchiu/TC_time_VWS_noET_34kt.nc'
ds_vws = nc.Dataset(vws_file)
time_var_vws = ds_vws.variables['time']
vws_var = ds_vws.variables['vws']

vws_dates = nc.num2date(time_var_vws[:], time_var_vws.units)
vws_years = np.array([d.year for d in vws_dates])

annual_vws = np.full((n_years, len(lat_map), len(lon_map)), np.nan)
for i, yr in enumerate(years):
    indices = np.where(vws_years == yr)[0]
    if len(indices) == 0: continue
    data_vws = vws_var[indices[0]:indices[-1] + 1, :, :]
    if np.ma.is_masked(data_vws):
        data_vws = data_vws.filled(np.nan)
    annual_vws[i, :, :] = np.nanmean(data_vws, axis=0)
ds_vws.close()

trend_vws = compute_linear_trend_map(annual_vws)

# ==========================================
# 3. 計算 Monsoon Trough (經緯度) 年平均
# ==========================================
print("--> Processing Monsoon Trough Data...")
with nc.Dataset('/Dellwork4/cwuchiu/TC_time_1979_2024_noET_34kt.nc') as nc_tc:
    time_tc = [str(x) for x in np.array(nc_tc.variables['time'])]

with nc.Dataset('/Dellwork4/cwuchiu/TC_time_MT_noET_34kt.nc') as nc_mg:
    la = np.array(nc_mg.variables['mt_lat'])
    lo = np.array(nc_mg.variables['mt_lon'])

sum_lat = np.zeros(n_years)
sum_lon = np.zeros(n_years)
count_mt = np.zeros(n_years)

for i in trange(len(time_tc), desc="MT averaging"):
    current_year = int(time_tc[i][0:4])
    if start_year <= current_year <= end_year:
        idx = current_year - start_year
        sp_mean_lat = np.nanmean(la[i])
        sp_mean_lon = np.nanmean(lo[i])
        
        if not np.isnan(sp_mean_lat):
            sum_lat[idx] += sp_mean_lat
            sum_lon[idx] += sp_mean_lon
            count_mt[idx] += 1

MG_annual_mean_lat = np.full(n_years, np.nan)
MG_annual_mean_lon = np.full(n_years, np.nan)

valid_mask = count_mt > 0
MG_annual_mean_lat[valid_mask] = sum_lat[valid_mask] / count_mt[valid_mask]
MG_annual_mean_lon[valid_mask] = sum_lon[valid_mask] / count_mt[valid_mask]
#%%
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter

# =========================================================================
# 3.5 高斯平滑處理 (支援 NaN 缺測值)
# =========================================================================
def nan_gaussian_filter(data, sigma=1.5):
    if not np.isnan(data).any():
        return gaussian_filter(data, sigma=sigma)
    
    nan_mask = np.isnan(data)
    data_zeroed = np.nan_to_num(data, nan=0.0)
    weights = (~nan_mask).astype(float)
    
    smoothed_data = gaussian_filter(data_zeroed, sigma=sigma)
    smoothed_weights = gaussian_filter(weights, sigma=sigma)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = smoothed_data / smoothed_weights
    
    result[smoothed_weights < 0.1] = np.nan
    result[nan_mask] = np.nan
    return result

smooth_sigma = 1.5
print(f"--> Applying Gaussian filter (sigma={smooth_sigma})...")
trend_850_smooth = nan_gaussian_filter(trend_850, sigma=smooth_sigma)
trend_vws_smooth = nan_gaussian_filter(trend_vws, sigma=smooth_sigma)

# 定義 a 圖等值線標籤格式 (顯示如 -4e-6, 0, 4e-6)
def fmt_vort(x):
    if abs(x) < 1e-10:
        return "0"
    return f"{x:.0e}".replace("e-0", "e-")

# =========================================================================
# 4. 統一繪製 2x2 子圖 (間距 4e-6 + 等值線標籤)
# =========================================================================
print("--> Plotting combined 2x2 figure...")
fig = plt.figure(figsize=(21, 12))

# 地圖經緯度刻度 (100-180E, 0-50N)
lon_ticks = np.arange(100, 181, 20)  # [100, 120, 140, 160, 180]
lat_ticks = np.arange(0, 51, 10)     # [0, 10, 20, 30, 40, 50]
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()

# -------------------------------------------------------------------------
# (a) 左上：850 hPa Vorticity Trend (間距 4e-6 + 標籤)
# -------------------------------------------------------------------------
ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())

# 設定區間：4e-6 一格 ([-8e-6, -4e-6, 0, 4e-6, 8e-6])
levels_vort = np.arange(-3e-6, 3.1e-6, 1e-6)     
ticks_vort = np.arange(-3e-6, 3.1e-6, 1e-6)      
cmap_vort = plt.get_cmap('RdBu_r', len(levels_vort) - 1)

# 1. 填色
cf1 = ax1.contourf(lon_map, lat_map, trend_850_smooth, 
                    levels=levels_vort, cmap=cmap_vort,
                    transform=ccrs.PlateCarree())

# 2. 等值線
cs1 = ax1.contour(lon_map, lat_map, trend_850_smooth, 
                   levels=levels_vort, colors='k', linewidths=0.7,
                   transform=ccrs.PlateCarree())

# 3. 加上等值線數字標籤
ax1.clabel(cs1, inline=True, fontsize=12, fmt=fmt_vort)

ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linestyle=':')
ax1.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
ax1.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
ax1.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)
ax1.tick_params(axis='both', labelsize=16)

# Colorbar
cax1 = inset_axes(ax1, width="3.5%", height="100%", loc='lower left',
                  bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax1.transAxes, borderpad=0)
fmt1 = ticker.ScalarFormatter(useMathText=True)
fmt1.set_powerlimits((0, 0))
cbar1 = plt.colorbar(cf1, cax=cax1, orientation='vertical', ticks=ticks_vort, format=fmt1)
cbar1.ax.tick_params(labelsize=14)
cbar1.set_label(r'Trend (s$^{-1}$ decade$^{-1}$)', fontsize=16)
ax1.set_title('(a) 850 hPa Vorticity Trend (1979-2024)', fontsize=18, fontweight='bold')

# -------------------------------------------------------------------------
# (b) 右上：Vertical Wind Shear Trend (標籤)
# -------------------------------------------------------------------------
ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())

# 設定區間：1 m/s 一格 ([-5, -4, ..., 4, 5])
levels_vws = np.arange(-1.5, 1.6, 0.5)             
ticks_vws = np.arange(-1.5, 1.6, 0.5)                  
cmap_vws = plt.get_cmap('RdBu_r', len(levels_vws) - 1)

# 1. 填色
cf2 = ax2.contourf(lon_map, lat_map, trend_vws_smooth, 
                    levels=levels_vws, cmap=cmap_vws,
                    transform=ccrs.PlateCarree())

# 2. 等值線
cs2 = ax2.contour(lon_map, lat_map, trend_vws_smooth, 
                   levels=levels_vws, colors='k', linewidths=0.7,
                   transform=ccrs.PlateCarree())

# 3. 加上等值線數字標籤
ax2.clabel(cs2, inline=True, fontsize=12, fmt='%1.0f')

ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linestyle=':')
ax2.set_extent([100, 180, 0, 50], crs=ccrs.PlateCarree())
ax2.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
ax2.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)
ax2.tick_params(axis='both', labelsize=16)

# Colorbar
cax2 = inset_axes(ax2, width="3.5%", height="100%", loc='lower left',
                  bbox_to_anchor=(1.02, 0., 1, 1), bbox_transform=ax2.transAxes, borderpad=0)
cbar2 = plt.colorbar(cf2, cax=cax2, orientation='vertical', ticks=ticks_vws)
cbar2.ax.tick_params(labelsize=14)
cbar2.set_label(r'VWS Trend (m s$^{-1}$ decade$^{-1}$)', fontsize=16)
ax2.set_title('(b) Vertical Wind Shear Trend (1979-2024)', fontsize=18, fontweight='bold')

# -------------------------------------------------------------------------
# (c) 左下：Monsoon Trough Longitude
# -------------------------------------------------------------------------
ax3 = fig.add_subplot(2, 2, 3)
slope_lon, intercept_lon, _, p_val_lon, _ = linregress(years, MG_annual_mean_lon)

ax3.plot(years, MG_annual_mean_lon, marker='o', linestyle='-', color='royalblue')
label_lon = rf"Trend: {slope_lon*10:.2f} $^\circ$E decade$^{{-1}}$ (p={p_val_lon:.2e})"
ax3.plot(years, years * slope_lon + intercept_lon, 'r--', label=label_lon)

ax3.set_title('(c) Monsoon Trough Longitude (1979-2024)', fontsize=18, fontweight='bold')
ax3.set_xlabel('Year', fontsize=16)
ax3.set_ylabel(r'Longitude ($^\circ$E)', fontsize=16)
ax3.tick_params(axis='both', labelsize=16)

ax3.set_xlim(1979, 2024)
ax3.set_xticks(np.linspace(1979, 2024, 10).astype(int))
ax3.set_ylim(120, 145)
ax3.set_yticks(np.linspace(120, 145, 6))
ax3.legend(loc='upper right', fontsize=16)
ax3.grid(True, linestyle=':', alpha=0.6)

# 固定高寬比以保持與上排地圖尺寸完全相同
ax3.set_box_aspect(50 / 80)

# -------------------------------------------------------------------------
# (d) 右下：Monsoon Trough Latitude
# -------------------------------------------------------------------------
ax4 = fig.add_subplot(2, 2, 4)
slope_lat, intercept_lat, _, p_val_lat, _ = linregress(years, MG_annual_mean_lat)

ax4.plot(years, MG_annual_mean_lat, marker='o', linestyle='-', color='darkgreen')
label_lat = rf"Trend: {slope_lat*10:.2f} $^\circ$N decade$^{{-1}}$ (p={p_val_lat:.2e})"
ax4.plot(years, years * slope_lat + intercept_lat, 'r--', label=label_lat)

ax4.set_title('(d) Monsoon Trough Latitude (1979-2024)', fontsize=18, fontweight='bold')
ax4.set_xlabel('Year', fontsize=16)
ax4.set_ylabel(r'Latitude ($^\circ$N)', fontsize=16)
ax4.tick_params(axis='both', labelsize=16)

ax4.set_xlim(1979, 2024)
ax4.set_xticks(np.linspace(1979, 2024, 10).astype(int))
ax4.set_ylim(13, 18)
ax4.set_yticks(np.linspace(13, 18, 9))
ax4.legend(loc='upper right', fontsize=16)
ax4.grid(True, linestyle=':', alpha=0.6)

# 固定高寬比以保持與上排地圖尺寸完全相同
ax4.set_box_aspect(50 / 80)


# ------------------------------------------
# 儲存與顯示
# ------------------------------------------
plt.tight_layout()
plt.savefig('Figure9.png', dpi=400, bbox_inches='tight')
plt.show()