import numpy as np
import netCDF4 as nc
from multiprocessing import Pool
import warnings
from datetime import datetime
import os

# === 0. 設定檔案路徑與基本參數 ===
file_path_hgt = 'Tvbar_hgt_adjust_hourly_mean_2004.nc'  # 輸入路徑
output_file   = 'WNPSH_ridge_lat_1979_2024.nc'          # 輸出的 nc 檔案路徑

n_years = 46                 # 1979 至 2024 共 46 年
hours_per_year = 4392        # 每年 6~11 月共 4392 小時
target_years = np.arange(1979, 2025)
MAX_JUMP_LAT = 1.0           # 相鄰網格最大允許跳躍緯度 (度)

# === 1. 定義 Worker 函數 (平行運算子進程) ===
def process_chunk(args):
    start_idx, count, f_path, lat_arr, lon_arr, max_jump = args
    
    ds = nc.Dataset(f_path, 'r')
    var = ds.variables['hgt500']
    data_chunk = var[start_idx : start_idx + count]
    ds.close()
    
    # -------------------------------------------------------------
    # 【防禦修正 1】確保 data_chunk 絕對是 3D: (count, lat, lon)
    # 避免資料自帶 level=1 維度導致 axis 判斷錯誤
    # -------------------------------------------------------------
    if data_chunk.ndim == 4:
        data_chunk = np.squeeze(data_chunk)
        if data_chunk.ndim == 4:
            data_chunk = data_chunk[:, 0, :, :]  # 若有多層則取第一層
            
    # 【防禦修正 2】確保經緯度座標絕對是 1D 向量
    lat_arr = np.squeeze(lat_arr)
    if lat_arr.ndim == 2:
        lat_arr = lat_arr[:, 0]
        
    lon_arr = np.squeeze(lon_arr)
    if lon_arr.ndim == 2:
        lon_arr = lon_arr[0, :]
    
    # 1. 沿緯度軸 (axis=1) 找最大值與索引
    max_val = np.max(data_chunk, axis=1)        # Shape: (count, lon)
    argmax_idx = np.argmax(data_chunk, axis=1)  # Shape: (count, lon)
    
    # 2. 門檻值遮罩 (> 5880 gpm)
    mask = (max_val > 5880)
    
    # 3. 映射回實際緯度
    raw_ridge_lats = lat_arr[argmax_idx].astype(float)
    raw_ridge_lats[~mask] = np.nan
    
    # 4. 空間連續性檢驗：相鄰經度跳躍不超過 max_jump 度
    n_times, n_lon = raw_ridge_lats.shape
    filtered_ridge_lats = np.full_like(raw_ridge_lats, np.nan)
    
    chunk_ridge_lat = np.full(n_times, np.nan, dtype=np.float32)
    chunk_ridge_tilt = np.full(n_times, np.nan, dtype=np.float32)
    
    for t in range(n_times):
        row_lats = raw_ridge_lats[t]
        row_hgt = max_val[t]
        
        valid_indices = np.where(~np.isnan(row_lats))[0]
        if len(valid_indices) == 0:
            continue
            
        # 找出副高最強核心 (最大位勢高度所在的經度) 作為錨點
        anchor_idx = valid_indices[np.argmax(row_hgt[valid_indices])]
        filtered_ridge_lats[t, anchor_idx] = row_lats[anchor_idx]
        
        # 向東 (右) 追蹤延伸
        curr_lat = float(row_lats[anchor_idx])  # 強制轉為純量 float
        for j in range(anchor_idx + 1, n_lon):
            if np.isnan(row_lats[j]):
                break
            val_lat = float(row_lats[j])        # 強制轉為純量 float，避免陣列比較錯誤
            if abs(val_lat - curr_lat) <= max_jump:
                filtered_ridge_lats[t, j] = val_lat
                curr_lat = val_lat
            else:
                break
                
        # 向西 (左) 追蹤延伸
        curr_lat = float(row_lats[anchor_idx])  # 強制轉為純量 float
        for j in range(anchor_idx - 1, -1, -1):
            if np.isnan(row_lats[j]):
                break
            val_lat = float(row_lats[j])        # 強制轉為純量 float
            if abs(val_lat - curr_lat) <= max_jump:
                filtered_ridge_lats[t, j] = val_lat
                curr_lat = val_lat
            else:
                break
                
        # 5. 計算該時間步的「脊線緯度平均」與「脊線傾角斜率」
        active_idx = np.where(~np.isnan(filtered_ridge_lats[t]))[0]
        if len(active_idx) > 0:
            # (A) 脊線緯度平均
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                chunk_ridge_lat[t] = np.nanmean(filtered_ridge_lats[t])
                
            # (B) 脊線傾角 (至少需 2 點擬合斜率；單點視為平直 0.0)
            if len(active_idx) >= 2:
                x_lon = lon_arr[active_idx]
                y_lat = filtered_ridge_lats[t, active_idx]
                slope, _ = np.polyfit(x_lon, y_lat, 1)
                chunk_ridge_tilt[t] = slope
            else:
                chunk_ridge_tilt[t] = 0.0
                
    return chunk_ridge_lat, chunk_ridge_tilt


# === 2. 主程式 ===
if __name__ == '__main__':
    ds_check = nc.Dataset(file_path_hgt, 'r')
    hgt_shape = ds_check.variables['hgt500'].shape
    
    # 自動識別空間維度大小
    if len(hgt_shape) == 4:
        n_lat, n_lon = hgt_shape[2], hgt_shape[3]
    else:
        n_lat, n_lon = hgt_shape[1], hgt_shape[2]
        
    # 取得緯度座標 (保證 1D)
    if 'lat' in ds_check.variables:
        lat = np.array(ds_check.variables['lat'][:])
    elif 'latitude' in ds_check.variables:
        lat = np.array(ds_check.variables['latitude'][:])
    else:
        lat = np.linspace(50, 0, n_lat)
        
    lat = np.squeeze(lat)
    if lat.ndim == 2:
        lat = lat[:, 0]
        
    # 取得經度座標 (保證 1D)
    if 'lon' in ds_check.variables:
        lon = np.array(ds_check.variables['lon'][:])
    elif 'longitude' in ds_check.variables:
        lon = np.array(ds_check.variables['longitude'][:])
    else:
        lon = np.linspace(100, 180, n_lon)
        
    lon = np.squeeze(lon)
    if lon.ndim == 2:
        lon = lon[0, :]
        
    total_len = hgt_shape[0]
    
    # 讀取來源檔案的時間變數資訊 (若存在)
    has_time_var = 'time' in ds_check.variables
    if has_time_var:
        src_time_vals = ds_check.variables['time'][:]
        src_time_units = getattr(ds_check.variables['time'], 'units', 'hours')
        src_time_calendar = getattr(ds_check.variables['time'], 'calendar', 'standard')
    ds_check.close()
    
    print(f"Dataset Total Timesteps: {total_len}")
    print(f"Detected 2D Grid Resolution: {len(lat)} latitudes x {len(lon)} longitudes")
    print(f"Continuity constraint: Adjacent ridge jump <= {MAX_JUMP_LAT}° lat")
    
    # 切片任務設定
    chunk_size = 2000
    tasks = []
    for i in range(0, total_len, chunk_size):
        count = min(chunk_size, total_len - i)
        tasks.append((i, count, file_path_hgt, lat, lon, MAX_JUMP_LAT))
    
    print(f"Start processing with Pool (Chunks: {len(tasks)})...")
    
    with Pool() as pool:
        results = pool.map(process_chunk, tasks)
    
    # 解包多行程回傳的脊線緯度與脊線傾角
    hourly_ridge_lat = np.concatenate([r[0] for r in results])
    hourly_ridge_tilt = np.concatenate([r[1] for r in results])
    print("Step 1 Done: Hourly ridge latitude and tilt calculated.")

    # === 3. 計算各年年平均 (6~11月) ===
    if total_len == n_years * hours_per_year:
        ridge_lat_2d = hourly_ridge_lat.reshape(n_years, hours_per_year)
        ridge_tilt_2d = hourly_ridge_tilt.reshape(n_years, hours_per_year)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            yearly_ridge_lat = np.nanmean(ridge_lat_2d, axis=1)
            yearly_ridge_tilt = np.nanmean(ridge_tilt_2d, axis=1)
    else:
        yearly_ridge_lat = []
        yearly_ridge_tilt = []
        for i in range(n_years):
            s_idx = i * hours_per_year
            e_idx = min((i + 1) * hours_per_year, total_len)
            if s_idx < total_len:
                yr_slice_lat = hourly_ridge_lat[s_idx:e_idx]
                yr_slice_tilt = hourly_ridge_tilt[s_idx:e_idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    yearly_ridge_lat.append(np.nanmean(yr_slice_lat))
                    yearly_ridge_tilt.append(np.nanmean(yr_slice_tilt))
            else:
                yearly_ridge_lat.append(np.nan)
                yearly_ridge_tilt.append(np.nan)
        yearly_ridge_lat = np.array(yearly_ridge_lat)
        yearly_ridge_tilt = np.array(yearly_ridge_tilt)

    print("Step 2 Done: Yearly ridge latitude and tilt created.")

    # === 4. 輸出為 NetCDF 檔案 ===
    print(f"Saving output to NetCDF: {output_file} ...")
    
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:
        # 1. 建立維度
        ds_out.createDimension('time', total_len)
        ds_out.createDimension('year', n_years)
        
        # 2. 建立座標變數
        var_year = ds_out.createVariable('year', 'i4', ('year',))
        var_year.standard_name = 'year'
        var_year.long_name = 'Year'
        var_year.units = 'year'
        var_year[:] = target_years
        
        if has_time_var:
            var_time = ds_out.createVariable('time', 'f8', ('time',))
            var_time.standard_name = 'time'
            var_time.long_name = 'Time'
            var_time.units = src_time_units
            var_time.calendar = src_time_calendar
            var_time[:] = src_time_vals
        else:
            var_time = ds_out.createVariable('time', 'i4', ('time',))
            var_time.long_name = 'Time step index'
            var_time[:] = np.arange(total_len)

        # 3. 建立資料變數
        # (A) 每小時脊線緯度
        var_hourly_lat = ds_out.createVariable('ridge_lat_hourly', 'f4', ('time',), fill_value=np.nan)
        var_hourly_lat.standard_name = 'latitude'
        var_hourly_lat.long_name = 'Hourly WNPSH Ridge Latitude'
        var_hourly_lat.units = 'degrees_north'
        var_hourly_lat.description = 'Mean ridge latitude across 100E-180E (hgt500 > 5880 gpm, adjacent jump <= 1 deg)'
        var_hourly_lat[:] = hourly_ridge_lat

        # (B) 各年 6~11 月平均脊線緯度
        var_yearly_lat = ds_out.createVariable('ridge_lat_yearly', 'f4', ('year',), fill_value=np.nan)
        var_yearly_lat.standard_name = 'latitude'
        var_yearly_lat.long_name = 'Annual Mean (Jun-Nov) WNPSH Ridge Latitude'
        var_yearly_lat.units = 'degrees_north'
        var_yearly_lat.description = 'Annual average (Jun-Nov) of hourly WNPSH ridge latitudes'
        var_yearly_lat[:] = yearly_ridge_lat

        # (C) 每小時脊線傾角斜率
        var_hourly_tilt = ds_out.createVariable('ridge_tilt_hourly', 'f4', ('time',), fill_value=np.nan)
        var_hourly_tilt.long_name = 'Hourly WNPSH Ridge Tilt Slope'
        var_hourly_tilt.units = 'degrees_north / degrees_east'
        var_hourly_tilt.description = 'Linear regression slope (d_lat/d_lon) of continuous WNPSH ridge line'
        var_hourly_tilt[:] = hourly_ridge_tilt

        # (D) 各年 6~11 月平均脊線傾角斜率
        var_yearly_tilt = ds_out.createVariable('ridge_tilt_yearly', 'f4', ('year',), fill_value=np.nan)
        var_yearly_tilt.long_name = 'Annual Mean (Jun-Nov) WNPSH Ridge Tilt Slope'
        var_yearly_tilt.units = 'degrees_north / degrees_east'
        var_yearly_tilt.description = 'Annual average (Jun-Nov) of hourly WNPSH ridge tilt slopes'
        var_yearly_tilt[:] = yearly_ridge_tilt

        # 4. 加入全域屬性
        ds_out.title = 'Western North Pacific Subtropical High (WNPSH) Ridge Latitude and Tilt'
        ds_out.source_file = file_path_hgt
        ds_out.period = '1979-2024 (June to November)'
        ds_out.spatial_domain = '100E - 180E, 0N - 50N'
        ds_out.subtropical_high_threshold = '5880 gpm'
        ds_out.max_adjacent_jump = f'{MAX_JUMP_LAT} degrees'
        ds_out.created_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print(f"Successfully created: {output_file}")