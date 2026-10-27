#%%

import os
import glob
from datetime import datetime
import numpy as np
import netCDF4 as nc
from multiprocessing import Pool
from tqdm import tqdm

# =============================================================================
# 1. 核心參數與系統資源防護鎖定
# =============================================================================
N_WORKERS = 32            # 可依伺服器負載調整（如 16 或 32）
CHUNK_SIZE = 128          # 每累積多少個時間步執行一次 NetCDF 批次整塊寫入
START_YEAR = 1979

DATA_DIR = '/Dellwork4/cwuchiu/UV_TC_noET_34kt/'
OUT_FILE = '/Dellwork4/cwuchiu/TC_time_steer_noET_34kt.nc'

# 強制底層線性代數函式庫單執行緒，防止子進程衍生出龐大執行緒風暴
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_WORKERS)

# Linux 核心層級物理隔離（綁定指定核心區間）
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
    """確保每個子進程嚴格限制為單執行緒"""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass

def _process_single_file(task_arg):
    """
    子進程任務：讀取單一檔案並計算 850-200 hPa 質量加權 steering flow
    """
    filepath, ftime, p_top, p_bot = task_arg
    try:
        with nc.Dataset(filepath, 'r') as ds:
            pres = ds.variables['pressure_level'][:]
            
            # 適應 (time, lev, lat, lon) 或 (lev, lat, lon)
            u_var = ds.variables['u']
            v_var = ds.variables['v']
            U = u_var[0] if u_var.ndim == 4 else u_var[:]
            V = v_var[0] if v_var.ndim == 4 else v_var[:]

            # ----------------------------------------------------
            # 向量化快速計算梯形積分 (Vectorized Trapezoidal Integration)
            # 避免 Python 在 vertical levels 的 for 迴圈
            # ----------------------------------------------------
            dp = pres[:-1] - pres[1:]
            weights = (dp / (p_bot - p_top))[:, np.newaxis, np.newaxis]
            
            U_mid = 0.5 * (U[:-1] + U[1:])
            V_mid = 0.5 * (V[:-1] + V[1:])

            U_steer = np.sum(U_mid * weights, axis=0).astype(np.float32)
            V_steer = np.sum(V_mid * weights, axis=0).astype(np.float32)

        return (ftime, U_steer, V_steer, None)

    except Exception as e:
        return (ftime, None, None, f"Error processing {os.path.basename(filepath)}: {str(e)}")

# =============================================================================
# 3. 輔助函數：NetCDF 輸出檔管理與 Chunk 寫入
# =============================================================================
def init_or_open_nc(out_path, sample_file, start_yr):
    """檢查或建立 NetCDF 檔，回傳 (Dataset, last_time)"""
    time_units = 'hours since 1970-01-01 00:00:00'
    calendar = 'standard'

    if os.path.exists(out_path):
        ds_out = nc.Dataset(out_path, 'a')
        tvar = ds_out.variables['time']
        if len(tvar) > 0:
            exist_times = nc.num2date(tvar[:], tvar.units, calendar=getattr(tvar, 'calendar', calendar))
            last_time = max(exist_times)
        else:
            last_time = datetime(start_yr, 1, 1, 0)
        print(f">>> 發現已存在輸出檔，接續處理時間: {last_time}")
    else:
        # 從範例檔獲取網格經緯度資訊
        with nc.Dataset(sample_file, 'r') as s_ds:
            lat = s_ds.variables['latitude'][:]
            lon = s_ds.variables['longitude'][:]

        ds_out = nc.Dataset(out_path, 'w')
        ds_out.createDimension('time', None)  # unlimited
        ds_out.createDimension('lat', len(lat))
        ds_out.createDimension('lon', len(lon))

        tvar = ds_out.createVariable('time', 'f8', ('time',))
        tvar.units = time_units
        tvar.calendar = calendar

        ds_out.createVariable('lat', 'f4', ('lat',))[:] = lat
        ds_out.createVariable('lon', 'f4', ('lon',))[:] = lon

        # 批次整塊壓縮儲存
        ds_out.createVariable('U_steer', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=1)
        ds_out.createVariable('V_steer', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=1)

        last_time = datetime(start_yr, 1, 1, 0)
        print(f">>> 已新建輸出檔: {out_path}，起始時間: {last_time}")

    return ds_out, last_time

def flush_chunk_to_nc(ds_out, chunk_buffer):
    """將緩衝區資料轉換為 3D Array 批次寫入 NetCDF"""
    if not chunk_buffer:
        return
    
    # 確保依照時間順序排列
    chunk_buffer.sort(key=lambda x: x[0])
    times = [x[0] for x in chunk_buffer]
    u_block = np.stack([x[1] for x in chunk_buffer], axis=0)
    v_block = np.stack([x[2] for x in chunk_buffer], axis=0)

    time_var = ds_out.variables['time']
    time_nums = nc.date2num(times, units=time_var.units, calendar=getattr(time_var, 'calendar', 'standard'))

    curr_idx = ds_out.dimensions['time'].size
    new_idx = curr_idx + len(times)

    # 批次切片寫入（效能遠高於一筆一筆寫）
    time_var[curr_idx:new_idx] = time_nums
    ds_out.variables['U_steer'][curr_idx:new_idx, :, :] = u_block
    ds_out.variables['V_steer'][curr_idx:new_idx, :, :] = v_block

    # 將記憶體快取強制作業系統 flush 至磁碟，防止異常中斷造成毀損
    ds_out.sync()
    chunk_buffer.clear()

# =============================================================================
# 4. 主平行執行流程
# =============================================================================
if __name__ == '__main__':
    # 1. 快速過濾所有檔案
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.nc')])
    if not all_files:
        print(f">>> 錯誤：在 {DATA_DIR} 找不到任何 .nc 檔！")
        exit()

    sample_filepath = os.path.join(DATA_DIR, all_files[0])
    ds_out, last_time = init_or_open_nc(OUT_FILE, sample_filepath, START_YEAR)

    # 2. 篩選尚未處理的任務
    tasks = []
    for f in all_files:
        try:
            t = datetime.strptime(f[:10], '%Y%m%d%H')
        except ValueError:
            continue

        if t > last_time:
            filepath = os.path.join(DATA_DIR, f)
            # 打包任務：(檔案路徑, 時間, p_top, p_bot)
            tasks.append((filepath, t, 200, 850))

    n_tasks = len(tasks)
    print(f"\n==================================================")
    print(f"待處理檔案數量: {n_tasks}")
    print(f"啟用進程數 (Workers): {N_WORKERS}")
    print(f"Chunk 緩衝區大小: {CHUNK_SIZE} 時步/批次")
    print(f"==================================================")

    if n_tasks == 0:
        print(">>> 所有檔案均已處理完畢，無需更新！")
        ds_out.close()
        exit()

    # 3. 啟動進程池進行串流平行計算與 Chunk 寫入
    chunk_buffer = []

    try:
        with Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
            # 使用 imap 串流回傳結果，chunksize=8 避免進程頻繁搶奪任務
            results_stream = pool.imap(_process_single_file, tasks, chunksize=8)
            
            for res in tqdm(results_stream, total=n_tasks, desc="Parallel Steering Calculation"):
                ftime, u_steer, v_steer, err = res
                
                if err is not None:
                    print(f"\n[Warning] {err}")
                    continue
                
                chunk_buffer.append((ftime, u_steer, v_steer))

                # 當緩衝區累積到 CHUNK_SIZE 時，寫入磁碟一次
                if len(chunk_buffer) >= CHUNK_SIZE:
                    flush_chunk_to_nc(ds_out, chunk_buffer)

        # 4. 寫入最後剩餘未滿 CHUNK_SIZE 的資料
        if len(chunk_buffer) > 0:
            flush_chunk_to_nc(ds_out, chunk_buffer)

    finally:
        ds_out.close()
        print("\n>>> NetCDF 輸出檔已安全關閉。")

    print(">>> 全部任務順利完成！")
