#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:59:04 2026

@author: cwuchiu
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import xarray as xr

# 1. 讀取資料
file_path = "/Dellwork5/cwuchiu/SH/Monthly_1979-2024_pre.nc"
ds = xr.open_dataset(file_path)

g = 9.80665  # 重力加速度，轉為 gpm


# 2. 定義氣壓層選取函數 (優先用氣壓值 sel，若無座標則用 index)
def get_level_data(ds, level_val, level_idx):
    if "pressure_level" in ds.coords and level_val in ds["pressure_level"].values:
        return ds["z"].sel(pressure_level=level_val) / g
    else:
        return ds["z"].isel(pressure_level=level_idx) / g


# 提取 1000 hPa (第 0 層) 與 500 hPa (第 15 層)
z_1000 = get_level_data(ds, 1000, 0)
z_500 = get_level_data(ds, 500, 15)

# 3. 緯度面積加權 (cos(lat))
weights = np.cos(np.deg2rad(ds["latitude"]))
weights.name = "weights"


# 4. 計算領域平均與 6-11 月年平均時間序列
def process_domain_annual_mean(da):
    # 篩選 6~11 月
    da_season = da.sel(valid_time=da["valid_time.month"].isin(range(6, 12)))
    # 領域加權平均
    da_domain = da_season.weighted(weights).mean(dim=["latitude", "longitude"])
    # 每年 6~11 月平均
    da_annual = da_domain.groupby("valid_time.year").mean(dim="valid_time")
    return da_annual


z1000_annual = process_domain_annual_mean(z_1000)
z500_annual = process_domain_annual_mean(z_500)

years = z1000_annual["year"].values
delta_years = years[-1] - years[0]  # 總年數跨度 (45 年)


# 5. 線性趨勢統計函數
def analyze_trend(years, vals):
    res = stats.linregress(years, vals)
    slope = res.slope  # gpm / year
    trend_decade = slope * 10  # gpm / decade
    total_increase = slope * delta_years  # 整個期間線性累積總增加幅度 (gpm)
    return {
        "slope": slope,
        "intercept": res.intercept,
        "trend_decade": trend_decade,
        "total_increase": total_increase,
        "p_val": res.pvalue,
        "r2": res.rvalue**2,
    }


stats_1000 = analyze_trend(years, z1000_annual.values)
stats_500 = analyze_trend(years, z500_annual.values)

# 印出比較數據
print("=" * 60)
print(f"分析時段: {years[0]} - {years[-1]} (6-11月) 領域平均增加幅度比較")
print("=" * 60)
for name, s in [("1000 hPa", stats_1000), ("500 hPa", stats_500)]:
    print(f"【{name}】")
    print(f"  - 累積總增加幅度 (Total Increase): {s['total_increase']:+.2f} gpm")
    print(f"  - 每十年趨勢 (Trend/Decade)      : {s['trend_decade']:+.2f} gpm/decade")
    print(f"  - 顯著性 (p-value)                : {s['p_val']:.3e} (R² = {s['r2']:.3f})")
print("=" * 60)
#%%
# 6. 繪製 1x2 子圖
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=400)

# 子圖設定參數
configs = [
    {
        "ax": axes[0],
        "data": z1000_annual.values,
        "stat": stats_1000,
        "title": "(a) 1000 hPa Geopotential Height",
        "color": "#1f77b4",
    },
    {
        "ax": axes[1],
        "data": z500_annual.values,
        "stat": stats_500,
        "title": "(b) 500 hPa Geopotential Height",
        "color": "#2ca02c",
    },
]

for cfg in configs:
    ax = cfg["ax"]
    y_vals = cfg["data"]
    s = cfg["stat"]
    c = cfg["color"]

    # 繪製每年實際觀測點與折線
    ax.plot(
        years,
        y_vals,
        marker="o",
        markersize=4.5,
        color=c,
        linewidth=1.5,
        alpha=0.85,
        label="Annual Mean (Jun-Nov)",
    )

    # 繪製回歸線
    trend_line = s["intercept"] + s["slope"] * years
    ax.plot(
        years,
        trend_line,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=(
            f"Trend: {s['trend_decade']:+.2f} gpm/decade\n"
            f"Total ΔZ: {s['total_increase']:+.2f} gpm\n"
            f"(p = {s['p_val']:.2e}, R² = {s['r2']:.2f})"
        ),
    )

    ax.set_title(cfg["title"], fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Geopotential Height (gpm)", fontsize=12)
    ax.set_xticks(np.linspace(1979,2024,10))
    ax.set_xlim(1979,2024)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax.set_xlim(1979,2024)


plt.tight_layout()
plt.savefig("z1000_z500_trend_comparison.png", dpi=300, bbox_inches="tight")
print("\n圖檔已儲存為: z1000_z500_trend_comparison.png")
plt.show()