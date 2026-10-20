#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 01:38:28 2026

@author: cwuchiu
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import t as student_t
import netCDF4 as nc

# ============================================================
# 1. 讀取資料
# ============================================================

file_path = '/Dellwork4/cwuchiu/Tvbar_hgt_monthly.nc'

with nc.Dataset(file_path) as ncfile:
    Tvbar = np.ma.filled(
        ncfile.variables['Tvbar_m'][:],
        np.nan
    ).astype(float)

# 預期維度：(time, lat, lon) = (276, 201, 321)
print('Tvbar shape:', Tvbar.shape)

if Tvbar.shape != (276, 201, 321):
    raise ValueError(
        '資料維度與預期不符，請確認 Tvbar_m 的維度排列。'
    )

# ============================================================
# 2. 建立經緯度
# ============================================================

lon = np.linspace(100, 180, 321)
lat = np.linspace(50, 0, 201)

# ============================================================
# 3. 計算年度 JJASON 面積加權平均虛溫
# ============================================================

# 假設 276 筆資料依序為：
# 1979 年 6–11 月、1980 年 6–11 月，……，2024 年 6–11 月
years = np.arange(1979, 2025)
n_years = len(years)

# 每年 6–11 月的天數
days_in_month = np.array([30, 31, 31, 30, 31, 30])

# 轉換為 (year, month, lat, lon)
Tvbar_4d = Tvbar.reshape(n_years, 6, 201, 321)

# 經緯度網格的面積近似正比於 cos(latitude)
lat_weights = np.cos(np.deg2rad(lat))[None, None, :, None]

# 同時考慮各月份的天數
weights = lat_weights * days_in_month[None, :, None, None]

# 對有效資料計算加權平均
valid = np.isfinite(Tvbar_4d)

numerator = np.sum(
    np.where(valid, Tvbar_4d * weights, 0.0),
    axis=(1, 2, 3)
)

denominator = np.sum(
    np.where(valid, weights, 0.0),
    axis=(1, 2, 3)
)

annual_Tvbar = np.divide(
    numerator,
    denominator,
    out=np.full(n_years, np.nan),
    where=denominator > 0
)

if not np.all(np.isfinite(annual_Tvbar)):
    raise ValueError('年度 JJASON 平均序列含有缺值。')

print('Annual sample size:', len(annual_Tvbar))

# ============================================================
# 4. Pettitt test
# ============================================================

def pettitt_test(data):
    """Pettitt 單一變點檢定，回傳分割位置、K 與近似 p 值。"""

    data = np.asarray(data, dtype=float)

    if data.ndim != 1 or len(data) < 3:
        raise ValueError('輸入資料必須是一維，且至少有 3 筆。')

    if not np.all(np.isfinite(data)):
        raise ValueError('輸入序列含有缺值。')

    n = len(data)

    # 各時間點與所有資料比較
    k_array = np.array([
        np.sum(np.sign(data[t] - data))
        for t in range(n)
    ])

    # 累積統計量；最後一個分割位置不納入搜尋
    U = np.cumsum(k_array)[:-1]

    idx = int(np.argmax(np.abs(U)))
    K = float(np.abs(U[idx]))

    # Pettitt test 常用的近似 p-value
    p_value = min(
        1.0,
        2.0 * np.exp(-6.0 * K**2 / (n**3 + n**2))
    )

    return idx, K, p_value


idx, K, p_value = pettitt_test(annual_Tvbar)


r1 = np.corrcoef(
    annual_Tvbar[:-1],
    annual_Tvbar[1:]
)[0, 1]


# 前期：1979–2003
# 後期：2004–2024
POST_START_YEAR = 2004
HAC_MAXLAGS = 2

out_dir = '/Dellwork5/cwuchiu/plot'
os.makedirs(out_dir, exist_ok=True)

pdf_path = os.path.join(
    out_dir, 'FigureS2_Tvbar_trend_change_2004.pdf'
)
tif_path = os.path.join(
    out_dir, 'FigureS2_Tvbar_trend_change_2004.tif'
)

x = np.asarray(years, dtype=float)
y = np.asarray(annual_Tvbar, dtype=float)

if len(x) != len(y):
    raise ValueError('years 與 annual_Tvbar 長度不一致。')

if not np.all(np.isfinite(y)):
    raise ValueError('annual_Tvbar 含有缺值。')

if not np.all(np.diff(x) == 1):
    raise ValueError('年份必須連續且間隔一年。')

# 全時段平均虛溫距平，用於繪圖
y_anom = y - np.mean(y)

# ============================================================
# 2. 建立分段線性迴歸
# ============================================================

# 以 2004 年為時間原點
t = x - 2004

# 2004 年開始的後期指標
post = (x >= POST_START_YEAR).astype(float)

# 後期經過年數；2004 年為 0
post_time = np.where(
    x >= POST_START_YEAR,
    x - POST_START_YEAR,
    0.0
)

# 模型：
# y = b0 + b1*t + b2*post + b3*post_time + error
#
# b1      = 1979–2003 年的趨勢斜率
# b1 + b3 = 2004–2024 年的趨勢斜率
# b2      = 2004 年相對於前期趨勢外推值的位置差異
# b3      = 前後期斜率差異

X = np.column_stack([
    np.ones(len(x)),
    t,
    post,
    post_time
])

fit = sm.OLS(y_anom, X).fit(
    cov_type='HAC',
    cov_kwds={'maxlags': HAC_MAXLAGS},
    use_t=True
)

b = np.asarray(fit.params)

# ============================================================
# 3. 計算趨勢、95% CI 與 p-value
# ============================================================

def contrast_summary(fit, contrast, scale=1.0):
    """計算線性組合的估計值、95% CI 及 p-value。"""

    result = fit.t_test(
        np.asarray(contrast, dtype=float)
    )

    estimate = float(
        np.asarray(result.effect).reshape(-1)[0]
    )

    ci = np.asarray(
        result.conf_int()
    ).reshape(-1, 2)[0]

    p_value = float(
        np.asarray(result.pvalue).item()
    )

    return (
        estimate * scale,
        ci[0] * scale,
        ci[1] * scale,
        p_value
    )


# 前期趨勢
pre_stats = contrast_summary(
    fit, [0, 1, 0, 0], scale=10
)

# 後期趨勢
post_stats = contrast_summary(
    fit, [0, 1, 0, 1], scale=10
)

# 前後期斜率差異
diff_stats = contrast_summary(
    fit, [0, 0, 0, 1], scale=10
)

# 位置跳躍
level_est = float(b[2])
level_ci = np.asarray(fit.conf_int())[2]
level_p = float(fit.pvalues[2])

print('\n========== Figure S2: Segmented Regression ==========')
print('Pre-period: 1979–2003')
print('Post-period: 2004–2024')
print(f'HAC maxlags = {HAC_MAXLAGS}')

print('\n[Pre-period trend: 1979–2003]')
print(f'Trend = {pre_stats[0]:.5f} K/decade')
print(f'95% CI = [{pre_stats[1]:.5f}, {pre_stats[2]:.5f}]')
print(f'p-value = {pre_stats[3]:.6g}')

print('\n[Post-period trend: 2004–2024]')
print(f'Trend = {post_stats[0]:.5f} K/decade')
print(f'95% CI = [{post_stats[1]:.5f}, {post_stats[2]:.5f}]')
print(f'p-value = {post_stats[3]:.6g}')

print('\n[Difference in trends]')
print(f'Slope difference = {diff_stats[0]:.5f} K/decade')
print(f'95% CI = [{diff_stats[1]:.5f}, {diff_stats[2]:.5f}]')
print(f'p-value = {diff_stats[3]:.6g}')

print('\n[Level shift in 2004]')
print(f'Level shift = {level_est:.5f} K')
print(f'95% CI = [{level_ci[0]:.5f}, {level_ci[1]:.5f}]')
print(f'p-value = {level_p:.6g}')

# ============================================================
# 4. 計算擬合值與 95% 信賴區間
# ============================================================

y_fit = X @ b

cov = np.asarray(fit.cov_params())

fit_var = np.einsum(
    'ij,jk,ik->i',
    X, cov, X
)

fit_se = np.sqrt(
    np.maximum(fit_var, 0.0)
)

critical_value = student_t.ppf(
    0.975, df=fit.df_resid
)

lower = y_fit - critical_value * fit_se
upper = y_fit + critical_value * fit_se

pre_mask = x <= 2003
post_mask = x >= 2004

# ============================================================
# 5. 繪製 Figure S2
# ============================================================

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

fig, ax = plt.subplots(
    figsize=(6.8, 3.8),
    constrained_layout=True
)

# 前期 95% 信賴區間
ax.fill_between(
    x[pre_mask],
    lower[pre_mask],
    upper[pre_mask],
    color='#0072B2',
    alpha=0.16,
    linewidth=0
)

# 後期 95% 信賴區間
ax.fill_between(
    x[post_mask],
    lower[post_mask],
    upper[post_mask],
    color='#D55E00',
    alpha=0.16,
    linewidth=0
)

# 前期迴歸線
ax.plot(
    x[pre_mask],
    y_fit[pre_mask],
    color='#0072B2',
    linewidth=2.0,
    label='Linear fit (1979–2003)'
)

# 後期迴歸線
ax.plot(
    x[post_mask],
    y_fit[post_mask],
    color='#D55E00',
    linewidth=2.0,
    label='Linear fit (2004–2024)'
)

# 年度觀測值
ax.scatter(
    x,
    y_anom,
    s=25,
    marker='o',
    facecolor='0.25',
    edgecolor='white',
    linewidth=0.45,
    zorder=4,
    label='Annual JJASON mean Tv'
)

# 2003 與 2004 年之間的分界
ax.axvline(
    2003.5,
    color='0.35',
    linestyle='--',
    linewidth=1.1,
    alpha=0.6
)

# 零距平參考線
ax.axhline(
    0,
    color='0.65',
    linewidth=0.7,
    linestyle=':'
)

ax.set_xlim(1978, 2025)
ax.set_xticks(np.arange(1979, 2025, 5))

ax.set_xlabel('Year')
ax.set_ylabel(
    'WNP JJASON mean Tv (K)'
)

ax.legend(
    loc='upper left',
    frameon=False,
    ncol=2,
    handlelength=2.2
)

ax.tick_params(
    axis='both',
    direction='out',
    length=3,
    width=0.8
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 儲存圖檔
fig.savefig('FigureS2.png',dpi=400,
    bbox_inches='tight'
)


plt.show()
plt.close(fig)

print('\nSaved:')
print(pdf_path)
print(tif_path)
