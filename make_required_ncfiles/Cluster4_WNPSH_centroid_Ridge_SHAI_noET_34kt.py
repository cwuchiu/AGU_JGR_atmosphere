#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:42:10 2026

@author: cwuchiu
"""

import warnings
import netCDF4 as nc
import numpy as np
from scipy.ndimage import label
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import trange

# ==============================================================
# 設定參數
# ==============================================================

nc_path = '/Dellwork4/cwuchiu/Tvbar_hgt_WNPSH_2004_not_set_noET_34kt.nc'
varname = 'hgt500_adjust'

threshold = 5880

n_cluster_active = 4

MAX_JUMP_LAT = 1.0
# 相鄰經度最大允許跳躍緯度 (degree)

# 使用 8-neighbor connectivity
CONNECTIVITY = np.ones(
    (3, 3),
    dtype=int
)


# ==============================================================
# 讀取資料與建立網格
# ==============================================================

ncfile = nc.Dataset(nc_path)

hgt = ncfile.variables[varname][:]

# Shape = (nt, nlat, nlon)
nt, nlat, nlon = hgt.shape

lon = np.linspace(
    100,
    180,
    nlon
)

lat = np.linspace(
    50,
    0,
    nlat
)

ncfile.close()


# ==============================================================
# 建立 2-D 網格
# ==============================================================

lon2d, lat2d = np.meshgrid(
    lon,
    lat
)

# 緯度面積權重
lat_weight = np.cos(
    np.deg2rad(lat2d)
)


# ==============================================================
# 建立 3 個 clustering features
# ==============================================================

# --------------------------------------------------------------
# Feature 1:
# Ridge latitude
# --------------------------------------------------------------
feat_ridge_lat = np.full(
    nt,
    np.nan
)


# --------------------------------------------------------------
# Feature 2:
# Centroid longitude
#
# 以 HGT - 5880 作為強度權重
# 並使用 cos(latitude) 面積加權
# --------------------------------------------------------------
feat_centroid_lon = np.full(
    nt,
    np.nan
)


# --------------------------------------------------------------
# Feature 3:
# SHAI
#
# 與主脊線相連的 HGT > 5880 GPM 區域面積
# --------------------------------------------------------------
feat_shai = np.full(
    nt,
    np.nan
)


print("\n==============================================")
print("Calculating 3 WNPSH clustering indices")
print("==============================================")

print("1. Ridge latitude")
print("2. Centroid longitude")
print("3. SHAI")


# ==============================================================
# 逐時間計算
# ==============================================================

for t in trange(nt):

    field = hgt[t, :, :]

    # ==========================================================
    # Step 1:
    # 建立 HGT > 5880 mask
    # ==========================================================

    sh_mask = field > threshold

    if not np.any(sh_mask):
        continue


    # ==========================================================
    # Step 2:
    # 每個 longitude 找最高 HGT
    # ==========================================================

    max_val = np.max(
        field,
        axis=0
    )

    argmax_idx = np.argmax(
        field,
        axis=0
    )


    # 最高 HGT > threshold
    ridge_mask = (
        max_val > threshold
    )


    raw_ridge_lats = (
        lat[argmax_idx]
        .astype(float)
    )

    raw_ridge_lats[
        ~ridge_mask
    ] = np.nan


    valid_indices = np.where(
        ~np.isnan(raw_ridge_lats)
    )[0]


    if len(valid_indices) == 0:
        continue


    # ==========================================================
    # Step 3:
    # 找 main ridge anchor
    # ==========================================================

    anchor_idx = valid_indices[
        np.argmax(
            max_val[
                valid_indices
            ]
        )
    ]


    anchor_lat_idx = argmax_idx[
        anchor_idx
    ]


    # ==========================================================
    # Step 4:
    # 建立連續 main ridge
    # ==========================================================

    filtered_ridge_lats = np.full(
        nlon,
        np.nan
    )


    filtered_ridge_lats[
        anchor_idx
    ] = raw_ridge_lats[
        anchor_idx
    ]


    # ==========================================================
    # 向東追蹤
    # ==========================================================

    curr_lat = raw_ridge_lats[
        anchor_idx
    ]


    for j in range(
        anchor_idx + 1,
        nlon
    ):

        if np.isnan(
            raw_ridge_lats[j]
        ):
            break


        if (
            abs(
                raw_ridge_lats[j]
                - curr_lat
            )
            <= MAX_JUMP_LAT
        ):

            filtered_ridge_lats[j] = (
                raw_ridge_lats[j]
            )

            curr_lat = (
                raw_ridge_lats[j]
            )

        else:
            break


    # ==========================================================
    # 向西追蹤
    # ==========================================================

    curr_lat = raw_ridge_lats[
        anchor_idx
    ]


    for j in range(
        anchor_idx - 1,
        -1,
        -1
    ):

        if np.isnan(
            raw_ridge_lats[j]
        ):
            break


        if (
            abs(
                raw_ridge_lats[j]
                - curr_lat
            )
            <= MAX_JUMP_LAT
        ):

            filtered_ridge_lats[j] = (
                raw_ridge_lats[j]
            )

            curr_lat = (
                raw_ridge_lats[j]
            )

        else:
            break


    # ==========================================================
    # Step 5:
    # 找有效 ridge
    # ==========================================================

    ridge_lon_indices = np.where(
        ~np.isnan(
            filtered_ridge_lats
        )
    )[0]


    if len(ridge_lon_indices) == 0:
        continue


    # ==========================================================
    # Feature 1:
    # Ridge latitude
    # ==========================================================

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning
        )

        feat_ridge_lat[t] = (
            np.nanmean(
                filtered_ridge_lats
            )
        )


    # ==========================================================
    # Step 6:
    # 建立 connected WNPSH component
    #
    # 只有與 main ridge 相連的 >5880 GPM 區域
    # 才會納入 centroid 與 SHAI
    # ==========================================================

    labeled_mask, num_features = label(
        sh_mask,
        structure=CONNECTIVITY
    )


    anchor_component = labeled_mask[
        anchor_lat_idx,
        anchor_idx
    ]


    if anchor_component <= 0:
        continue


    # main WNPSH
    main_component = (
        labeled_mask
        == anchor_component
    )


    # ==========================================================
    # Feature 2:
    # Centroid longitude
    # ==========================================================

    # HGT excess above threshold
    intensity = np.maximum(
        field - threshold,
        0.0
    )


    # 只保留 main WNPSH
    intensity_main = (
        intensity
        * main_component
    )


    # cos(latitude) area weighting
    weighted_intensity = (
        intensity_main
        * lat_weight
    )


    total_intensity = np.sum(
        weighted_intensity
    )


    if total_intensity > 0:

        feat_centroid_lon[t] = (
            np.sum(
                lon2d
                * weighted_intensity
            )
            / total_intensity
        )


    # ==========================================================
    # Feature 3:
    # SHAI
    #
    # 使用 cosine latitude weighted area
    # ==========================================================

    feat_shai[t] = np.sum(
        main_component
        * lat_weight
    )


# ==============================================================
# 分離：
#
# Cluster 0 = 無/弱 WNPSH
# Cluster 1~4 = Active WNPSH
# ==============================================================

is_empty_case = (
    np.isnan(feat_ridge_lat)
    |
    np.isnan(feat_centroid_lon)
    |
    np.isnan(feat_shai)
)


final_labels = np.full(
    nt,
    -1,
    dtype=int
)


# ==============================================================
# Cluster 0
# ==============================================================

final_labels[
    is_empty_case
] = 0


n_empty = np.sum(
    is_empty_case
)


print("\n==============================================")
print("WNPSH Detection")
print("==============================================")


print(
    f"Cluster 0 (Weak/No WNPSH): "
    f"{n_empty:6d} cases "
    f"({n_empty / nt * 100:5.1f}%)"
)


# ==============================================================
# Active WNPSH
# ==============================================================

valid_idx = ~is_empty_case


n_active = np.sum(
    valid_idx
)


print(
    f"Active WNPSH: "
    f"{n_active:6d} cases "
    f"({n_active / nt * 100:5.1f}%)"
)


# ==============================================================
# 組合 3 個 clustering features
# ==============================================================

X_active = np.column_stack([

    # 1. Ridge latitude
    feat_ridge_lat[
        valid_idx
    ],

    # 2. Centroid longitude
    feat_centroid_lon[
        valid_idx
    ],

    # 3. SHAI
    feat_shai[
        valid_idx
    ]

])


# ==============================================================
# Z-score standardization
# ==============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_active
)


# ==============================================================
# K-means clustering
# ==============================================================

kmeans = KMeans(
    n_clusters=n_cluster_active,
    random_state=42,
    n_init='auto'
)


active_labels = kmeans.fit_predict(
    X_scaled
)


# Cluster 1~4
final_labels[
    valid_idx
] = active_labels + 1


# ==============================================================
# 輸出分群統計結果
# ==============================================================

total_clusters = (
    n_cluster_active + 1
)


print("\n==============================================")
print("WNPSH K-means Clustering Results")
print("==============================================")


for c in range(
    total_clusters
):

    count = np.sum(
        final_labels == c
    )


    # ==========================================================
    # Cluster 0
    # ==========================================================

    if c == 0:

        print(
            f"Cluster 0 "
            f"(Weak/No WNPSH): "
            f"N={count:6d} "
            f"({count / nt * 100:5.1f}%)"
        )

        continue


    # ==========================================================
    # Active cluster statistics
    # ==========================================================

    c_mask = (
        final_labels == c
    )


    mean_ridge = np.mean(
        feat_ridge_lat[
            c_mask
        ]
    )


    mean_centroid = np.mean(
        feat_centroid_lon[
            c_mask
        ]
    )


    mean_shai = np.mean(
        feat_shai[
            c_mask
        ]
    )


    print(
        f"Cluster {c} "
        f"(N={count:6d}, "
        f"{count / nt * 100:5.1f}%): "

        f"Ridge lat="
        f"{mean_ridge:6.2f}°N | "

        f"Centroid lon="
        f"{mean_centroid:7.2f}°E | "

        f"SHAI="
        f"{mean_shai:10.2f}"
    )

# 準備畫布
import matplotlib.pyplot as plt

# ======================
# 準備畫布與陣列 (展示活躍的 Cluster 1 ~ 4)
# ======================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

cluster_count = np.zeros(n_cluster_active, dtype=int)
cluster_mean = np.zeros((n_cluster_active, nlat, nlon))

# 統計 Cluster 0 (無副高) 的個數供總標題顯示
n_cluster0 = np.sum(final_labels == 0)

# ======================
# 繪製各群空間頻率圖
# ======================
for i, c in enumerate(range(1, total_clusters)):
    ax = axes[i]
    
    # 1. 找出屬於該群的時間索引
    idx_c = (final_labels == c)
    cnt = np.sum(idx_c)
    cluster_count[i] = cnt
    
    if cnt == 0:
        ax.text(0.5, 0.5, f"Cluster {c}\n(No Cases)", ha='center', va='center', fontsize=12)
        continue

    # 2. 計算該群的副高出現頻率 (5880 gpm 發生機率: 0.0 ~ 1.0)
    # 直接由 hgt 判定，節省記憶體
    cluster_freq = np.mean(hgt[idx_c] > threshold, axis=0)
    cluster_mean[i] = cluster_freq
    
    # 3. 取得該群「新特徵」的平均值 (西伸脊點 & 脊線緯度)
    #avg_west = np.mean(feat_west_point_lon[idx_c])
    avg_lat = np.mean(feat_ridge_lat[idx_c])
    stat_str = f"Ridge Lat: {avg_lat:.1f}°N"

    
cluster_case_number = cluster_count
strong = nc.Dataset('/Dellwork4/cwuchiu/Cluster4_WNPSH_centroid_Ridge_SHAI_noET_34kt.nc','w',format='NETCDF4')

# Dimensions
strong.createDimension('lon',321)
strong.createDimension('lat',201)
strong.createDimension('k_cluster',4)
strong.createDimension('which_cluster',59335)
#strong.createDimension('cluster0',428)




strong.createVariable('hgt_adjust_cluster',np.float32, ('k_cluster','lat','lon'))
strong.createVariable('cluster_case_number',np.float32, ('k_cluster'))
strong.createVariable('which_cluster',np.float32, ('which_cluster'))
#strong.createVariable('cluster0',np.float32, ('cluster0'))


strong.variables['hgt_adjust_cluster'][:] = cluster_mean
#strong.variables['cluster0'][:] = cluster0
strong.variables['cluster_case_number'][:] = cluster_case_number
strong.variables['which_cluster'][:] = final_labels



strong.close()
