#%%
from numpy import where, linspace,nan,meshgrid, array, squeeze, where, pi, stack,zeros, loadtxt, arcsin, concatenate, nanmin, nanmean, nanmax, array
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import netCDF4 as nc
from wrf import (getvar,interplevel)
import pandas as pd
import wrf
import numpy as np
from scipy.integrate import dblquad, quad
import matplotlib.lines as lines
from scipy.stats import linregress
from scipy.sparse import diags, kron, identity, csr_matrix
from scipy.sparse.linalg import spsolve
from netCDF4 import Dataset
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

ncfile = nc.Dataset('/Dellwork4/cwuchiu/Tvbar_hgt_monthly.nc')

lon = np.linspace(100,180,321)
lat = np.linspace(50,0,201)
lon, lat = array(meshgrid(lon, lat))

year_num=21
Tvbar = np.array(ncfile.variables['Tvbar_m'])[276-year_num*6:276]
#ncfile.close()



Tvbar_reshaped = Tvbar.reshape((year_num,6,201,321))
# 計算 10 年的平均季節循環 (4320, 201, 321)
Tvbar_SeasonalCycle = np.nanmean(Tvbar_reshaped, axis=0)

# 計算 anomaly：原始資料 - 平均季節循環（broadcasting）
anomaly = (Tvbar_reshaped - Tvbar_SeasonalCycle).reshape((year_num*6,201,321))  # shape = (10, 4320, 201, 321)

Tv = Tvbar


#================================


time = np.linspace(2024-year_num+1,2024,year_num*6)
fit_curve = np.zeros((year_num*6))
anomaly_s = np.nanmean(anomaly,axis=(1,2))
def calculate_aic(y, y_pred, k):
    n = len(y)
    resid = y - y_pred
    sse = np.sum(resid**2)
    aic = n * np.log(sse / n) + 2 * k
    return aic


# 比較不同多項式階數的擬合
max_degree = 7
aic_values = []

for deg in range(1, max_degree + 1):
    coeffs = np.polyfit(time, anomaly_s, deg)
    y_pred = np.polyval(coeffs, time)
    k = deg + 1  # 參數數量（含截距）
    aic = calculate_aic(anomaly_s, y_pred, k)
    aic_values.append(aic)

    # 找出最佳模型
    best_deg = np.argmin(aic_values) + 1     #4 
    best_coeffs = np.polyfit(time, anomaly_s, best_deg)
    best_fit = np.polyval(best_coeffs, time)
    
    fit_curve[:] = best_fit
    print(f'deg={best_deg}')



#================================
strong = nc.Dataset('fit_curve_deg4_2004.nc','w',format='NETCDF4')

# Dimensions
strong.createDimension('latitude', 201)
strong.createDimension('longitude', 321)
strong.createDimension('time', 126)

strong.createVariable('longitude',np.float32, 'longitude')
strong.createVariable('latitude',np.float32, 'latitude')
strong.createVariable('fit_curve',np.float32, ('time'))


lon = np.linspace(100,180,321)
lat = np.linspace(50,0,201)
strong.variables['longitude'][:] = lon[:]
strong.variables['latitude'][:] = lat[:]
strong.variables['fit_curve'][:] = fit_curve


strong.close()
