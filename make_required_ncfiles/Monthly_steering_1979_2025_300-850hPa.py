#%%
from numpy import where, linspace,nan,meshgrid, array, squeeze, where, min, pi, stack,zeros, loadtxt, arcsin, mean, concatenate, sum, nanmin, nanmean, nanmax, array
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
import matplotlib.patches as mpatches
import xarray as xr
from windspharm.xarray import VectorWind
from tqdm import trange
from numpy.fft import rfftn, irfftn, fftfreq, rfftfreq
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colors as colors
import scipy.stats as st
from scipy.stats import gaussian_kde
from matplotlib.colors import TwoSlopeNorm, LogNorm
import pywt
from scipy import signal
from scipy.stats import chi2

U_steer = np.zeros((282,201,321))
V_steer = np.zeros((282,201,321))
ncfile = nc.Dataset('/Dellwork4/cwuchiu/Monthly_UV/Monthly_UV.nc')
lat = np.array(ncfile.variables['latitude'])
lon = np.array(ncfile.variables['longitude'])
U = ncfile.variables['u']
V = ncfile.variables['v']
pres = np.array(ncfile.variables['pressure_level'])
for i in range(282):
    for j in range(pres.shape[0]-1):
        U_steer[i,:,:] += (U[i,j,160:361,400:721]+U[i,j+1,160:361,400:721])*(pres[j]-pres[j+1])/2/(850-300)
        V_steer[i,:,:] += (V[i,j,160:361,400:721]+V[i,j+1,160:361,400:721])*(pres[j]-pres[j+1])/2/(850-300)
ncfile.close()  

import numpy as np
import netCDF4 as nc
from joblib import Parallel, delayed

# ===============================
# 讀檔（一次，轉成 numpy）
# ===============================
ncfile = nc.Dataset('/Dellwork4/cwuchiu/Monthly_UV/Monthly_UV.nc')

lat  = np.array(ncfile.variables['latitude'])
lon  = np.array(ncfile.variables['longitude'])
pres = np.array(ncfile.variables['pressure_level'])

# ⭐ 關鍵：轉成 numpy array
U = np.array(ncfile.variables['u'])
V = np.array(ncfile.variables['v'])

ncfile.close()

nt = U.shape[0]

# ===============================
# 單一時間步計算
# ===============================
def compute_steering(i):
    U_i = np.zeros((201, 321))
    V_i = np.zeros((201, 321))

    for j in range(pres.shape[0] - 4):
        dp = (pres[j] - pres[j+1]) / 2 / (850 - 300)

        U_i += (U[i, j, 160:361, 400:721] +
                U[i, j+1, 160:361, 400:721]) * dp

        V_i += (V[i, j, 160:361, 400:721] +
                V[i, j+1, 160:361, 400:721]) * dp

    return i, U_i, V_i


# ===============================
# 平行計算
# ===============================
U_steer = np.zeros((nt, 201, 321))
V_steer = np.zeros((nt, 201, 321))

results = Parallel(n_jobs=8, backend="loky")(
    delayed(compute_steering)(i) for i in range(nt)
)

for i, Ui, Vi in results:
    U_steer[i] = Ui
    V_steer[i] = Vi
 

strong = nc.Dataset('/Dellwork4/cwuchiu/Monthly_steering_1979_2025_300-850hPa.nc','w',format='NETCDF4')

# Dimensions
strong.createDimension('month', 282)
strong.createDimension('latitude', 201)
strong.createDimension('longitude', 321)


strong.createVariable('longitude',np.float32, ('longitude'))
strong.createVariable('latitude',np.float32, ('latitude'))
strong.createVariable('U_steer',np.float32, ('month','latitude','longitude'))
strong.createVariable('V_steer',np.float32, ('month','latitude','longitude'))



lon = np.linspace(100,180,321)
lat = np.linspace(50,0,201)


strong.variables['longitude'][:] = lon
strong.variables['latitude'][:] = lat
strong.variables['U_steer'][:] = U_steer
strong.variables['V_steer'][:] = V_steer

strong.close()



