from sklearn.cluster import KMeans
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
import matplotlib.patches as mpatches
import xarray as xr
#from windspharm.xarray import VectorWind
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
from scipy import stats
from sklearn.preprocessing import StandardScaler

#%%
with Dataset("Tvbar_hourly.nc", "w", format="NETCDF4") as ds:
    # 建立維度
    ds.createDimension("time", None)  # 無限維度，可持續新增
    ds.createDimension("lat", 201)
    ds.createDimension("lon", 321)

    # 建立變數
    times = ds.createVariable("time", "i4", ("time",))
    Tvbar = ds.createVariable("Tvbar", "f4", ("time", "lat", "lon"), zlib=True)
    hgt500 = ds.createVariable("hgt500", "f4", ("time", "lat", "lon"), zlib=True)
    hgt1000 = ds.createVariable("hgt1000", "f4", ("time", "lat", "lon"), zlib=True)
    time_index = 0
    for year in range(1979,2025):
        year = str(year)
        for month in range(6,12):
            if month<10:
                month = '0'+str(month)
            else:
                month = str(month)
            for day in range(1,32):
                if day<10:
                    day = '0'+str(day)
                else:
                    day = str(day)
                try:
                    ncfile = nc.Dataset('/Dellwork9/cwuchiu/daily/Hourly_'+year+'_'+month+'_'+day+'.nc')
                
            
                    lon = ncfile.variables['longitude']
                    lat = ncfile.variables['latitude']
                    lon,lat = array(lon),array(lat)
                    lon, lat = meshgrid(lon, lat)
                    # 500hPa = pres[15]
                    
                    pres = array(ncfile.variables['pressure_level']) 
                    
                    
                    
                    
                    
                    nn=24
                    
                    
                    for k in range(0,nn):
                        tvbar = np.zeros((201,321))
                        Tv = (array(ncfile.variables['t'][k,:,:,:]))*(1+0.608*array(ncfile.variables['q'][k,:,:,:]))
                        
                    
                        P=pres
                        if P[0] > P[-1]:
                            P = P[::-1]
                            Tv = Tv[::-1,:,:]
                        
                        lnP = np.log(P)
                        
                        
                        
                        for i in range(P.shape[0] - 1):
                            d_lnP = lnP[i+1] - lnP[i]
                            Tv_avg = 0.5 * (Tv[i,:,:] + Tv[i+1,:,:])
                            tvbar += Tv_avg * d_lnP/np.log(1000/500)
    
                        z500 = array(ncfile.variables['z'][k,15,:,:])/9.8
                        z1000 = array(ncfile.variables['z'][k,0,:,:])/9.8
                    
                
                        # 寫入時間索引範圍
                        Tvbar[time_index:time_index+1, :, :] = tvbar
                        hgt500[time_index:time_index+1, :, :] = z500
                        hgt1000[time_index:time_index+1, :, :] = z1000
                        times[time_index:time_index+1] = np.arange(time_index, time_index+1)
                
                        time_index += 1
                    
                    
                    
                    
                    ncfile.close()  
                    
                    print(year,'/',month,'/',day)
                    
                except:
                    pass