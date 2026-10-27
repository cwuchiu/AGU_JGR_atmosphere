#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 16:19:23 2025

@author: cwuchiu
"""


from numpy import where, linspace,nan,meshgrid, array, squeeze, where, pi, stack,zeros, loadtxt, arcsin, mean, concatenate, nanmin, nanmean, nanmax, array
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

def Tvbar_fit_curve(year,month):
    nc_coef = nc.Dataset('/Dellwork4/cwuchiu/fit_curve_deg4_2004.nc')
    fit = array(nc_coef.variables['fit_curve'])
    m = (year-2004)*6+month-6
    Tvbar_diff = fit[m] - fit[0]
    nc_coef.close()
    return Tvbar_diff


ncfile = nc.Dataset('/Dellwork4/cwuchiu/Tvbar_hourly.nc')
Tv = ncfile.variables['Tvbar']
z500 = ncfile.variables['hgt500']



with Dataset("Tvbar_hgt_adjust_hourly_mean_2004.nc", "w", format="NETCDF4") as ds:
    # 建立維度
    ds.createDimension("time", None)  # 無限維度，可持續新增
    ds.createDimension("lat", 201)
    ds.createDimension("lon", 321)

    # 建立變數
    times = ds.createVariable("time", "i4", ("time",))
    Tvbar = ds.createVariable("Tvbar", "f4", ("time", "lat", "lon"), zlib=True)
    hgt500 = ds.createVariable("hgt500", "f4", ("time", "lat", "lon"), zlib=True)
    time_index = 0

            
    for i in trange(202032):
                
        Tvbar[time_index:time_index+1, :, :] = Tv[time_index]
        if i<109800:
            hgt500[time_index:time_index+1, :, :] = z500[time_index]
        else:
            if i%4392<720:
                hgt500[time_index:time_index+1, :, :] = z500[time_index]-Tvbar_fit_curve(i//4392+1979,6)*20.3
            elif (i%4392>=720) and (i%4392<1464):
                hgt500[time_index:time_index+1, :, :] = z500[time_index]-Tvbar_fit_curve(i//4392+1979,7)*20.3
            elif (i%4392>=1464) and (i%4392<2208):
                hgt500[time_index:time_index+1, :, :] = z500[time_index]-Tvbar_fit_curve(i//4392+1979,8)*20.3
            elif (i%4392>=2208) and (i%4392<2928):
                hgt500[time_index:time_index+1, :, :] = z500[time_index]-Tvbar_fit_curve(i//4392+1979,9)*20.3
            elif (i%4392>=2928) and (i%4392<3672):
                hgt500[time_index:time_index+1, :, :] = z500[time_index]-Tvbar_fit_curve(i//4392+1979,10)*20.3
            else:
                hgt500[time_index:time_index+1, :, :] = z500[time_index]-Tvbar_fit_curve(i//4392+1979,11)*20.3
        times[time_index:time_index+1] = np.arange(time_index, time_index+1)
            
        time_index += 1
                    
   
                    
ncfile.close()
