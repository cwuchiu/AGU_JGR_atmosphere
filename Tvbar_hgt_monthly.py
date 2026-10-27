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
from scipy.ndimage import zoom

#%%
def Tvbar_fit_curve(year):
    nc_coef = nc.Dataset('/Dellwork4/cwuchiu/fit_curve_deg4_2004.nc')
    fit = array(nc_coef.variables['fit_curve'])
    Tvbar_diff = np.zeros((201,321))
    for i in range(201):
        for j in range(321):
            month = (year-2011)*6
            Tvbar_diff[i,j] = fit[i,j,month] - fit[i,j,0]
    nc_coef.close()
    return Tvbar_diff

#%%
lon = np.linspace(100,180,321)
lat = np.linspace(50,0,201)
lon, lat = array(meshgrid(lon, lat))




ncfile = nc.Dataset('/Dellwork4/cwuchiu/Tvbar_hourly.nc')

Tvbar = ncfile.variables['Tvbar']
hgt500 = ncfile.variables['hgt500']

# 1979-2024 = 24*183*46 = 202032
time4 = [1980,1984,1988,1992,1996,2000,2004,2008,2012,2016,2020,2024]
hgt500_m = np.zeros((276,201,321))
Tvbar_m = np.zeros((276,201,321))
hgt_adjust_m = np.zeros((276,201,321))



i=0
iii=0
for year in range(1979,2025):
    yyy = str(year)
    for month in range(6,12):
        aa=0
        bb=0
        if month<10:
            mmm = '0'+str(month)
        else:
            mmm = str(month)
        if month==7 or month==8 or month==10:
            for day in range(1,32):
                if day<10:
                    ddd = '0'+str(day)
                else:
                    ddd=str(day)
                for hour in range(0,24):
                    if hour<10:
                        hhh = '0'+str(hour)
                    else:
                        hhh=str(hhh)
                    
                    hgt500_m[iii]+=hgt500[i]
                    Tvbar_m[iii]+=Tvbar[i]
                    
                    i+=1
            hgt500_m[iii] = hgt500_m[iii]/(24*31)
            Tvbar_m[iii] = Tvbar_m[iii]/(24*31)
            if year>=2004:
                hgt_adjust_m[iii] = np.array(hgt500_m[iii]-Tvbar_fit_curve(year)*23)
            else:
                hgt_adjust_m[iii] = hgt500_m[iii]
            
            iii+=1
            
        else:
            for day in range(1,31):
                if day<10:
                    ddd = '0'+str(day)
                else:
                    ddd=str(day)
                for hour in range(0,24):
                    if hour<10:
                        hhh = '0'+str(hour)
                    else:
                        hhh=str(hhh)
                    hgt500_m[iii]+=hgt500[i]
                    Tvbar_m[iii]+=Tvbar[i]
                    
                    i+=1
            hgt500_m[iii] = hgt500_m[iii]/(24*30)
            Tvbar_m[iii] = Tvbar_m[iii]/(24*30)
            if year>=2004:
                hgt_adjust_m[iii] = np.array(hgt500_m[iii]-Tvbar_fit_curve(year)*23)
            else:
                hgt_adjust_m[iii] = hgt500_m[iii]
            iii+=1
    print(year)
ncfile.close()



strong = nc.Dataset('Tvbar_hgt_monthly.nc','w',format='NETCDF4')

# Dimensions
strong.createDimension('time', 276)
strong.createDimension('lon', 321)
strong.createDimension('lat', 201)

strong.createVariable('hgt500_m',np.float32, ('time','lat','lon'))
strong.createVariable('Tvbar_m',np.float32, ('time','lat','lon'))
strong.createVariable('hgt_adjust_m',np.float32, ('time','lat','lon'))

strong.variables['hgt500_m'][:] = hgt500_m
strong.variables['Tvbar_m'][:] = Tvbar_m
strong.variables['hgt_adjust_m'][:] = hgt_adjust_m
strong.close()