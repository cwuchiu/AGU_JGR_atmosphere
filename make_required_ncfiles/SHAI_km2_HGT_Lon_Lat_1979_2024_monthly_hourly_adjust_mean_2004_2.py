#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:40:15 2026

@author: cwuchiu
"""
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

def Tvbar_fit_curve(year,month):
    nc_coef = nc.Dataset('/Dellwork4/cwuchiu/fit_curve_deg4_2004.nc')
    fit = array(nc_coef.variables['fit_curve'])
    m = (year-2004)*6+month-6
    Tvbar_diff = fit[m] - fit[0]
    nc_coef.close()
    return Tvbar_diff

ncfile = nc.Dataset('/Dellwork4/cwuchiu/Tvbar_hourly.nc')

hgt500 = ncfile.variables['hgt500']
Tvbar = ncfile.variables['Tvbar']

lon = np.linspace(100,180,321)
lat = np.linspace(50,0,201)

lon,lat = np.meshgrid(lon,lat)

lon_hourly = np.zeros(202032)
lat_hourly = np.zeros(202032)
weighted_lon_hourly = np.zeros(202032)
weighted_lat_hourly = np.zeros(202032)
west_hourly = np.zeros(202032)
east_hourly = np.zeros(202032)
north_hourly = np.zeros(202032)
south_hourly = np.zeros(202032)

SHAI_hourly = np.zeros(202032)
SHAI_monthly = np.zeros(276)
lon_monthly = np.zeros(276)
lat_monthly = np.zeros(276)
weighted_lon_monthly = np.zeros(276)
weighted_lat_monthly = np.zeros(276)
east_monthly = np.zeros(276)
west_monthly = np.zeros(276)
north_monthly = np.zeros(276)
south_monthly = np.zeros(276)
HGT_monthly = np.zeros((276,201,321))

i=0
iii=0
kkk=0
for year in trange(1979,2025):
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
                    if year>2004:
                        hgt_adjust = array(hgt500[i]-Tvbar_fit_curve(year,month)*20.3)
                        aa+= hgt_adjust.copy()
                        
                        #lat_hourly[i] = np.nanmean(lat[hgt_adjust>5880]*anom)/np.nanmean(anom)
                        #lon_hourly[i] = np.nanmean(lon[hgt_adjust>5880]*anom)/np.nanmean(anom)
                        mask = (hgt_adjust > 5880) & (lon >= 100) & (lat <= 50) & (lat>=0)
                        anom = hgt_adjust[mask]-5880
                        cos_lat = np.cos(np.radians(lat[mask]))
                        weights = anom* cos_lat

                        if len(anom)==0:
                            lat_hourly[i] = np.nan
                            lon_hourly[i] = np.nan
                            weighted_lat_hourly[i] = np.nan
                            weighted_lon_hourly[i] = np.nan
                            west_hourly[i] = np.nan
                            east_hourly[i] = np.nan
                            north_hourly[i] = np.nan
                            south_hourly[i] = np.nan
                        else:
                            weighted_lat_hourly[i] = np.average(lat[mask], weights=weights)
                            weighted_lon_hourly[i] = np.average(lon[mask], weights=weights)
                            lat_hourly[i] = np.round(np.nanmean(lat[mask]),2)
                            lon_hourly[i] = np.round(np.nanmean(lon[mask]),2)
                            west_hourly[i] = np.nanmin(lon[mask])
                            east_hourly[i] = np.nanmax(lon[mask])
                            north_hourly[i] = np.nanmax(lat[mask])
                            south_hourly[i] = np.nanmin(lat[mask])
                    else:
                        hgt_adjust = hgt500[i].copy()
                        aa+= hgt_adjust.copy()
                    
                        mask = (hgt_adjust > 5880) & (lon >= 100) & (lat <= 50) & (lat>=0)
                        anom = hgt_adjust[mask]-5880
                        cos_lat = np.cos(np.radians(lat[mask]))
                        weights = anom* cos_lat

                        if len(anom)==0:
                            lat_hourly[i] = np.nan
                            lon_hourly[i] = np.nan
                            weighted_lat_hourly[i] = np.nan
                            weighted_lon_hourly[i] = np.nan
                            west_hourly[i] = np.nan
                            east_hourly[i] = np.nan
                            north_hourly[i] = np.nan
                            south_hourly[i] = np.nan
                        else:
                            weighted_lat_hourly[i] = np.average(lat[mask], weights=weights)
                            weighted_lon_hourly[i] = np.average(lon[mask], weights=weights)
                            lat_hourly[i] = np.round(np.nanmean(lat[mask]),2)
                            lon_hourly[i] = np.round(np.nanmean(lon[mask]),2)
                            west_hourly[i] = np.nanmin(lon[mask])
                            east_hourly[i] = np.nanmax(lon[mask])
                            north_hourly[i] = np.nanmax(lat[mask])
                            south_hourly[i] = np.nanmin(lat[mask])
                        
                    SHAI_hourly[i] = np.sum(hgt_adjust>5880)
                        
                    i+=1
            aa = aa/(24*31)
            
            SHAI_monthly[iii] = np.nanmean(SHAI_hourly[kkk:kkk+24*31])
            lat_monthly[iii] = np.nanmean(lat_hourly[kkk:kkk+24*31])
            lon_monthly[iii] = np.nanmean(lon_hourly[kkk:kkk+24*31])
            weighted_lat_monthly[iii] = np.nanmean(weighted_lat_hourly[kkk:kkk+24*31])
            weighted_lon_monthly[iii] = np.nanmean(weighted_lon_hourly[kkk:kkk+24*31])
            east_monthly[iii] = np.nanmean(east_hourly[kkk:kkk+24*31])
            west_monthly[iii] = np.nanmean(west_hourly[kkk:kkk+24*31])
            north_monthly[iii] = np.nanmean(north_hourly[kkk:kkk+24*31])
            south_monthly[iii] = np.nanmean(south_hourly[kkk:kkk+24*31])
            HGT_monthly[iii] = aa


            
            iii+=1
            kkk+=24*31
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
                    if year>2004:
                        hgt_adjust = array(hgt500[i]-Tvbar_fit_curve(year,month)*20.3)
                        aa+= hgt_adjust.copy()
                        mask = (hgt_adjust > 5880) & (lon >= 100) & (lat <= 50) & (lat>=0)
                        anom = hgt_adjust[mask]-5880
                        cos_lat = np.cos(np.radians(lat[mask]))
                        weights = anom* cos_lat
                        if len(anom)==0:
                            lat_hourly[i] = np.nan
                            lon_hourly[i] = np.nan
                            weighted_lat_hourly[i] = np.nan
                            weighted_lon_hourly[i] = np.nan
                            west_hourly[i] = np.nan
                            east_hourly[i] = np.nan
                            north_hourly[i] = np.nan
                            south_hourly[i] = np.nan
                        else:
                            weighted_lat_hourly[i] = np.average(lat[mask], weights=weights)
                            weighted_lon_hourly[i] = np.average(lon[mask], weights=weights)
                            lat_hourly[i] = np.round(np.nanmean(lat[mask]),2)
                            lon_hourly[i] = np.round(np.nanmean(lon[mask]),2)
                            west_hourly[i] = np.nanmin(lon[mask])
                            east_hourly[i] = np.nanmax(lon[mask])
                            north_hourly[i] = np.nanmax(lat[mask])
                            south_hourly[i] = np.nanmin(lat[mask])
                    else:
                        hgt_adjust = hgt500[i].copy()
                        aa+= hgt_adjust.copy()
                    
                        mask = (hgt_adjust > 5880) & (lon >= 100) & (lat <= 50) & (lat>=0)
                        anom = hgt_adjust[mask]-5880
                        cos_lat = np.cos(np.radians(lat[mask]))
                        weights = anom* cos_lat
                        if len(anom)==0:
                            lat_hourly[i] = np.nan
                            lon_hourly[i] = np.nan
                            weighted_lat_hourly[i] = np.nan
                            weighted_lon_hourly[i] = np.nan
                            west_hourly[i] = np.nan
                            east_hourly[i] = np.nan
                            north_hourly[i] = np.nan
                            south_hourly[i] = np.nan
                        else:
                            weighted_lat_hourly[i] = np.average(lat[mask], weights=weights)
                            weighted_lon_hourly[i] = np.average(lon[mask], weights=weights)
                            lat_hourly[i] = np.round(np.nanmean(lat[mask]),2)
                            lon_hourly[i] = np.round(np.nanmean(lon[mask]),2)
                            west_hourly[i] = np.nanmin(lon[mask])
                            east_hourly[i] = np.nanmax(lon[mask])
                            north_hourly[i] = np.nanmax(lat[mask])
                            south_hourly[i] = np.nanmin(lat[mask])
                        
                    SHAI_hourly[i] = np.sum(hgt_adjust>5880)
                    i+=1

            aa = aa/(24*30)
            
            SHAI_monthly[iii] = np.nanmean(SHAI_hourly[kkk:kkk+24*30])
            lat_monthly[iii] = np.nanmean(lat_hourly[kkk:kkk+24*30])
            lon_monthly[iii] = np.nanmean(lon_hourly[kkk:kkk+24*30])
            weighted_lat_monthly[iii] = np.nanmean(weighted_lat_hourly[kkk:kkk+24*30])
            weighted_lon_monthly[iii] = np.nanmean(weighted_lon_hourly[kkk:kkk+24*30])
            HGT_monthly[iii] = aa
            east_monthly[iii] = np.nanmean(east_hourly[kkk:kkk+24*30])
            west_monthly[iii] = np.nanmean(west_hourly[kkk:kkk+24*30])
            north_monthly[iii] = np.nanmean(north_hourly[kkk:kkk+24*30])
            south_monthly[iii] = np.nanmean(south_hourly[kkk:kkk+24*30])
            
            
            
            iii+=1
            kkk+=24*30
    print('year = ',year)

ncfile.close()



strong = nc.Dataset('SHAI_km2_HGT_Lon_Lat_1979_2024_monthly_hourly_adjust_mean_2004_2.nc','w',format='NETCDF4')

# Dimensions
strong.createDimension('time', 202032)
strong.createDimension('month', 276)
strong.createDimension('latitude', 201)
strong.createDimension('longitude', 321)


strong.createVariable('longitude',np.float32, ('longitude'))
strong.createVariable('latitude',np.float32, ('latitude'))

strong.createVariable('SHAI_hourly',np.float32, ('time'))
strong.createVariable('SHAI_monthly',np.float32, ('month'))

strong.createVariable('Lat_hourly',np.float32, ('time'))
strong.createVariable('Lat_monthly',np.float32, ('month'))

strong.createVariable('Lon_hourly',np.float32, ('time'))
strong.createVariable('Lon_monthly',np.float32, ('month'))

strong.createVariable('weighted_lat_hourly',np.float32, ('time'))
strong.createVariable('weighted_lat_monthly',np.float32, ('month'))

strong.createVariable('weighted_lon_hourly',np.float32, ('time'))
strong.createVariable('weighted_lon_monthly',np.float32, ('month'))

strong.createVariable('east_hourly',np.float32, ('time'))
strong.createVariable('east_monthly',np.float32, ('month'))

strong.createVariable('west_hourly',np.float32, ('time'))
strong.createVariable('west_monthly',np.float32, ('month'))

strong.createVariable('north_hourly',np.float32, ('time'))
strong.createVariable('north_monthly',np.float32, ('month'))

strong.createVariable('south_hourly',np.float32, ('time'))
strong.createVariable('south_monthly',np.float32, ('month'))

strong.createVariable('HGT_monthly',np.float32, ('month','latitude','longitude'))



lon = np.linspace(100,180,321)
lat = np.linspace(50,0,201)

strong.variables['SHAI_hourly'][:] = SHAI_hourly*111*111*np.cos(30*pi/180)
strong.variables['SHAI_monthly'][:] = SHAI_monthly*111*111*np.cos(30*pi/180)
strong.variables['longitude'][:] = lon
strong.variables['latitude'][:] = lat
strong.variables['Lon_hourly'][:] = lon_hourly
strong.variables['Lon_monthly'][:] = lon_monthly
strong.variables['Lat_hourly'][:] = lat_hourly
strong.variables['Lat_monthly'][:] = lat_monthly
strong.variables['weighted_lon_hourly'][:] = weighted_lon_hourly
strong.variables['weighted_lon_monthly'][:] = weighted_lon_monthly
strong.variables['weighted_lat_hourly'][:] = weighted_lat_hourly
strong.variables['weighted_lat_monthly'][:] = weighted_lat_monthly
strong.variables['HGT_monthly'][:] = HGT_monthly

strong.variables['west_hourly'][:] = west_hourly
strong.variables['west_monthly'][:] = west_monthly
strong.variables['east_hourly'][:] = east_hourly
strong.variables['east_monthly'][:] = east_monthly
strong.variables['north_hourly'][:] = north_hourly
strong.variables['north_monthly'][:] = north_monthly
strong.variables['south_hourly'][:] = south_hourly
strong.variables['south_monthly'][:] = south_monthly


strong.close()
