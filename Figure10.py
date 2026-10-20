#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:22:08 2026

@author: cwuchiu
"""

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


ncfile = nc.Dataset('/Dellwork5/cwuchiu/SH/Monthly_1979-2024_pre.nc')
hgt = array(ncfile.variables['z'][:,15,:,:])/9.81

lon = ncfile.variables['longitude']
lat = ncfile.variables['latitude']
lon,lat = array(lon),array(lat)
lon, lat = meshgrid(lon, lat)
# 500hPa = pres[15]
pres = array(ncfile.variables['pressure_level'])
ncfile.close()

hgt_all = np.zeros(46)

lon4 = np.zeros((4,721,1440))

for i in range(4):
    lon4[i] = lon
for i in range(46):
    aa = hgt[4*i:4*(i+1),:,:]
    hgt_all[i] = mean(aa[(lon4 >= 100) & (lon4 <= 180) & (lat >= 0) & (lat <= 50)])
    



tuse = ['0606','0707','0808','0909','1010','1111']
ttuse = ['06','07','08','09','10','11']
# (time,lon,lat)
gpm500_future = np.zeros((156,213,341))
gpm500_future_w = np.zeros((156,213,341))

tt=0
for i in range(2075,2101):
    for j in range(6):
        ncfile = nc.Dataset('/Dellwork5/cwuchiu/HiRAM/future_clim/mon_'+str(i)+str(i)+'_'+tuse[j]+'.nc')
        u = array(ncfile.variables['u'][5,:,:])
        v = array(ncfile.variables['v'][5,:,:])
        hgt = array(ncfile.variables['hgt'][5,:,:])
        gpm500_future[tt,:,:] = hgt
        tt+=1

gpm_future = np.zeros(26)

for i in range(26):
    gpm_future[i] = mean(gpm500_future[6*i:6*i+4,:,:])



tuse = ['0606','0707','0808','0909','1010','1111']
# (time,lon,lat)

gpm500_present = np.zeros((156,213,341))

tt=0
for i in range(1979,2005):
    for j in range(6):
        ncfile = nc.Dataset('/Dellwork5/cwuchiu/HiRAM/present_clim/mon_'+str(i)+str(i)+'_'+tuse[j]+'.nc')
        hgt = array(ncfile.variables['hgt'][5,:,:])

        gpm500_present[tt,:,:] = hgt
        tt+=1
        
gpm_present = np.zeros(26)

for i in range(26):
    gpm_present[i] = mean(gpm500_present[6*i:6*i+4,:,:])



plt.figure(figsize = (10,6))
plt.plot(np.linspace(1979,2024,46),hgt_all,'o-',color='navy', linewidth=2, markersize=6)
#plt.plot(np.linspace(2075,2100,26),gpm_future,'r')
plt.plot(np.linspace(1979,2004,26),gpm_present,'o-',color='darkgreen', linewidth=2, markersize=6)

plt.xlabel('year',fontsize=16)
plt.ylabel('Geopotential Height (gpm)',fontsize=16)
plt.xlim(1979,2004)
plt.xticks(np.linspace(1979,2004,6),fontsize=16)
plt.yticks(fontsize=16)
plt.legend(['ERA5','HiRAM_present'],fontsize=16)
plt.title('WNP Mean Geopotential Height from 1979 to 2004',fontsize=20,fontweight='bold')
plt.tight_layout()
plt.savefig('Figure10.png')
plt.show()