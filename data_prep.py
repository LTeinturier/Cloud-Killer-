# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:45:37 2020

@author: lucas
"""

import numpy as np
import weather_creator as wc
import Model_Init as M_init
import netCDF4
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#import init as var
#import Utilities as util

def fake_data(nslices,w, Days, Acloud, ndata, timespan,phispan):
    hPerDay = int((w/(2*np.pi))**(-1))
#    surfDict = M_init.initialPlanet(nslices,plot = False)
#    surf = np.fromiter(surfDict.values(), dtype = float)
    surf = 0.5*np.random.rand(nslices)
    print("The planet surface albedo is {}".format(surf))
    finalTime    = []
    apparentTime = []
    maxvaldA = 0.9#1-np.max(surf)
    Delta_A = maxvaldA*np.random.rand(Days,nslices)
    ts_effalb = M_init.effectiveAlbedo(surf,Delta_A)
    for i in range(1,Days+1):
        effective = ts_effalb[i-1,:]
        time,apparent = M_init.Forwardmodel(effective,w,time_days=timespan,
                                            long_frac=phispan,phi_obs_0=0,
                                            n = 1000,plot= False,alb = True)
        finalTime.append(time+hPerDay*(i-1))
        apparentTime.append(apparent)
    finalTime= np.asarray(finalTime).flatten()
    apparentTime = np.asarray(apparentTime).flatten()
    t,a = wc.extractN(finalTime,apparentTime,ndata,Days) #extraction a la résolution temporelle EPIC
    a = np.asarray(a)
    t = np.asarray(t)
    a_err = 0.02 * a # consistent with the treatment done on EPIC data
    gaussian_noise = np.random.normal(0, 0.02*np.mean(a),len(a))#[np.random.normal(loc=0,scale=0.02*a[i]) for i in range(len(a))]
    a += gaussian_noise
    return t,a,a_err,Delta_A,surf, ts_effalb




def EPIC_data(day, plot=False):
    """
    Input: a date (int) after 13 June 2015 00:00:00, a boolean indicating 
    whether or not to plot the data
    Output: time, longitude (deg), apparent albedo, error on apparent albedo, 
    a bool indicating if dataset contains NaNs
    """
        # RETRIEVE DATA
    data = netCDF4.Dataset("dscovr_single_light_timeseries.nc") # netCDF4 module used here
    data.dimensions.keys()
    radiance = data.variables["normalized"][:] # lightcurves for 10 wavelengths

    # Constants used throughout
    SOLAR_IRRAD_780 = 1.190 # Units: W m^-2 nm^-1

    # Constant arrays used throughout
    RAD_780 = radiance[9] # lightcurves for 780 nm
    #time in seconds since June 13, 2015 00:00:00 UTC
    TIME_SECS = radiance[10]
    #time in days since June 13, 2015  00:00:00 UTC
    TIME_DAYS = TIME_SECS/86148.0 #86148 = 23.93h

    #longitude at SOP/SSP: convert UTC at SOP/SSP to longitude 
    #longitude is 2pi at t=0 and decreases with time
    SOP_LONGITUDE = [(2*np.pi-(t%86148.0)*(2*np.pi/86148.0))%(2*np.pi) for t in TIME_SECS]
    #longitude in degrees rather than radians
    #SOP_LONGITUDE_DEG = [l*180.0/np.pi for l in SOP_LONGITUDE]
    SOP_LONGITUDE_DEG = np.rad2deg(SOP_LONGITUDE)
    # starting on the desired day
    n=0
    while (TIME_DAYS[n] < day):
        n += 1 # this n is where we want to start
    # EPIC takes data between 13.1 to 21.2 times per day
    # need to import 22 observations and then truncate to only one day
    t = TIME_DAYS[n:n+22]
    longitude = SOP_LONGITUDE_DEG[n:n+22]
    flux_rad = RAD_780[n:n+22] # Units: W m^-2 nm^-1
    
    # conversion to "reflectance" according to Jiang paper
    reflectance = flux_rad*np.pi/SOLAR_IRRAD_780 

    # truncate arrays to span one day only
    while ((t[-1] - t[0]) > 1.0):   # while t spans more than one day
        t = t[0:-1]                 # truncate arrays 
        longitude = longitude[0:-1]
        flux_rad = flux_rad[0:-1]
        reflectance = reflectance[0:-1]

    # error on reflectance
    reflectance_err = 0.02*reflectance # assuming 2% error     
    # add gaussian noise to the data with a variance of up to 2% mean reflectance
    gaussian_noise = np.random.normal(0, 0.02*np.mean(reflectance), len(reflectance))
    reflectance += gaussian_noise
    
    # check for nans in the reflectance data
    contains_nan = False 
    number_of_nans = 0
    for f in flux_rad:
        if math.isnan(f) == True:
            number_of_nans += 1
            contains_nan = True     
    #if contains_nan: # data not usable
    #    print("CAUTION: "+str(number_of_nans)+" points in this set are NaN")
       # return t, longitude, reflectance, reflectance_err, contains_nan
    
    # if we want to plot the raw data
    if (plot):
        # plotting reflectance over time
        fig = plt.figure()
        ax1 = fig.add_subplot(111)    
        ax1.errorbar((t - t[0])*24, reflectance, yerr=reflectance_err, fmt='.', 
                     markerfacecolor="cornflowerblue", 
                     markeredgecolor="cornflowerblue", color="black")
        ax1.set_ylabel("Apparent Albedo "+r"$A^*$", size=18)
        ax1.set_xlabel("T-minus 13 June 2015 00:00:00 UTC [Days]", size=18)
        title = r"EPIC data [$d$ = {}, $\phi_0$ = {}] ".format(util.date_after(day),np.round(np.deg2rad(longitude[0]),3))
        plt.title(title)
        plt.rcParams.update({'font.size':14})
        plt.show()
    data.close()
    return t, longitude, reflectance, reflectance_err, contains_nan

def multiple_EPIC_days(Days,DaySim):
    """
    Function to load multiple EPIC days of data to be fed to runSatellowan
    Inputs : 
        Days : number of days, int
        DaySim : starting day of simulation, also an int
    Output : data -> list containing all the data
    """
    t = np.empty((0))
    reflectance = np.empty((0))
    reflectance_err = np.empty((0))
    contains_nan = np.empty((0))
    for k in range(Days):
        a,b,c,d,e = EPIC_data(DaySim+k,plot = False) # there will be a plot of the initial albedo map coming from the call of runstaellowan
        if c.shape[0] != 22: #number of data point we want, then we interpolate
            fig12 = plt.figure()
            plt.title("Interpolation")
            plt.errorbar(a,c,yerr = d, fmt = '.',markersize = 8, solid_capstyle = 'projecting', capsize = 4, label = 'EPIC')
            f = interp1d(a,c,'cubic')
            a = np.linspace(a[0],a[-1],22,endpoint = True) ##le probleme est ici !
#            a = np.linspace(a[0],a[0]+1,22,endpoint=True)
#            print("time on day {} is {}".format(k+1,a))
            #print("for day {}, there is {} points".format(k+1,c.shape[0]))
            c = f(a)            
            #plt.plot(a,c,'o--k',label = 'Interpolation')
            #plt.title("Day {} after {}".format(k+1,DaySim))
            #plt.legend()
            #plt.show()
            # error on reflectance
            d = 0.02*c # assuming 2% error     
            # add gaussian noise to the data with a variance of up to 2% mean reflectance
            gaussian_noise = np.random.normal(0, 0.02*np.mean(c), len(c))
            c += gaussian_noise
        t = np.append(t,a)
        reflectance=np.append(reflectance,c)
        reflectance_err=np.append(reflectance_err,d)
        contains_nan=np.append(contains_nan,e)
#        print("pour le jour {},il y a {} points de mesures".format(k+1,len(a)))
    t               = np.asarray(t)
    reflectance     = np.asarray(reflectance)
    reflectance_err = np.asarray(reflectance_err)
    contains_nan    = np.asarray(contains_nan)
    data = [t,reflectance,DaySim,reflectance_err,contains_nan]
    return data
