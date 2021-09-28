# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:42:45 2020

@author: lucas
"""

import netCDF4
import numpy as np
import init as var
""""

routine to save data in a netcdf4 file. 
routine to read the same file 
"""

def writefile(t,mcmc_surf,mcmc_Delta_A,mean_mcmc_effAlb,mcmc_lcurve,surf,Delta_A,effAlb,lcurve,percentile,chain,lcurve_err,nomdufichier):
    nomfichier = str(nomdufichier)+'.nc'
    ncfile = netCDF4.Dataset(nomfichier,'w',format='NETCDF4')
    Slices_dim = ncfile.createDimension('Nslices',var.NUMOFSLICES)
    Day_dim    = ncfile.createDimension('Day',var.DAYS)
    para_x_day = ncfile.createDimension("nbrdonnée",var.DAYS*var.NDATA)
    npoint = mcmc_lcurve.shape[1]
    nbrdata    = ncfile.createDimension("nombredepoint",npoint)
    nfree =  var.NUMOFSLICES + var.DAYS*var.NUMOFSLICES
    nbrfreeparam = ncfile.createDimension("nbrfreeparam",nfree)
    deu = ncfile.createDimension("deux",2)
    nw  = ncfile.createDimension("nwalkers",var.NWALKERS)
    nst = ncfile.createDimension("nsteps",var.NSTEPS[0])
    surf_th = ncfile.createVariable('surface_theorique',np.float64,("Nslices"))
    surf_th.long_name = "'true' surface albedo map"
    ncfile.variables["surface_theorique"][:] = surf
    effAlb_th = ncfile.createVariable("Alb_TOA_th",np.float64,("Day","Nslices"))
    effAlb_th.long_name = "'true' Top Of Atmosphere map"
    ncfile.variables['Alb_TOA_th'][:] = effAlb
    tr_clouds = ncfile.createVariable("trueDelta_A",np.float64,("Day","Nslices"))
    tr_clouds.long_name = "'true Delta_A'"
    ncfile.variables["trueDelta_A"][:] = Delta_A
    mcmc_cloud = ncfile.createVariable("mcmc_Delta_A",np.float64,("Day","Nslices"))
    mcmc_cloud.long_name = 'Delta_Acomputed by MCMC'
    ncfile.variables["mcmc_Delta_A"][:] = mcmc_Delta_A
    effAlb_MCMC = ncfile.createVariable("Alb_TOA_mcmc",np.float64,("Day","Nslices"))
    effAlb_MCMC.long_name = "MCMC results for the Top Of Atmosphere albedo map"
    ncfile.variables["Alb_TOA_mcmc"][:] = mean_mcmc_effAlb
    surf_MCMC  = ncfile.createVariable("surface_mcmc",np.float64,("Nslices"))
    surf_MCMC.long_name = " MCMC results for the surface albedo map"
    ncfile.variables['surface_mcmc'][:] = mcmc_surf
    mcmc_light = ncfile.createVariable("mcmc_lightcurve",np.float64,("Day","nombredepoint"))
    mcmc_light.long_name = 'MCMC mean predicted lightcurve'
    ncfile.variables["mcmc_lightcurve"][:] = mcmc_lcurve
    trlightcurve = ncfile.createVariable("th_lightcurve",np.float64,("nbrdonnée"))
    trlightcurve.long_name = 'True lightcurve'
    ncfile.variables['th_lightcurve'][:]= lcurve
    time_lcurve = ncfile.createVariable("time",np.float64,("nbrdonnée"))
    ncfile.variables["time"][:]=t
    lightc_err = ncfile.createVariable("lcurve_err",np.float64,('nbrdonnée'))
    lightc_err.long_name = 'error on true lightcurve'
    ncfile.variables["lcurve_err"][:] = lcurve_err
    perc  = ncfile.createVariable("percentile",np.float64,('nbrfreeparam',"deux"))
    perc.long_name = '16 and 84 percentiles on the free paramters of the model'
    ncfile.variables["percentile"][:] = percentile
    c   = ncfile.createVariable("chain",np.float64,("nwalkers","nsteps","nbrfreeparam"))
    c.long_name = 'chain computed by mcmc'
    ncfile.variables["chain"][:] = chain
    ncfile.close()

def retrievedata(ncfile):
    """
    ncfile is the name of the netcdf4 file that contains the data we want
    """
    nc = netCDF4.Dataset(ncfile,'r',format = 'NETCDF4')
    dic = nc.variables
    surf = nc.variables["surface_theorique"][:]
    TOA_alb = nc.variables["Alb_TOA_th"][:]
    mcmc_TOA = nc.variables["Alb_TOA_mcmc"][:]
    surf_mcmc = nc.variables["surface_mcmc"][:]
    mcmc_Delta_A = nc.variables["mcmc_Delta_A"][:]
    Delta_A = nc.variables["trueDelta_A"][:]
    true_lightcurve = nc.variables["th_lightcurve"][:]
    mcmclightcurve = nc.variables["mcmc_lightcurve"][:]
    lcurve_err = nc.variables["lcurve_err"][:]
    percentile = nc.variables["percentile"][:]
    chain   = nc.variables["chain"][:]
    t = nc.variables["time"][:]
    nc.close()
    return surf,TOA_alb,Delta_A,mcmc_TOA,surf_mcmc,mcmc_Delta_A,mcmclightcurve,true_lightcurve,percentile,chain,lcurve_err,t

