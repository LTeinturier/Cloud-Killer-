# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:42:45 2020

@author: lucas
"""

import netCDF4
import numpy as np
import init as var
""""
script called to create a csv file with data from the new model 
basically, i need to write the theoretical albedo surface (surf variable),
the data effective albedo (d variable), the MCMC computed surface map (alb variable),
the effective MCMC computed map (effAlb), and i can include the Acloud computed

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

def retrievelessdata(ncfile):
    nc = netCDF4.Dataset(ncfile,'r', format = 'NETCDF4')
    surf_mcmc  = nc.variables["surface_mcmc"][:]
    surf       = nc.variables["surface_theorique"][:]
    percentile = nc.variables["percentile"][:]
    return surf_mcmc, surf, percentile
    
    
    
def repeat_write(dic,surf,clouds,effAlb,lcurve,nomdufichier):
    nslices = surf.shape[0]
    nday    = clouds.shape[0]
    ndim    = 1 + nslices*(1+nday) # nbr of free param
    nomfichier = str(nomdufichier)+'.nc'
    ncfile = netCDF4.Dataset(nomfichier,'w',format='NETCDF4')
    ncfile.createDimension('Nslices',nslices)
    ncfile.createDimension('Day',nday)
    ncfile.createDimension('nbrdonnées',nday*var.NDATA)
    ncfile.createDimension('nwalkers',var.NWALKERS)
    ncfile.createDimension('nsteps',var.NSTEPS[0])
    ncfile.createDimension('ndim',ndim)
    ncfile.createVariable("surf_th",np.float64,('Nslices'))
    ncfile.variables["surf_th"][:] = surf
    ncfile.createVariable('clouds_th',np.float64,('Day','Nslices'))
    ncfile.variables['clouds_th'][:] = clouds
    ncfile.createVariable("TOA albedo th",np.float64,('Day','Nslices'))
    ncfile.variables["TOA albedo th"][:] = effAlb
    ncfile.createVariable("lcurve",np.float64,('nbrdonnées'))
    ncfile.variables["lcurve"][:] = lcurve
    for i in range(len(dic.keys())):
        ncfile.createVariable("chain{}".format(i),np.float64,('nwalkers','nsteps','ndim'))
        ncfile.variables["chain{}".format(i)][:] = dic['{}'.format(i)]
    ncfile.close()
    
def repeat_write2(dic,nom,nrepeat):
    clouds = dic["Delta_A0"]
    nslices_data = clouds.shape[1]
    nslice_mcmc  = dic['mcmcsurf0'].shape[0]
    nday    = clouds.shape[0]
    chain = dic["chain0"]
    nwalkers = chain.shape[0]
    nsteps = chain.shape[1]
    ndim    = chain.shape[-1] # nbr of free param
    lc_size = len(dic["th_lc0"])
    nomfichier = str(nom)+'.nc'
    ncfile = netCDF4.Dataset(nomfichier,'w',format = 'NETCDF4')
    ncfile.createDimension('Nslices_data',nslices_data)
    ncfile.createDimension('Nslices_mcmc',nslice_mcmc)
    ncfile.createDimension('Day',nday)
    ncfile.createDimension('ndim',ndim)
    ncfile.createDimension("deux",2)
    ncfile.createDimension("nwalkers",nwalkers)
    ncfile.createDimension("nsteps",nsteps)
    ncfile.createDimension("pt_lightcurve",lc_size)
    for j in range(nrepeat):     
        print("on écrit l'itération {}".format(j+1))
        ncfile.createVariable("mcmcsurf{}".format(j),np.float64,('Nslices_mcmc'))
        ncfile.createVariable("surf{}".format(j),np.float64,('Nslices_data'))
        ncfile.createVariable('Delta_A{}'.format(j),np.float64,('Day','Nslices_data'))
        ncfile.createVariable('mcmc_Delta_A{}'.format(j),np.float64,('Day','Nslices_mcmc'))
        ncfile.createVariable("percentile{}".format(j),np.float64,('ndim','deux'))
        ncfile.createVariable("chain{}".format(j),np.float64,('nwalkers','nsteps','ndim'))
        ncfile.createVariable("th_lc{}".format(j),np.float64,("pt_lightcurve"))
        ncfile.createVariable("th_lc_err{}".format(j),np.float64,("pt_lightcurve"))
        ncfile.variables["mcmcsurf{}".format(j)][:] = dic["mcmcsurf{}".format(j)]
        ncfile.variables["surf{}".format(j)][:] = dic["surf{}".format(j)]
        ncfile.variables["Delta_A{}".format(j)][:] = dic["Delta_A{}".format(j)]
        ncfile.variables["mcmc_Delta_A{}".format(j)][:] = dic["mcmc_Delta_A{}".format(j)]
        ncfile.variables["percentile{}".format(j)][:] = dic["percentile{}".format(j)]
        ncfile.variables["chain{}".format(j)][:] = dic["chain{}".format(j)]
        ncfile.variables["th_lc{}".format(j)][:] = dic["th_lc{}".format(j)]
        ncfile.variables["th_lc_err{}".format(j)][:] = dic["th_lc_err{}".format(j)]
    print("tout est écrit, on ferme")
    ncfile.close()

def repeat_read(ncfile,nbrrepeat):
    nc = netCDF4.Dataset(ncfile,'r',format = 'NETCDF4')
    surf   = nc.variables["surf_th"][:]
    clouds = nc.variables["clouds_th"][:]
    effAlb = nc.variables["TOA albedo th"][:]
    lcurve = nc.variables["lcurve"][:]
    dic = {}
    for i in range(nbrrepeat):
        key = '{}'.format(i)
        dic[key] = nc.variables["chain{}".format(i)][:]
    nc.close()
    return dic,surf,clouds,effAlb,lcurve
    
def repeat_read2(ncfile,nbrrepeat,chain=False,percentile=False):
    nc = netCDF4.Dataset(ncfile,'r',format = 'NETCDF4')
    print("nc opened in wd")
    dic = {}
    for i in range(nbrrepeat):
        print("loading iteration",i)
        dic["mcmcsurf{}".format(i)] = nc.variables["mcmcsurf{}".format(i)][:]
        dic["surf{}".format(i)] = nc.variables["surf{}".format(i)][:]
        dic["Delta_A{}".format(i)] = nc.variables["Delta_A{}".format(i)][:]
        dic["mcmc_Delta_A{}".format(i)] = nc.variables["mcmc_Delta_A{}".format(i)][:]
        if percentile:
            dic["percentile{}".format(i)] = nc.variables["percentile{}".format(i)][:]
        if chain:
            dic["chain{}".format(i)] = nc.variables["chain{}".format(i)][:]
    nc.close()
    return dic