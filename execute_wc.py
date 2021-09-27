# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:43:00 2020

@author: lucas
"""


import weather_creator as wc
import writedata_2 as write
import init as var
#import routine_plopt as rplt
import numpy as np
import data_prep as datap
steps  = var.NSTEPS[0] 
burnin = var.BURNIN
epic = True 

if epic == True:
    daySim = 730 #start at 790 in the others fit
    data = datap.multiple_EPIC_days(var.DAYS,daySim) # load epic data for one or multiple day
    t,mcmc_surf_alb,mcmc_Delta_A,mean_mcmc_effAlb, mean_mcmc_lcurve, a, percentile, chain, a_err = wc.RunSatellowan(var.NUMOFSLICES,var.NUMOFSLICES,var.ACLOUD,var.WW,var.NDATA,var.DAYS,
                                                                                                                                           var.NWALKERS,steps, var.TIMESPAN,var.PHISPAN,var.BURNIN,
                                                                                                                                           plot = False, mcmc = True, walker = False, forming = True,
                                                                                                                                           EPIC = data)
    surf = np.zeros((var.NUMOFSLICES))
    Delta_A = np.zeros((var.DAYS,var.NUMOFSLICES))
    effAlb = Delta_A

else:
##let's call wc.RunSatellowan ON FAKE DATA, all the ouptus are numpy.ndarray type
#    print("on est la hein")
    t,mcmc_surf_alb,mcmc_Delta_A,mean_mcmc_effAlb, mean_mcmc_lcurve,surf,Delta_A, effAlb, a, percentile, chain, a_err = wc.RunSatellowan(var.NUMOFSLICES,var.NUMOFSLICES,var.ACLOUD,var.WW,var.NDATA,var.DAYS,
                                                                                                                                  var.NWALKERS,steps,var.TIMESPAN,var.PHISPAN,var.BURNIN,
                                                                                                                                  plot = False, mcmc = True, walker = False, forming = True,
                                                                                                                                  EPIC = None)
#init_pos,fclouds,surf = wc.RunSatellowan(var.NUMOFSLICES,var.ACLOUD, var.WW, var.NDATA,var.DAYS,var.NWALKERS,steps, var.TIMESPAN, var.PHISPAN, var.BURNIN, plot = False, mcmc = True, walker = False, forming = True, EPIC =None)
#init_pos = np.asarray(init_pos)

if epic == True:
    nom='EPIC_{}jours_{}slices_{}steps_{}burn_startday{}_test2'.format(var.DAYS,var.NUMOFSLICES,steps,burnin,daySim)
else:
    nom = 'Fake_{}jours_{}slices_{}steps_{}burn_loguniform_3_0_da01'.format(var.DAYS,var.NUMOFSLICES,steps,burnin)
write.writefile(t,mcmc_surf_alb,mcmc_Delta_A,mean_mcmc_effAlb,mean_mcmc_lcurve,surf,Delta_A,effAlb,a,percentile,chain,a_err,nom)
