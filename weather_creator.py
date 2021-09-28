# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:07:35 2020

@author: lucas
"""

import numpy as np
import matplotlib.pyplot as plt
import random as ran
import init as var
import data_prep as data
import Model_Init as M_init
import MCMC as m
import Utilities as util


    
def extractN(time,apparent,n,Day):
    """
    Function that extracts N evenly spaced data from the apparent albedo array given. 
    It also makes sure that the first (0th hour) and the last (23th hour) is included.
    """
    limit = len(apparent)
    print (limit)
    diff = int(limit/(n*Day))
    indicesFinal = []
    for j in range(Day):
        indices = [i*diff+j*int((limit/Day)) for i in range(n) if i*diff<limit/Day]
        indicesFinal.append(indices)
    
    indicesFinal= np.asarray(indicesFinal).flatten()
    t = [time[i] for i in indicesFinal]
    a = [apparent[i] for i in indicesFinal]
    return t,a        


def RunSatellowan(numOfSlices,nslices_data, Acloud,w,ndata,Days,nwalkers,nsteps,timespan,
                  phispan,burning,plot = False, mcmc = True, walker = True, forming = True, EPIC = None):
    """
    Function that does everything basically. 
    Loads the synthetic or EPIC data, then calls the MCMC part on them.
    Make a plot of the lightcurve, the mcmc lightcurve and few samples of the mcmc
    Inputs : numOfSlices : nbr of slices
             Acloud : True cloud albedo used for synthetic data
             w : rotation speed of the planet
             Day : number of Days the simulations is spanning
             nwalkers : number of walkers
             nsteps : number of steps for the mcmc
             timespan : fraction of day the simulation spans
             phispan : fraction of lonfitude the simulation spans
             burning : number of steps to disregard in the mcmc
             plot : to plot or not to plot
             mcmc : to mcmc or not to mcmc
             walker : to plot the walker and cornerplot or not
             forming : not used, but too choose if we generate clouds or not
             Epic : If None, synthetic data is created, otherwise needs to give the Epic data
            
    Ouputs : surface albedo map, fclouds, TOA albedo map, lightcurve, ALbedo of the clouds, computed by the mcmc
            for synthetic data, we also return the true surface, true fclouds, true TOA albedo map, true light curve
            percentile is the 16th and 84th percentil on each paramters, for each slices and days
    """
    
#    hPerday = int((w/(2*np.pi))**(-1))
    repetition = 0
    dic = {} # dictionnary to put all the results in
    while repetition < var.NBRREPEAT:
        print("iteration n° {}".format(repetition))
        if type(EPIC) == type(None):
            t,a,a_err,Delta_A,surf,effAlb = data.fake_data(nslices_data,w,Days,Acloud,ndata,timespan,phispan)
            print("Done extracting, there is {} slices for the MCMC, and {} that generated the synthetic lightcurve ".format(numOfSlices, nslices_data))            
        else : #if EPIC data is used
            t     = EPIC[0]
            a     = EPIC[1]
            a_err = EPIC[3]
            t     = (t-t[0])*24
            Delta_A = np.zeros((Days,numOfSlices))
            surf = np.zeros(numOfSlices) #so there's no mistakes when run on EPIC data. Will be saved in netcdf as a vector filled with zeros


        if (mcmc):
            mean_mcmc_time,mean_mcmc_lcurve,mean_mcmc_effAlb,chain = m.MCMC(nwalkers,nsteps,numOfSlices,
                                                                        t,a,a_err,timespan,phispan,
                                                                        burning)
            print("Got the MCMC results, all good here")

        if var.NBRREPEAT == 1:
            results       = m.mcmc_results(chain, burning)
            percentile    = m.mcmc_percentiles(chain,burning) # already a numpy array
            mcmc_surf_alb = np.asarray(results[:numOfSlices])
            mcmc_Delta_A  = np.asarray(results[numOfSlices:]).reshape((Days,numOfSlices))
            if type(EPIC)==type(None):
                return t,mcmc_surf_alb,mcmc_Delta_A,mean_mcmc_effAlb, mean_mcmc_lcurve,surf,Delta_A, effAlb, a, percentile, chain, a_err
            else: 
                return t,mcmc_surf_alb,mcmc_Delta_A,mean_mcmc_effAlb, mean_mcmc_lcurve, a, percentile, chain, a_err
        else: 
            results = m.mcmc_results(chain,burning)
            dic['mcmcsurf{}'.format(repetition)]     = np.asarray(results[:numOfSlices])
            dic["surf{}".format(repetition)]         = surf
            dic["Delta_A{}".format(repetition)]      = Delta_A
            dic["mcmc_Delta_A{}".format(repetition)] = np.asarray(results[numOfSlices:]).reshape((Days,numOfSlices))
            dic['percentile{}'.format(repetition)]   = m.mcmc_percentiles(chain,burning) 
            dic["chain{}".format(repetition)]        = chain
            dic["th_lc{}".format(repetition)]        = a
            dic["th_lc_err{}".format(repetition)]    = a_err
            repetition +=1
    return dic 
    