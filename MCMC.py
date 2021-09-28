# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:23:28 2020

@author: lucas
"""

import numpy as np
import Model_Init as M_Init
import matplotlib.pyplot as plt
import corner
import scipy.optimize as op
import emcee
import init as var

### stat stuff ###

def lnlike(albedo,time,ref,ref_err,timespan,phispan):
    """
    albedo is a nslices+nslices*nday elements array contening 
        - asurf : ndarray of size numOfslices, albedo of surface -> albedo[2:nslices+2]
        - fclouds : ndarray of size (numDay, numOfSlices) contening all the fcloud ->albedo[nslices +2:] #needs to be reshaped
        - cloudAlbedo : float (we assume all the clouds to have the same albedo) ->albedo[1]
        - rotrate : float for the angular velocity of the planet -> albedo[0]
    time, ref,ref_err are the actual time since begin of simulation in hours
        light curve and error on light curve
    timspan : time in days on which the model should span
    phispan :  longitude in fraction of 2pi on which the model should span
    
    """
    nslices     = var.NUMOFSLICES 
    nday        = var.DAYS 
    asurf       = albedo[:nslices]

    Delta_A    = albedo[nslices:].reshape((nday,nslices))
    timepts     = len(time)
    ## Compute TOA albedo  and the light curve via the toa albedo computed: 
    model_ref  = []
    t_effalb = M_Init.effectiveAlbedo(asurf,np.exp(Delta_A)) # shape ndayx nslices
    for day in range(nday):

        ### obtain the lightcurve from the TOA albedo with the forward model
        temp_time, temp_ref = M_Init.Forwardmodel(t_effalb[day,:],var.WW,timespan, phispan,timepts//nday,  #we divide timepts by nday because we run the forxward model on 1 day at a time
                                                0, plot =False, alb = True)
        model_ref.append(temp_ref)
    model_ref  = np.asarray(model_ref).reshape((-1)) #*1000 is the n in forwardmodel
    ##compute ln(likelihood using chisq)
    chisq_num   = np.power(np.subtract(ref,model_ref),2) #(data-model)^2
    chisq_denom = np.power(ref_err,2) #(error)^2
    res = -0.5 * sum(chisq_num/chisq_denom + np.log(2*np.pi) +np.log(np.power(ref_err,2)))
    return res

def opt_lnlike(albedo,time,ref,ref_err,timespan, phispan):
    # print("In opt_lnlike")
    nll = lambda *args: - lnlike(*args)
    nslices = var.NUMOFSLICES
    bound = [(0.000001,1) for i in range(nslices)] #bounds for the surface albedo
    for i in range(nslices,len(albedo)):
        bound.append((-6,0)) # bounds for the clouds only
    bound = tuple(bound)
    result = op.minimize(nll,albedo,args=(time,ref,ref_err,timespan,phispan),method = 'L-BFGS-B', bounds = bound)
    return result["x"]
   


def lnprior(alpha):
    """
    the alpha here is nslices of surface, then Ln(dA)
    """
    surf = alpha[:var.NUMOFSLICES]
    logdA = alpha[var.NUMOFSLICES:]
    if np.all(surf>=0) and np.all(surf<1.) and np.all(logdA<0) and np.all(logdA>-6): # if valid values no lower bounds on logdA
        return 0.0
    return -np.inf #otherwise probability goes to 0

def lnpost(alpha, time, ref, ref_err,timespan,phispan):
    """
    Input: guesses for the fit parameters (alpha, an array of albedos,
    representing A(phi)) and the time, lightcurve, and error on the lightcurve 
    of the data being fit 
    Output: ln(posterior)
    """
    lp = lnprior(alpha)
    if not np.isfinite(lp): # if ln(prior) is -inf (prior->0) 
        return -np.inf      # then ln(post) is -inf too
    return lp + lnlike(alpha, time, ref, ref_err,timespan,phispan)

### end of stat stuff    
    

### MCMC part now 
    
def mcmc_results(chain,burnin):
    
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    
    Averages the position of all walkers in each dimension of parameter space 
    to obtain the mean MCMC results 
    
    Output:  arrayrepresenting the mean clouds albedo, the mean surface albedo map and the mean fclouds found via MCMC
    """

    ndims = len(chain[0][0])
    flat  = flatten_chain(chain, burnin)
    
    mcmc_params = []
    for n in range(ndims): # for each dimension
        mcmc_params.append(np.median(flat[:,n])) # append the mean
    return mcmc_params
    
def init_walkers(alpha, time, ref, ref_err, ndim, nwalkers,timespan,phispan):
    """
    Input: guesses for the fit parameters (alpha, an array of cloud albedo (1 value),
    surface albedo map (nslices value ) andfraction of clouds on each slices for each day,
    the time, lightcurve,and error on the lightcurve ofthe data being fit,
    the number of dimensions (i.e., nslices +nday*nslices +1), and the number of walkers to initialize
    
    Initializes the walkers in albedo-space in a Gaussian "ball" centered 
    on the parameters which maximize the likelihood.
    
    Output: the initial positions of all walkers in the ndim-dimensional 
    parameter space
    """

    opt_albs = opt_lnlike(alpha, time, ref, ref_err,timespan,phispan)
    print("Initial guess before MCMC :")
    # print(opt_albs)
    np.savetxt("beforeMCMC.txt",opt_albs) #generate a txt file with the parameters before the MCMC
        # generate walkers in Gaussian ball
    pos = np.array([opt_albs + 1e-2*np.random.randn(ndim) for i in range(nwalkers)])
    compteur=0
    for nw in range(nwalkers): #corrects if the gaussian ball is to large
        if lnprior(pos[nw,:])==-np.inf:
            pos[nw,:]=np.array([opt_albs+1e-4*np.random.randn(ndim)])
            compteur+=1

    
    return pos

def make_chain(nwalkers, nsteps, ndim, t,r,r_err,timespan,phispan,alb=True):
    """
    Inputs : the number of walkers, the number of steps to take in the chain
            the number of dim =(nslices +1 +nday*nslices), time, lightcurve and lightcurve 
            error on the data being fit. timespan and phispan can be tuned as well
            
    Runs MCMC on either EPIC data for the given day of interest to see if MCMC 
    can obtain the map A(phi) which produced the lightcurve, OR, runs MCMC with
    some artificial albedo map A(phi) to see if MCMC can recover the input map.
    
    Output: an emcee sampler object's chain
    """
    # if making chain for real EPIC data or fake data
    if alb: 
        t = np.asarray(t)
        r = np.asarray(r)
        r_err = np.asarray(r_err)

    # if neither a day nor an articial albedo map is supplied
    else:
        print("Error: please supply either a day of interest in the EPIC data \
              or a synthetic array of albedo values.")
        return
    print ("Got my albedo, Thank you!")
    nslices = var.NUMOFSLICES # number of slices
    nday    = var.DAYS        # number of day the simulation is running
    # guess: alb is 0.25 for the surface albedo
    surf = [0.1 for n in range(nslices)]
    Delta_A = np.log(0.9*np.random.rand(nday*nslices)) # Ln of a random init
    init_guess = np.append(surf,Delta_A)
    # better guess: maximize the likelihood
    print("Maximizing the Likelihood right now ")
#    opt_params  = opt_lnlike(init_guess, t, r, r_err,timespan,phispan)  ## also done in init_walkers using the parameters ouptuted here
    print("Likelihood maximized !")
    # initialize nwalkers in a gaussian ball centered on the opt_params
    print ("Intializing Walkers...")
    init_pos = init_walkers(init_guess, t, r, r_err, ndim, nwalkers,timespan,phispan)
    print ("Walkers initialized, ready for destruction!")

    # set up the sampler object and run MCMC 
    print ("Setting up chain")
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(t, r, r_err,timespan,phispan))
    sampler.run_mcmc(init_pos, nsteps, progress = True)
    print ("chain completed")
    return sampler.chain

def flatten_chain(chain, burnin):
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    Output: a flattened chain, ignoring all steps pre-burnin
    """
    ndim = len(chain[0][0]) # number of params being fit 
    return chain[:,burnin:,:].reshape(-1, ndim)

def MCMC(nwalkers,nsteps,numOfSlices,time,app,app_err,timespan,phispan,burning):
    """
    MCMC main function. 
    Inputs : nwalkers : nbr of walkers, nstep : nbr of steps, numOfSlices : nbr of slices
             time, app, app_err : coming from the data, timespan and phispan : time and longitude 
             on which the simulation spans, burning = steps to disregards in the mcmc,
             hPerDay = 24*nbrday, hperday, i,ax and plot should disapear
    Outputs : mean_mcmc_time : mean mcmc time that might be bugué
              mean_mcmc_ref : lightcurve computed with the mean mcmc results
              chain : the different chains
              :
    """
    nday  = var.DAYS 
    ndim  = numOfSlices + numOfSlices * nday #asurf,  serie of fclouds (nslices parameters per day ie nslices*nday)
    chain = make_chain(nwalkers,nsteps,ndim,time,app,app_err,timespan,phispan,alb = True)
    print("chain is made")
    
    mean_mcmc_params = mcmc_results(chain,burning)
    # compute the effective albedo, to compute the mean mcmc lightcurve
    mean_mcmc_surfAlb   = np.asarray(mean_mcmc_params[:numOfSlices])
    mean_mcmc_Delta_A = np.asarray(mean_mcmc_params[numOfSlices:]).reshape((nday,numOfSlices))

    #need a loop here because of the ts fclouds
    mean_mcmc_ref    = []
    mean_mcmc_time   = []
    mean_mcmc_albedo = M_Init.effectiveAlbedo(mean_mcmc_surfAlb,np.exp(mean_mcmc_Delta_A))
    for day in range(nday):
        temp_mean_time, temp_mean_mcmc = M_Init.Forwardmodel(mean_mcmc_albedo[day,:],var.WW,time_days=
                                                             timespan,long_frac=phispan,
                                                             n=5000,plot=False,alb=True)
        mean_mcmc_ref.append(temp_mean_mcmc)
        mean_mcmc_time.append(temp_mean_time+24*day)
    
    print("The MCMC is computed ! ")
    mean_mcmc_ref    = np.asarray(mean_mcmc_ref)
    mean_mcmc_time   = np.asarray(mean_mcmc_time)
    mean_mcmc_albedo = np.asarray(mean_mcmc_albedo)
    
    return mean_mcmc_time, mean_mcmc_ref, mean_mcmc_albedo,chain


def mcmc_percentiles(chain, burnin, pers=[16,84]):
    """
    Input: an emcee sampler chain and the steps taken during the burnin
    
    Output: an array of the percentiles desired by the user (default: 16th and 
    84th) found by MCMC
    """
    ndims = len(chain[0][0]) # obtain no. of dimensions
    flat = flatten_chain(chain, burnin) # flattened chain, post-burnin
    
    mcmc_percentiles = []
    for n in range(ndims): # for each dimension 
        mcmc_percentiles.append(np.percentile(flat[:,n], pers, axis=0))
    mcmc_percentiles = np.asarray(mcmc_percentiles)
    return mcmc_percentiles
