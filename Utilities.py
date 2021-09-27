import numpy as np 
import itertools
from astropy.time import Time
import MCMC as m
import Model_Init as M_Init

#Utilities
def roll(Dict,shift):
    slices = np.fromiter(Dict.keys(), dtype=int)
    albedo = np.fromiter(Dict.values(),dtype=float)
    albedo = np.roll(albedo,shift)
    slices = np.roll(slices,shift)
    Dict.clear()
    for i in slices:
        Dict[i] = albedo[i]
    return Dict

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def date_after(d):
    """
    Input: an integer d
    
    Quickly find out the actual calendar date of some day in the EPIC dataset. 
    
    Output: the date, d days after 2015-06-13 00:00:00.000
    """
    
    t_i = Time("2015-06-13", format='iso', scale='utc')  # make a time object
    t_new_MJD = t_i.mjd + d # compute the Modified Julian Day (MJD) of the new date
    t_new = Time(t_new_MJD, format='mjd') # make a new time object
    t_new_iso = t_new.iso # extract the ISO (YY:MM:DD HH:MM:SS) of the new date
    t_new_iso = t_new_iso.replace(" 00:00:00.000", "") # truncate after DD
    return t_new_iso

def zscore(truth,mcmc,percentile): 
    if len(truth.shape) >1 : truth = truth.flatten()
    sig = percentile[:,1]-percentile[:,0]
    inter   = (truth-mcmc)/sig
    zscore  = np.mean(inter)
    std     = np.std(inter)
    return zscore ,std 

def distrib_zscore(dic,true,burning,label):
    nrepeat = len(dic.keys())
    Z = np.empty((nrepeat))
    if label =='surf':
        nslices = true.shape[0]
        for n in range(nrepeat):
           mcmc = m.mcmc_results(dic['{}'.format(n)],burning)[1:nslices+1]
           perc = m.mcmc_percentiles(dic['{}'.format(n)],burning)[1:nslices+1,:]
           Z[n]=zscore(true,mcmc,perc)
    elif label =='clouds':
        nslices = true.shape[1]
        for n in range(nrepeat):
            mcmc = m.mcmc_results(dic['{}'.format(n)],burning)[nslices+1:]
            perc = m.mcmc_percentiles(dic['{}'.format(n)],burning)[nslices+1:,:]
            Z[n] = zscore(true,mcmc,perc)
    else: # if we run zscore on all data
        for n in range(nrepeat):
            mcmc = m.mcmc_results(dic['{}'.format(n)],burning)
            perc = m.mcmc_percentiles(dic['{}'.format(n)],burning)
            Z[n] = zscore(true,mcmc,perc)
    return Z  

def dicresults(dic,burning):
    nrepeat = len(dic.keys())
    ndim    = dic['0'].shape[2]
    dicres = np.empty((ndim,nrepeat))
    dicperc = np.empty((2,ndim,nrepeat))
    error = np.empty((2,ndim,nrepeat))
    for n in range(nrepeat):
        dicres[:,n] = m.mcmc_results(dic['{}'.format(n)],burning)
        dicperc[:,:,n] = m.mcmc_percentiles(dic['{}'.format(n)],burning).T
        error[0,:,n] = dicres[:,n]-dicperc[0,:,n]
        error[1,:,n] = dicperc[1,:,n]-dicres[:,n]
    return dicres, dicperc,error

def meanchain(dic,burning):
    nrepeat = len(dic.keys())
    meanres = np.zeros((dic['0'].shape[2]))
    meanchain = np.empty((dic['0'].shape[0],dic['0'].shape[1],dic['0'].shape[2]))
    for k in range(nrepeat):
        flat = m.flatten_chain(dic['{}'.format(k)],2000)
        meanchain += dic['{}'.format(k)]
        meanres += np.mean(flat, axis = 0)
    meanres   /= nrepeat # compute mean over all simulations
    meanchain /= nrepeat
    return meanchain,meanres

def gelmanrubin(dic,burn):
    nbrrepeat = len(dic.keys())
    ndim = dic['0'].shape[2]
    mean = {}
    var  = {}
    dblmean = np.zeros(ndim) # array size ndim
    meanvar = np.zeros(ndim) #array size ndim
    varofmean = np.zeros(ndim)#array size ndimension
    for n in range(nbrrepeat):
        key = '{}'.format(n)
        chain = m.flatten_chain(dic[key],burn)
        mean[key] = np.mean(chain,axis = 0)
        var[key]  = np.var(chain, axis = 0)
        dblmean  += mean[key]
        meanvar  += var[key]
    dblmean /= nbrrepeat
    meanvar /= nbrrepeat
    for n in range(nbrrepeat):
        key = '{}'.format(n)
        varofmean += np.power(mean[key]-dblmean,2)
    varofmean /= (nbrrepeat-1)
    B = varofmean- 1/nbrrepeat*meanvar
    R = np.sqrt(1 + B/meanvar)
    return R

def TOA(dic,burning,nsl):
    nrepeat = len(dic.keys())
    mres = dicresults(dic,burning)
    msurf = mres[1:nsl+1,:]
    mclouds = mres[nsl+1:,:].reshape(-1,nsl,nrepeat)
    nday = mclouds.shape[0]
    TOA = np.empty((nday,nsl,nrepeat))
    for n in range(nrepeat):
        for day in range(nday):
            TOA[day,:,n] = M_Init.effectiveAlbedo(nsl,mres[0,n],
                                                  plot = False,calClouds = 
                                                  mclouds[day,:,n],
                                                  calsurf = msurf[:,n])
    return TOA

def TOA_uncertainties(dic,burning,nsl,nday):
    nrepeat = len(dic.keys())
    error = np.empty((2,nday,nsl,nrepeat))
    for n in range(nrepeat):
        res = m.mcmc_results(dic['{}'.format(n)],burning)
        perc = m.mcmc_percentiles(dic['{}'.format(n)],burning)
        low,high = res-perc[:,0], perc[:,1]-res
        perc_fclouds_up   = high[nsl+1:].reshape(-1,nsl)
        perc_fclouds_down = low[nsl+1:].reshape(-1,nsl)
        for d in range(nday):
            up = M_Init.effectiveAlbedo(nsl,high[0], plot = False, calClouds =
                                    perc_fclouds_up[d,:],calsurf = high[1:nsl+1])
            down = M_Init.effectiveAlbedo(nsl,low[0],plot = False, calClouds = 
                                          perc_fclouds_down[d,:], calsurf=low[1:nsl+1])
            error[0,d,:,n] = down
            error[1,d,:,n] = up
    error = np.reshape(error,(2,nsl*nday,nrepeat))
    maxuncer = np.empty((2,nsl*nday))
    for n in range(error.shape[1]):
        ind = np.argmax(error[1,n,:]-error[0,n,:])
        maxuncer[:,n] = error[:,n,ind]
        
    return maxuncer

def TOA_mean_uncertainties(dic,burning,nsl,nday):
    nrepeat = len(dic.keys())
    error = np.empty((2,nday,nsl,nrepeat))
    for n in range(nrepeat):
        res = m.mcmc_results(dic['{}'.format(n)],burning)
        perc = m.mcmc_percentiles(dic['{}'.format(n)],burning)
        low,high = res-perc[:,0], perc[:,1]-res
        perc_fclouds_up   = high[nsl+1:].reshape(-1,nsl)
        perc_fclouds_down = low[nsl+1:].reshape(-1,nsl)
        for d in range(nday):
            up = M_Init.effectiveAlbedo(nsl,high[0], plot = False, calClouds =
                                    perc_fclouds_up[d,:],calsurf = high[1:nsl+1])
            down = M_Init.effectiveAlbedo(nsl,low[0],plot = False, calClouds = 
                                          perc_fclouds_down[d,:], calsurf=low[1:nsl+1])
            error[0,d,:,n] = down
            error[1,d,:,n] = up
    error = np.reshape(error, (2,nsl*nday,nrepeat))
    mean_uncertainties = np.empty((2,nsl*nday))
    mean_uncertainties[0,:]= np.mean(error[0,:,:], axis = 1)
    mean_uncertainties[1,:] = np.mean(error[1,:,:],axis = 1)
    return mean_uncertainties

def zscore_per_slice(surftot,truesurf):
    ###perc is shape (2,nsl,nrepeat)
    nslices = truesurf.shape[0]
    nrepeat = surftot.shape[1]
    zscore = np.empty((nslices,nrepeat))
    for n in range(nrepeat):
        sigma = np.std(surftot[:,n])
        zscore[:,n] = (surftot[:,n]-truesurf)/sigma
    return zscore
        