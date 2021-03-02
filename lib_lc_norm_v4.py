import numpy as np
import matplotlib.pyplot as plt
import csaps
from csaps import UnivariateCubicSmoothingSpline as ss
from astropy.stats import sigma_clip
import warnings
from scipy.optimize import minimize_scalar
import os
from os.path import join as opj
import lightkurve as lk
from os import listdir
from glob import glob
from astropy.io import fits



def get_lightcurves(kid, quarters=range(18), quality_bitmask='default', return_file_dirs=False):
    '''
    Load Kepler data from the remote PC hard drive and put it into a lightkurve class variable
    
    INPUTS:
    
    kid - (str) kepler ID
    
    quarters - (array or list) the Kepler quarters you want to load
    
    quality_bitmask - (str) which identifies the quality flag bitmask that should be used to mask out bad cadences. 
    
                OPTIONS:
                “none”: no cadences will be ignored (quality_bitmask=0).

                “default”: cadences with severe quality issues will be ignored (quality_bitmask=1130799).

                “hard”: more conservative choice of flags to ignore (quality_bitmask=1664431). This is known to remove good data.

                “hardest”: removes all data that has been flagged (quality_bitmask=2096639). This mask is not recommended.

                See the KeplerQualityFlags class on Lightkurve website for details on the bitmasks.
    
    return_file_dirs - (bool) will cause function to also return the physical locations of each FITS file loaded here
    
    
    OUTPUTS:
    
    lc - (class lightkurve.LightCurve) The light curve data in Lightkurve (package) format
    
    dirs - (array) A numpy array of the directories each file came from. 
                   This output is enabled by return_file_dirs. 
                   Same order as "quarters" variable.
    
    '''
    
    # Make sure this is a numpy array
    quarters = np.array(quarters)
    
    # Where the kepler data is stored
    data_dir0 = '/media/etdisk1/kepdata/data/lightcurves'
    
    # Find the folder for this target. It expects prepended zeros, so we use a wildcard to find the folder
    folder_name = glob(opj(data_dir0, '*' + kid))[0]
    
    # Find all the available files for this target
    file_names = listdir(opj(data_dir0, folder_name))

    #print('Found', len(file_names), 'quarters for this target')
    
    lcs, dirs, qns = [], [], []
    
    for file_num in range(len(file_names)):
        
        # Path to the fits file
        lc_path = opj(data_dir0, folder_name, file_names[file_num])
        
        # Get Kepler quarter number of this fits file
        this_file_quarter = fits.getheader(lc_path, 0)['QUARTER']
        
        if np.any(this_file_quarter == quarters): # Was this quarter requested?
            
            qns.append(this_file_quarter)
            
            dirs.append(lc_path)
            
    # Apply sort to directories to make sure they are in ascending order
    dirs = np.hstack(dirs)[np.argsort(qns)]
    
    for file_path in dirs:
        
        # Extract light curve data from the fits file
        lcs.append(lk.lightcurvefile.KeplerLightCurveFile(file_path, quality_bitmask=quality_bitmask))
        
    
    # Make lightkurve collection
    lcs = lk.collections.LightCurveFileCollection(lcs)
    
    # Stich all quarters together to make a single light curve
    if len(file_names) > 1:
        lc = lcs.PDCSAP_FLUX.stitch()
    else:
        lc = lcs[0]

    # Remove nans form the data
    lc = lc.remove_nans()
    
    si = np.argsort(lc.time)
    
    lc.time = lc.time[si]
    lc.flux = lc.flux[si]
    lc.flux_err = lc.flux_err[si]
    
    if return_file_dirs:
        
        return lc, dirs
        
    else:
        
        return lc
    
    
def normalize_region(wl, flux, ferr, sds, plot_q=True, ss_smo=1e-3):
    
    __, gi = stdFiltIt(wl, np.copy(flux), 
                       np.ones_like(flux), 
                       sds, smo=ss_smo, plot_q=plot_q)

    cont_fit = csaps.UnivariateCubicSmoothingSpline(wl[gi], 
                                                    flux[gi], 
                                                    smooth=ss_smo
                                                   )(wl)

    if plot_q:
        
        plt.figure(figsize=(10, 5))
        plt.plot(wl, flux, '.-k')
        plt.plot(wl, cont_fit, '-r')
        plt.title('Fit')
        plt.xlabel('time')
        plt.ylabel('Flux')
        
        plt.grid(1)
        
        plt.show()
    
    #####################################
    # Normalize continuum

    flux_norm = flux / cont_fit
    
    ferr_norm = ferr / cont_fit

    if plot_q:
        
        plt.figure(figsize=(10, 5))
        #plt.plot(wl, flux_norm, '.-k')
        plt.plot(wl, flux_norm, '-', c=[0.5, 0.5, 0.5], linewidth=1, label='Flux')
        
        if 1:
            plt.fill_between(wl, flux_norm - ferr_norm, flux_norm + ferr_norm, 
                             step='mid', 
                             color=[0.85, 0.85, 0.85], label='Flux Error')
        
        plt.axhline(1.0, color='m', linewidth=1, label='Continuum')
            
        plt.title('Normalized')
        
        plt.xlabel('time [day]')
        plt.ylabel('Flux [ppt]')
        
        plt.grid(1)
        
        plt.show()
    
    return cont_fit, flux_norm, ferr_norm


def wlr2ir(wl, wl_bounds):
    
    line_idx_bnds = [0, 0]

    line_idx_bnds[0] = np.argwhere(np.abs(wl - wl_bounds[0]) == np.nanmin(np.abs(wl - wl_bounds[0])))
    line_idx_bnds[1] = np.argwhere(np.abs(wl - wl_bounds[1]) == np.nanmin(np.abs(wl - wl_bounds[1])))

    # What indicies span within the wavelength bounds?
    return np.arange(line_idx_bnds[0], line_idx_bnds[1] + 1)


def stdFiltIt(wl, arr_1d, weights_1d, sdms, smo, plot_q):
    
    if plot_q:
        plt.figure(figsize=(10, 5))
        plt.title('Filter plot')
        plt.plot(wl, arr_1d, '.k')

    fit = arr_1d + np.nan
    
    for sdm in sdms:

        gi = np.logical_and(np.isfinite(weights_1d), np.isfinite(arr_1d))
        
        fit = csaps.UnivariateCubicSmoothingSpline(wl[gi], arr_1d[gi], smooth=smo)(wl)

        dy_sd = np.std(arr_1d[gi] - fit[gi]) * sdm

        gi = np.logical_and(np.abs(arr_1d - fit) <= dy_sd, gi)

        if plot_q:
            plt.plot(wl[np.logical_not(gi)], arr_1d[np.logical_not(gi)], 'xr')
            plt.plot(wl, fit + dy_sd, ':g')
            plt.plot(wl, fit - dy_sd, ':r')
            plt.grid(True)

        arr_1d[np.logical_not(gi)] = np.nan

    if plot_q:
        plt.show()

    return arr_1d, gi


def bjd2bkjd(bjd):
    
    return bjd - 2454833.0


def bkjd2bjd(bkjd):
    
    return bkjd + 2454833.0


def mask_mulit_transits(t0_bkjd, period_day, duration_day, time_bkjd, flux, 
                        lc=None, return_index=True):
    
    gi = ~np.ones(len(flux), dtype=bool)
    
    for p, t, d in zip(period_day, t0_bkjd, duration_day):
        
        
        gi0 = mask_transits(t, p, d, 
                            time_bkjd, flux, 
                            lc=None, return_index=True)
        
        #print(p, t, len(gi0) - np.sum(gi0))
        
                    
        gi = gi | gi0
                    
    return gi

def get_transit_times(t0_bkjd, period_day, duration_day, time_bkjd, flux):
    
    t0f_bkjd = np.copy(t0_bkjd)
   
    while t0f_bkjd > time_bkjd[0]:
       
        t0f_bkjd -= period_day
        
    #t0f_bkjd += period_day
        
    while t0f_bkjd < time_bkjd[0]:
       
        t0f_bkjd += period_day
        
    #t0f_bkjd -= period_day
       
    return np.arange(t0f_bkjd, time_bkjd.max(), step=period_day)
    

def mask_transits(t0_bkjd, period_day, duration_day, time_bkjd, flux, 
                  lc=None, return_index=True):
   
    # http://www.raa-journal.org/docs/Supp/ms4253tab1.txt
       
    if lc != None:
        
        time_bkjd = np.array(lc.astropy_time.value)

        flux = lc.flux
   
    t0f_bkjd = np.copy(t0_bkjd)
      
    tots_bkjd = get_transit_times(t0f_bkjd, period_day, duration_day, time_bkjd, flux)
   
    # For each tranist find all datapoints within the tranist duration window
   
    ntot_idx = np.ones(len(flux), dtype=bool)
   
    for tot in tots_bkjd:
       
        tot_idx = (time_bkjd > tot - duration_day / 2) & (time_bkjd < tot + duration_day / 2)
       
        ntot_idx = ntot_idx & ~tot_idx
    
    if return_index:
        return ~ntot_idx # Returns where transits occur
    else:
        return time_bkjd[ntot_idx], flux[ntot_idx]


def plot_transits_v0(t0_bkjd, period_day, duration_day, time_bkjd, flux):
   
    # http://www.raa-journal.org/docs/Supp/ms4253tab1.txt
   
    #time_bkjd = np.array(lc.astropy_time.value)
    
#     if time_bkjd[0] > 2400000.5:
        
#         time_bkjd = time_bkjd - 2400000.5
   
    #flux = lc.flux
   
    t0f_bkjd = np.copy(t0_bkjd)
   
    tots_bkjd = get_transit_times(t0f_bkjd, period_day, duration_day, time_bkjd, flux)
   
    # For each tranist find all datapoints within the tranist duration window
   
    ntot_idx = np.ones(len(flux), dtype=bool)
   
    for tot in tots_bkjd:
       
        tot_idx = (time_bkjd > tot - duration_day / 2) & (time_bkjd < tot + duration_day / 2)
       
        ntot_idx = ntot_idx & ~tot_idx
        
        plt.axvline(tot, c='r')
       
    plt.plot(time_bkjd[ntot_idx], flux[ntot_idx], '.k')
    plt.plot(time_bkjd[~ntot_idx], flux[~ntot_idx], '.r')
    
    plt.xlabel('time [day]')
    plt.ylabel('flux [ppt]');
    plt.show(block=0)
    
    
def plot_mp_transits_v0(t0_bkjd0, period_day0, duration_day, time_bkjd, flux, toi):
   
    # http://www.raa-journal.org/docs/Supp/ms4253tab1.txt
    
    t0f_bkjd0 = np.copy(t0_bkjd0)
   
    plt.figure(figsize=(10, 5))
    #plt.plot(time_bkjd, flux, '.r')
    cw = ['r','g','c', 'b', 'm']
    
    for it in np.arange(len(period_day0)):
        
        #print(it, period_day0, t0f_bkjd0)
        
        period_day, t0f_bkjd = period_day0[it], t0f_bkjd0[it]

        tots_bkjd = get_transit_times(t0f_bkjd, period_day, duration_day[it], time_bkjd, flux)

        # For each tranist find all datapoints within the tranist duration window
        ntot_idx = np.ones(len(flux), dtype=bool)

        for tot in tots_bkjd:

            tot_idx = (time_bkjd > tot - duration_day[it] / 2) & (time_bkjd < tot + duration_day[it] / 2)

            ntot_idx = ntot_idx & ~tot_idx

            plt.axvline(tot, c=cw[it])

        if it == 0:
            plt.plot(time_bkjd[ntot_idx], flux[ntot_idx], '.-k')
            
        if it == toi:
            plt.plot(time_bkjd[~ntot_idx], flux[~ntot_idx], '.', c=cw[it], 
                     label='P='+ str(np.around(period_day, 1)) + '  t0='+ str(np.around(t0f_bkjd, 1)) + '  d='+ str(np.around(duration_day[it], 3)) + ' (TOI)')
        
        else:
            plt.plot(time_bkjd[~ntot_idx], flux[~ntot_idx], '.', c=cw[it], 
                     label='P='+ str(np.around(period_day, 1)) + '  t0='+ str(np.around(t0f_bkjd, 1)) + '  d='+ str(np.around(duration_day[it], 3)))
        

    plt.legend()
    plt.xlabel('time [day]')
    plt.ylabel('flux [ppt]');
    plt.show(block=0)
    
    
def meas_bic(flux, cont_fit):

    sigma = np.median(np.diff(flux)) * 1.48 / np.sqrt(2)


    ssr = np.sum( (flux - cont_fit)**2 )


    npoints = len(flux)


    penalty_coeff = 1.0


    nparams = 1.0 # number of knots + degree of spline - 1

    # The following term is -2*ln(L), where L is the likelihood of the data
    # given the model, under the assumption that the model errors are iid
    # Gaussian with mean 0 and standard deviation sigma.
    likelihood_term = npoints * np.log(2 * np.pi * sigma**2) + ssr / sigma**2

    # Bayesian information criterion.
    bic = likelihood_term + penalty_coeff * nparams * np.log(npoints)
    
    return bic



def meas_sigdif(time, flux, weights, cont_fit, sigma_i):

    #df = flux / cont_fit
    
    #sigma_f = np.std(np.diff(df)) #* 1.48 / np.sqrt(2)
    
    sigma_f = meas_sig(time, flux / cont_fit, weights, plot_q=0)
    
    return (sigma_i - sigma_f)**2


def split(all_time, all_flux, gap_width=0.75, dft=0.05, index_only=False):
    """Splits a light curve on discontinuities (gaps).
    This function accepts a light curve that is either a single segment, or is
    piecewise defined (e.g. split by quarter breaks or gaps in the in the data).
    Args:
    all_time: Numpy array or list of numpy arrays; each is a sequence of time
        values.
    all_flux: Numpy array or list of numpy arrays; each is a sequence of flux
        values of the corresponding time array.
    gap_width: Minimum gap size (in time units) for a split.
    Returns:
    out_time: List of numpy arrays; the split time arrays.
    out_flux: List of numpy arrays; the split flux arrays.
    """
    
    import collections
    
    if index_only:
        idx = np.arange(len(all_time), dtype=int)
    
    # Handle single-segment inputs.
    # We must use an explicit length test on all_time because implicit conversion
    # to bool fails if all_time is a numpy array, and all_time.size is not defined
    # if all_time is a list of numpy arrays.
    if len(all_time) > 0 and not isinstance(all_time[0], collections.Iterable):  # pylint:disable=g-explicit-length-test
        
        all_time = [all_time]
        all_flux = [all_flux]

    out_time = []
    out_flux = []
    out_idx = []
    for time, flux in zip(all_time, all_flux):

        start = 0

        for end in range(1, len(time) + 1):

            splitit = 0
            # Choose the largest endpoint such that time[start:end] has no gaps.
            if end == len(time):
                splitit = 1
                
            elif (time[end] - time[end - 1] > gap_width):
                splitit = 1
                
            elif np.abs(flux[end] - flux[end - 1]) > dft:
                splitit = 1

            if splitit:
                
                if index_only:
                    out_idx.append(idx[start:end])
                else:
                    out_time.append(time[start:end])
                    out_flux.append(flux[start:end])
                start = end

    if index_only:
        return out_idx
    else:
        return out_time, out_flux
    
    
def meas_sig(time, flux, weights, plot_q=False):
    
    df0 = np.diff(flux)
    
    #df = sigma_clip(df0, sigma=8, maxiters=1)
    
    #df = sigma_clip(df, sigma=5, maxiters=2)
    
    #df = sigma_clip(df, sigma=3, maxiters=2)
    
    gi = np.ones(len(df0), dtype=bool)
    
    for sdm in [8, 5, 3]:
        
        sd = np.std(df0[gi])
        
        gi = gi & ((df0 > -sd * sdm) & (df0 < sd * sdm))
    
    
    if plot_q:
        
        x = np.copy(time[:-1])
        
        #__, gi, __ = np.intersect1d(df0, df, return_indices=1)
        
        #gi = np.sort(gi)
        
        plt.figure(figsize=(10, 4))
        plt.title(str(np.median(np.abs(df0))))
        plt.plot(x, df0, '.k', label='Orig dF')
        plt.plot(x[gi], df0[gi], '.g', label='Final dF')
        plt.legend()
        
    return np.median(np.abs(df0[gi]))
        

def meas_sig_old(time, flux, weights, plot_q=False):
    
    from astropy.stats import sigma_clip
    
    df = np.diff(flux)
    
    x = np.copy(time[:-1]) #np.arange(len(df))
    
    dff1 = csaps.UnivariateCubicSmoothingSpline(x, 
                                                df, 
                                                smooth=0.1,
                                                weights=weights[:-1],
                                                )(x)

    
#     print('dff1', dff1)
    
    dfn = df - dff1
    
    dfnf = np.copy(dfn)
    
#     plt.figure(figsize=(10, 4))
#     plt.plot(x, df, '.k', label='Orig dF')
#     plt.plot(x, dff1, '-c', label='dF fit 1')
    
#     plt.figure(figsize=(10, 4))
#     plt.plot(x, dfn, '.k', label='Orig dF')

    for sd in [5]:

        dfnf = sigma_clip(dfnf, sigma=sd, maxiters=2)

    __, gi, __ = np.intersect1d(dfn, dfnf, return_indices=1)

    gi = np.sort(gi)
    
    
#     plt.figure(figsize=(10, 4))
#     plt.plot(x[gi], dfn[gi], '.k', label='Orig dF')
    
#     try:

    dff = csaps.UnivariateCubicSmoothingSpline(x[gi], 
                                                df[gi], 
                                                smooth=0.1,
                                                weights=weights[:-1][gi],
                                                )(x)
#     except:
        
#         print('FAIL', len(x), len(gi))
        
#         #FAIL 271 270 36338 -36608
#         #FAIL 271 270 36338 -36608
        
#         return

    dfn = df - dff

    if plot_q:
        plt.figure(figsize=(10, 4))
        plt.plot(x, df, '.k', label='Orig dF')
        plt.plot(x, dff1, '-c', label='dF fit 1')
        plt.plot(x, dff, '-m', label='dF fit 2')
        plt.legend()
    
    
    for sd in [5, 3]:
        
        dfn = sigma_clip(dfn, sigma=sd, maxiters=5)

    if plot_q:
        plt.figure(figsize=(10, 7))
        plt.title('RMS = ' + str(np.std(dfn)))
        plt.plot(x, dfn, '.k')
        
    return np.std(dfn)


def filt_ssfit(time, flux, weights, gi_mask, smo, sds, return_filt_index=False, plot_q=False):
    
    from csaps import UnivariateCubicSmoothingSpline as ss
    
    flux_fit = ss(time[gi_mask], flux[gi_mask], smooth=smo, weights=weights[gi_mask])(time)

    df = flux - flux_fit

    dff = np.copy(df)

#     for sd in sds:

#         dff = sigma_clip(dff, sigma=sd, maxiters=3)

#     __, gi, __ = np.intersect1d(df, dff, return_indices=1)

#     gi = np.sort(gi)
    
#     bi = np.ones(len(time), dtype=bool)
#     bi[gi] = False
#     gi = ~bi

    gi = np.copy(gi_mask)
    
    gi_cnt_0 = np.sum(gi)

    for sdm in sds:

        sd = np.std(dff[gi])

        gi_new = ((dff > -sd * sdm) & (dff < sd * sdm))
        
        if np.sum(gi_new) < gi_cnt_0 * 0.5:
            
            print('Filter might be too strong. Attempted to filter >50% of the points in a section. Not applying this filter.')
            break
        
        gi = gi & gi_new
        
    
    # Apply mask
    #gi = gi & gi_mask
    bi = ~gi
    
    cf = ss(time[gi], flux[gi], smooth=smo, weights=weights[gi])(time)
    
    if plot_q:
        
        # Show the filter plot
        plt.figure(figsize=(10, 7))
        plt.title('Filter Stage (single)')
        plt.plot(time, flux, '.k')
        plt.plot(time[bi], flux[bi], 'xr')
        plt.plot(time, cf, '-m')
        plt.show()
        plt.pause(0.1)

    if return_filt_index:
        return cf, bi
    else:
        return cf

    
def iter_filt_ssfit(time, flux, weights, gi_mask, smo, sds, iters=3, return_filt_index=False, plot_q=False):
    
    time2, flux2 = np.copy(time), np.copy(flux)
    
    gi = np.copy(gi_mask) #ones(len(time), dtype=bool)
    
    for i in range(iters):
        
        cf, bi = filt_ssfit(time2, flux2, weights, 
                            gi, smo, sds, 
                            return_filt_index=True, 
                            plot_q=0)
        
        gi = ~bi
        
        
    if plot_q:
        
        # Show the filter plot
        plt.figure(figsize=(10, 7))
        plt.title('Filter Stage')
        plt.plot(time, flux, '.k')
        plt.plot(time[~gi_mask], flux[~gi_mask], '.', c=[0,1,0,0.8])
        plt.plot(time[~gi & gi_mask], flux[~gi & gi_mask], 'xr')
        plt.plot(time, cf, '-m')
        plt.show()
        plt.pause(0.1)
        
    if return_filt_index:
        return cf, ~gi
    else:
        return cf
    
    
    
def normlc(time, flux, smooth_params_to_try, fit_sds, time_gap_thresh, 
           mask_transit_q, t0, p, duration, 
           flux_gap_thresh=0.005, plot_q=False):
    
    
    ####################################################################################
    # Split the light curve into sections based on time and flux gaps 
    
    all_time, all_flux = split(time, flux, 
                               gap_width=time_gap_thresh, 
                               dft=flux_gap_thresh)

    print('Data broken into ' + str(len(all_time)) + ' sections')

    if plot_q:
        
        plt.figure(figsize=(10, 7))
        plt.title('Data broken into ' + str(len(all_time)) + ' sections')

        for ci in np.arange(len(all_flux)):

            plt.plot(all_time[ci], all_flux[ci], '.')
        
        plt.show()
        plt.pause(1.0)
        
        
    # Cut sections that are lest than 50 points long
    cnt = np.array([len(all_time[i]) for i in range(len(all_time))])
    
    gi = cnt > 30
    
    all_time0, all_flux0 = np.copy(all_time), np.copy(all_flux)
    
    all_time, all_flux = [], []
    for i in np.arange(len(all_time0)):
        
        if gi[i]:
            all_time.append(all_time0[i])
            all_flux.append(all_flux0[i])
        
        

    ####################################################################################
    # Fit each section
    
    smos = np.copy(smooth_params_to_try) #np.linspace(0.999, 1-1e-10, 200)
    
    cont_fit, time_norm, flux_norm, outliers = np.array([]), np.array([]), np.array([]), np.array([])
    cont_fit_list, flux_norm_list, outliers_list = [], [], []
    
    for cn in range(len(all_time)):
    #for cn in range(1):

        # Mask the transit
        if mask_transit_q:
            
            transit_idx = mask_transits(t0_bkjd=t0, period_day=p, duration_day=duration, 
                                        time_bkjd=all_time[cn], 
                                        flux=all_flux[cn], 
                                        lc=None, return_index=True)
        else:

            transit_idx = np.zeros(len(all_time[cn]), dtype=bool)
            

        weights = np.float64(~transit_idx)
        
        weights[weights==0.0] = 1e-5
        
        bics = np.ones_like(smos)

        sigma = meas_sig(all_time[cn][~transit_idx], all_flux[cn][~transit_idx], weights, plot_q=0)

        for i, smo in enumerate(smos):

            # Make the fit
            cont_fit_tmp = iter_filt_ssfit(all_time[cn], 
                                           all_flux[cn], weights, 
                                           ~transit_idx, 
                                           smo, sds=fit_sds, 
                                           return_filt_index=0, 
                                           plot_q=0)
            
            # Measure the fit reliabillity based on some metric
            bics[i] = meas_sigdif(all_time[cn][~transit_idx], 
                                  all_flux[cn][~transit_idx], 
                                  cont_fit_tmp[~transit_idx], 
                                  sigma_i=sigma)


        # Determine the best smoothing param
        bv = np.argmin(bics)

        idx = np.arange(3) - 1

        idx = bv + idx

        pf_fail = True
        if (idx[0] < 0) | (idx[-1] > len(smos)-1):
            
            bsmo = smos[bv]
            
        else:
            
            pfc = np.polyfit(smos[idx], bics[idx], 2)

            bsmo = - pfc[1] / (2 * pfc[0])
            
            pf_fail = False

        
        if 0:#plot_q:

            plt.figure(figsize=(10, 7))
            plt.title('Sigma difference results  |  Section ' + str(cn) + ' | Best smoothing param = ' + str(bsmo))
            plt.plot(smos, bics, '.k', label='Sigma')
            
            if not pf_fail:
                plt.plot(smos[idx], np.polyval(pfc, smos[idx]), '-m', label='Sigma fit')
                plt.plot(bsmo, np.polyval(pfc, bsmo), '*g', label='Best match')
                
            plt.show()
            
        
        # Make the final fit using the solved smoothing param
        cont_fit_tmp, outliers_tmp = iter_filt_ssfit(all_time[cn], 
                                                     all_flux[cn], 
                                                     ~transit_idx, 
                                                     bsmo, sds=fit_sds, 
                                                     return_filt_index=1, 
                                                     plot_q=0)
        
        
        flux_norm_tmp = all_flux[cn] / cont_fit_tmp
        
        
        cont_fit_list.append(cont_fit_tmp)
        
        flux_norm_list.append(flux_norm_tmp)
        
        outliers_list.append(outliers_tmp)
        
        
        time_norm = np.concatenate((time_norm, all_time[cn]))
        
        cont_fit = np.concatenate((cont_fit, cont_fit_tmp))
        
        flux_norm = np.concatenate((flux_norm, flux_norm_tmp))
        
        outliers = np.concatenate((outliers, outliers_tmp))
        
        
    if plot_q:
        
        # Mask the transit
        if mask_transit_q:

            transit_idx = mask_transits(t0_bkjd=t0, period_day=p, duration_day=duration, 
                                        time_bkjd=time_norm, flux=flux_norm, 
                                        lc=None, return_index=True)
        else:

            transit_idx = np.zeros(len(all_time[cn]), dtype=bool)

        plt.figure(figsize=(10, 7))
        
        #plt.plot(time_norm, flux_norm, '.k')
        plt.plot(time_norm, flux_norm, '.k')
        
        if mask_transit_q:
            plt.plot(time_norm[transit_idx], flux_norm[transit_idx], '.', 
                     c=[0,1,0,0.8], label='Transit (Area Masked During Fit)')
        
        plt.grid(1)
        plt.show()
        
    return time_norm, flux_norm, cont_fit, outliers



def rms(y):

    gi = np.isfinite(y)

    y = y[gi]

    return np.sqrt(np.mean(y ** 2))


def min_smo_OLD(smo, time, flux, weights, transit_idx, fit_sds, sigma, vs):
    
    global smo_h, lsq_h
    
    smo = smo / vs
    
    #print(smo)
    
    # Make the fit
    cont_fit_tmp = iter_filt_ssfit(time, 
                                   flux, 
                                   weights, 
                                   ~transit_idx, 
                                   smo, 
                                   sds=fit_sds, 
                                   return_filt_index=0, 
                                   plot_q=0)

    # Measure the fit reliabillity based on some metric
#     bics = meas_sigdif(time, 
#                        flux, 
#                        weights, 
#                        cont_fit_tmp, 
#                        sigma_i=sigma)

    sigma_f = rms(flux - cont_fit_tmp)
    
    bics = (sigma - sigma_f)**2
    
    smo_h.append(smo)
    lsq_h.append(bics)
    
    return bics


def min_smo(smo, time, flux, weights, transit_idx, fit_sds, sigma, vs, t_bins0):
    
#     smo_h, lsq_h, c1_h, c2_h = [], [], [], []
    
    smo = smo / vs
    
    # Make the fit
    cont_fit_tmp, bi = iter_filt_ssfit(time, 
                                       flux, 
                                       weights, 
                                       ~transit_idx, 
                                       smo, 
                                       sds=fit_sds, 
                                       return_filt_index=1, 
                                       plot_q=0)
    
    gif = ~bi
    
    t_bins = t_bins0[gif]
    
    df = flux[gif] - cont_fit_tmp[gif]
    
    sigma_f = rms(df)
    
    dtb = np.zeros(np.max(t_bins)+1, dtype=float)
    
    for i in range(np.max(t_bins)+1):
        
        gi = t_bins == i
        
        ti = df[gi] > 0
        bi = df[gi] < 0
        
        #dtb[i] = np.abs(np.sum(df[gi][ti]) + np.sum(df[gi][bi]))
        
        a = np.sum(df[gi][ti]) + 1e-6
        
        b = np.sum(df[gi][bi]) + 1e-6
        
        dtb[i] = np.abs(1 - np.abs(a / b))
        
    w1 = 100
    #c1 = (sigma - sigma_f)**2
    c1 = np.abs(1 - np.abs(sigma + 1e-6 / (sigma_f + 1e-6)))
    
    w2 = 1
    #c2 = np.sum(dtb)
    c2 = np.sum(dtb) / len(dtb)
    
    #print(c1, c2, smo, np.sum(dtb) , len(dtb))
    
    #bics = w1 * c1 + w2 * c2
    
    bics = c1 * 1.0 + c2 * 1.0 + smo * 0.5  #**1.3036
    
    #bics = c1 * c2#$* np.abs(smo - 0.5)**0.1  #**1.3036
    
    #bics = np.sum(dtb)
    
#     c1_h.append(c1)
#     c2_h.append(c2)
#     smo_h.append(smo)
#     lsq_h.append(bics)
    
    return bics