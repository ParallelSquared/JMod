"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

import numpy as np
import pandas as pd
import re
import time
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import tqdm
from functools import partial
from pyteomics import mass 
from kneed import KneeLocator

from SpecLib import load_tsv_speclib,load_tsv_lib, loadSpecLib
import load_files 
import SpecLib
import Jplot as jp
import config
import iso_functions as iso_f

from SpectraFitting import fit_to_lib
from scipy.interpolate import LSQUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
#from scipy.optimize import isotonic_regression
from statistics import quantiles
from miscFunctions import within_tol
from scipy import signal
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import auc
import warnings
import dill
dill.settings['recurse'] = True
import itertools 
import h5py
import copy

from scipy.interpolate import interp1d
import statsmodels.api as sm


from mass_tags import tag_library, mTRAQ,mTRAQ_678, mTRAQ_02468, diethyl_6plex, tag6

from miscFunctions import feature_list_mz, feature_list_rt, createTolWindows, within_tol,moving_average, \
    closest_ms1spec, closest_peak_diff,split_frag_name, unstring_floats, fragment_cor, np_pearson_cor


from FinetuneFns import fine_tune_rt, one_hot_encode_sequence
from read_output import names, dtypes

colours = ["tab:blue","tab:orange","tab:green","tab:red",
'tab:purple',
'tab:brown',
'tab:pink',
'tab:gray',
'tab:olive',
'tab:cyan']


def twostepfit(x,y,n_knots=2,z=None,k1=1):
    if z is None:
        z= np.ones_like(x)
    y_exists = np.isfinite(y)
    x_exists = np.isfinite(x)*y_exists
    x=np.array(x)[x_exists]
    y=np.array(y)[x_exists]
    z=np.array(z)[x_exists]
    y_range = np.max(y)-np.min(y)
    sorted_idxs = np.argsort(x)
    sort_x = np.array(x)[sorted_idxs]
    sort_y = np.array(y)[sorted_idxs]
    sort_z = np.array(z)[sorted_idxs]
    knots = quantiles(sort_x,n=n_knots)
    spl = spline(sort_x,sort_y,knots,w=sort_z,k=k1)
    # plt.scatter(x,y,s=1)
    # plt.scatter(x,spl(x),s=1)
    # find outliers and remove; points over 1/4 of the y range away from prediction
    _bool = abs(spl(sort_x)-sort_y)<(y_range/4)
    spl2 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
    # plt.scatter(sort_x[_bool],sort_y[_bool],c=np.log10(sort_z[_booxl]),s=1)
    # plt.scatter(sort_x[_bool],spl2(sort_x[_bool]),s=1)
    # plt.scatter(x,y,s=1)
    # plt.scatter(x,spl2(x),s=1)
    return spl2

def threestepfit(x,y,n_knots=2,z=None,k1=1):
    if z is None:
        z= np.ones_like(x)
    y_exists = np.isfinite(y)
    x_exists = np.isfinite(x)*y_exists
    x=np.array(x)[x_exists]
    y=np.array(y)[x_exists]
    z=np.array(z)[x_exists]
    y_range = np.max(y)-np.min(y)
    sorted_idxs = np.argsort(x)
    sort_x = np.array(x)[sorted_idxs]
    sort_y = np.array(y)[sorted_idxs]
    sort_z = np.array(z)[sorted_idxs]
    knots = quantiles(sort_x,n=n_knots)
    spl = spline(sort_x,sort_y,knots,w=sort_z,k=1)
    # poly = np.polyfit(sort_x, sort_y, w=sort_z, deg=5)
    # sort_x+=np.arange(len(sort_x))*1e-7
    # spl  = InterpolatedUnivariateSpline(sort_x,sort_y,w=np.log10(sort_z),k=5)
    # plt.plot(sort_x,np.polyval(poly, sort_x))
    # plt.scatter(x,y,s=1)
    # plt.scatter(x,spl(x),s=1)
    # find outliers and remove; points over 1/4 of the y range away from prediction
    _bool = abs(spl(sort_x)-sort_y)<(y_range/4)
    # knots = quantiles(sort_x,n=4)
    spl2 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
    # spl2 = UnivariateSpline(sort_x,sort_y)
    # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
    # plt.scatter(x,spl2(x),s=1)
    
    _bool = abs(spl2(sort_x)-sort_y)<(y_range/8)
    
    # knots = quantiles(sort_x,n=n_knots)
    spl3 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
    # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
    # plt.scatter(sort_x[_bool],spl3(sort_x[_bool]),s=1)
    
    return spl3


from numpy.polynomial import legendre as leg
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

def sgd_fit(x,y,z=None,deg=2,penalty="l1"):
    if z is None:
        z= np.ones_like(x)
    y_exists = np.isfinite(y)
    x_exists = np.isfinite(x)*y_exists
    x=np.array(x)[x_exists]
    y=np.array(y)[x_exists]
    z=np.array(z)[x_exists]
    
    sorted_idxs = np.argsort(x)
    sort_x = np.array(x)[sorted_idxs]
    sort_y = np.array(y)[sorted_idxs]
    sort_z = np.array(z)[sorted_idxs]
    
    
    sort_x = StandardScaler().fit_transform(sort_x[:,np.newaxis])
    V=leg.legvander(sort_x.flatten(),deg)
    
    
    sgdr = SGDRegressor(loss="huber",max_iter=10000,alpha=0.0001,penalty=penalty,learning_rate="adaptive")
    sgdr.fit(V, sort_y)
    
    # plt.scatter(sort_x,sort_y,s=1)
    # plt.scatter(sort_x,sgdr.predict(V),s=1)
    # plt.yscale("log")
    # plt.ylim(0,70)
    
    def spl(x_vals):
        V_vals = leg.legvander(x_vals,deg)
        return sgdr.predict(V_vals)
    
    return spl

def initstepfit(x,y,n_knots=2,z=None,k1=1):
    ### like above but initial guess is just a straight line from [min_x,min_] to [max_x,max_y]
    if z is None:
        z= np.ones_like(x)
    y_exists = np.isfinite(y)
    x_exists = np.isfinite(x)*y_exists
    x=np.array(x)[x_exists]
    y=np.array(y)[x_exists]
    z=np.array(z)[x_exists]
    y_range = np.max(y)-np.min(y)
    x_range = np.max(x)-np.min(x)
    sorted_idxs = np.argsort(x)
    sort_x = np.array(x)[sorted_idxs]
    sort_y = np.array(y)[sorted_idxs]
    sort_z = np.array(z)[sorted_idxs]
    knots = quantiles(sort_x,n=n_knots)
    # spl = spline(sort_x,sort_y,knots,w=sort_z,k=1)
    # plt.scatter(x,y,s=1)
    # plt.scatter(x,spl(x),s=1)
    # plt.plot(x,((y_range/x_range)*x)+min(y)-((y_range/x_range)*min(x)))
    _bool = np.abs((((y_range/x_range)*sort_x)+min(y)-((y_range/x_range)*min(x)))-sort_y)<(y_range/4)
    # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
    # find outliers and remove; points over 1/4 of the y range away from prediction
    # _bool = np.abs(spl(sort_x)-sort_y)<(y_range/4)
    # knots = quantiles(sort_x,n=4)
    spl2 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
    # spl2 = UnivariateSpline(sort_x,sort_y)
    # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
    # plt.scatter(x,spl2(x),s=1)
    
    _bool = np.abs(spl2(sort_x)-sort_y)<(y_range/8)
    
    # knots = quantiles(sort_x,n=n_knots)
    spl3 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
    # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
    # plt.scatter(sort_x[_bool],spl3(sort_x[_bool]),s=1)
    
    return spl3



def lowess_fit(x,y,frac=.2, it=3):

    # plt.scatter(x,y,s=1)
    
    lowess = sm.nonparametric.lowess(y, x, frac=frac,it=it)
    
    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]
    
    # run scipy's interpolation. There is also extrapolation I believe
    f = interp1d(lowess_x, lowess_y, bounds_error=False,fill_value=(min(lowess_y),max(lowess_y)))
    
    return f



def get_diff(mz,peaks,window,tol):
    
    log_diff = within_tol(mz, peaks, atol=0, rtol=tol) 
    idxs = np.where(log_diff[...,0])[0]
    
    
    # if a match
    if idxs.size > 0:
        
        # in case of multiple matches
        # select idx with smallest error
        closest_idx = idxs[np.argmin(log_diff[idxs,1])]
        
        return (peaks[closest_idx]-mz)/mz
    
    else:
        return np.nan
            

def lin_func(x,a,b,c):
    return (a*np.array(x[0]))+(b*np.array(x[1]))+c

def curve_param(x,y,z,func=lin_func):
    # z are predictions
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    z_exists = np.isfinite(z) # some diffs may be nans
    parameters, covariance = curve_fit(func, [x[z_exists],y[z_exists] ], z[z_exists])
    return parameters

def contains_peak(window,mz):
    return window[0]<mz<window[1]

def max_intens(peaks,window):
    
    window_peaks = peaks[np.logical_and(peaks[:,0]>window[0],peaks[:,0]<window[1])]
                                        
    if len(window_peaks)>0:
        return window_peaks[np.argmax(window_peaks[:,1])]
    else:
        return np.zeros(2)
    
        
# find largest peak in each spectrum
def get_largest(dia_spectra):
    
    ms1spectra = dia_spectra.ms1scans
    ms2spectra = dia_spectra.ms2scans
    all_windows = [tuple(i.ms1window) for i in ms2spectra]
    set_windows = sorted(set(all_windows))
    all_windows = np.stack(all_windows)
    min_mz = np.min(all_windows[:,0])
    max_mz = np.max(all_windows[:,1])
    
    # num_windows if non-overalpping
    ms2_ms1_ratio = round(len(ms2spectra)/len(ms1spectra))
    peaks_per_window = config.n_most_intense//ms2_ms1_ratio
   
    mp_window_mp_spec = np.array([np.append(np.pad([max_intens(i.peak_list().T,window) for window in all_windows[idx*ms2_ms1_ratio:(idx+1)*ms2_ms1_ratio]],((0,ms2_ms1_ratio),(0,0)))[:ms2_ms1_ratio],
                                            np.ones(ms2_ms1_ratio)[:,np.newaxis]*idx,1) 
                                  for idx,i in enumerate(ms1spectra)])
    
    top_window = np.argsort(-mp_window_mp_spec[:,:,1],axis=0)[:peaks_per_window]
    top_mzs = np.array([i[j] for i,j in zip(np.transpose(mp_window_mp_spec,[1,0,2]),top_window.T)])
    top_mzs[:,:,2] = (top_mzs[:,:,2]*ms2_ms1_ratio)+np.arange(ms2_ms1_ratio)[:,np.newaxis]
    return top_mzs.reshape(-1,3)


    
    
def closest_feature(mz,rt,dino_features,rt_tol,mz_tol):
    
    _bool = np.logical_and(dino_features.rtStart<(rt+rt_tol),(rt-rt_tol)<dino_features.rtEnd)
    
    feature_mzs = np.array(dino_features.mz[_bool])
    
    log_diff = within_tol(mz, feature_mzs, atol=0, rtol=mz_tol) 
    idxs = np.where(log_diff[...,0])[0]
    
    
    # if a match
    if idxs.size > 0:
        
        # in case of multiple matches
        # select idx with smallest error
        closest_idx = idxs[np.argmin(log_diff[idxs,1])]
        
        return (feature_mzs[closest_idx]-mz)/mz
    
    else:
        return np.nan
        
def closest_feature2(mz,rt,dino_features,rt_tol,mz_tol):
    
    _bool = np.logical_and(dino_features.rtStart<(rt+rt_tol),(rt-rt_tol)<dino_features.rtEnd)
    
    feature_mzs = np.array(dino_features.mz[_bool])
    
    log_diff = within_tol(mz, feature_mzs, atol=0, rtol=mz_tol)
    idxs = np.where(log_diff[...,0])[0]
    
    
    # if a match
    if idxs.size > 0:
        
        # in case of multiple matches
        # select idx with smallest error
        closest_idx = idxs[np.argmin(log_diff[idxs,1])]
        
        return feature_mzs[closest_idx]
    
    else:
        return np.nan
    
def get_tol(x):
    x= x[np.isfinite(x)]
    max_scale = np.max(x)
    min_scale = np.min(x)
    bin_size = (max_scale-min_scale)/300 # found to work well but not optimised
    bins = np.arange(min_scale,max_scale+bin_size,bin_size)
    vals,bins = np.histogram(x,bins)
    smooth_vals = moving_average(vals, 10) # found to work well but not optimised
    _,peak=signal.find_peaks(smooth_vals,height=max(smooth_vals),width=1,rel_height=.5)
    return(peak["widths"][0]*bin_size*2) # twice fwhm

def closest_spec(dia_rt_mzwin,mz,rt):
    
    _bool = np.logical_and(dia_rt_mzwin[:,1]<mz,mz<dia_rt_mzwin[:,2])
    contender_idxs = np.where(_bool)
    contenders = dia_rt_mzwin[_bool]
    if len(contenders)==0: ## should not happen has been observed with mismade acquisition schemes
        return 0
    closest_idx = contender_idxs[0][np.argmin(np.abs(contenders[:,0]-rt))]
    return closest_idx
    # try:
    #     closest_idx = contender_idxs[0][np.argmin(np.abs(contenders[:,0]-rt))]
    #     return closest_idx
    # except:
    #     print(mz,rt)
    #     return 0
    
def closest_spec(dia_rt_mzwin, mz, rt):
    contender_idxs = np.where((dia_rt_mzwin[:,1] < mz) & (mz < dia_rt_mzwin[:,2]))[0]
    
    if contender_idxs.size == 0:  # More efficient size check
        return 0
    
    contenders = dia_rt_mzwin[contender_idxs, 0]  # Extract only necessary column
    closest_idx = contender_idxs[np.argmin(np.abs(contenders - rt))]
    
    return closest_idx
    
def gaussian(x, amplitude, mean, stddev):
    return (amplitude/ (np.abs(stddev) * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

def fwhm(stddev):
    return 2 * np.sqrt(2 * np.log(2)) * stddev

# NB: Need to make the folowing changes to this code
## if there is a background uniform distribution, it will not fit the gaussian well
# therefore we can subtract the min val in all bins fram all bins and then fit
### This seems to work for some data but need to robustly test
def fit_gaussian(data,init_std=None,bin_n=50):
    
    data = np.array(data)
    data = data[~np.isnan(data)]
    # Create a histogram
    hist, bin_edges = np.histogram(data, bins=bin_n, density=True)
    
    ### Need to test
    # background = np.min(hist)
    # hist-=background
    
    # Find peaks in the histogram
    # peaks, _ = signal.find_peaks(hist, height=0.01, distance=10)
    peaks, _ = signal.find_peaks(hist, height=max(hist)*0.5, distance=10)
    
    # Find the highest peak
    highest_peak_index = np.argmax(hist[peaks])
    highest_peak_height = hist[peaks][highest_peak_index]
    highest_peak_x = bin_edges[peaks][highest_peak_index]
    
    # Calculate the width of the highest peak using Gaussian fit
    # split bins in 2 to get x values
    x_data = (bin_edges[:-1] + bin_edges[1:]) / 2
    y_data = hist
    
    if init_std is None:
        init_std = 2*np.subtract(*bin_edges[1::-1])
    
    fit_params, _ = curve_fit(gaussian, x_data, y_data, p0=[highest_peak_height, highest_peak_x, init_std])
    
    return fit_params#, background

from math import erf, sqrt
def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / sqrt(2.0))) / 2.0

def half_gaussian_cdf(x, mean, stddev):
    return stats.halfnorm.cdf(x, loc=mean, scale=stddev)

def exp_cdf(x, loc, mean):
    return stats.expon.cdf(x,loc=loc,scale=mean)


def fit_errors(errors,limit=10,percentile=.999):
    ### try to fit half gaussian or exponential to absolute rt errors
    
    errors_filtered = np.array(errors)[np.array(errors)<limit]
    errors_filtered = np.sort(np.append(errors_filtered,limit))
    
    p = np.arange(len(errors_filtered)) / (len(errors_filtered) - 1)
    ### MAD
    mad = np.median(np.abs(errors_filtered-np.median(errors_filtered)))
    
    #### gaussian
    g_cdf = stats.halfnorm.cdf(np.abs(errors_filtered),loc=0,scale=mad*1.4826)
    g_cdf_sq_err = np.sum(np.power(g_cdf-p,2))
    
    ## exponential
    e_cdf = stats.expon.cdf(np.abs(errors_filtered),loc=0,scale=mad/np.log(2))
    e_cdf_sq_err = np.sum(np.power(e_cdf-p,2))
    
    ### pick best and return boundary
    if e_cdf_sq_err<g_cdf_sq_err:
        scale_param = mad/np.log(2)
        boundary = stats.expon.ppf(percentile,loc=0,scale=scale_param)
        print("Fitted Exponential to RT errors")
    else:
        scale_param = mad*1.4826
        boundary = stats.halfnorm.ppf(percentile,loc=0,scale=scale_param)
        print("Fitted Gaussian to RT errors")
        
    # print(boundary)
    return boundary

    
    
# all_emp_diffs=all_emp_diffs
# all_pred_diffs = all_pred_diffs

# limit=3 ## exlcude RT diffs larger than this (outliers)
# emp_data = np.sort(np.abs(all_emp_diffs)[np.abs(all_emp_diffs) < limit])
# emp_data = np.append(emp_data,limit)
# emp_p = np.arange(len(emp_data)) / (len(emp_data) - 1)
# emp_cdf_auc = auc(emp_data,emp_p)
# pred_data = np.sort(np.abs(all_pred_diffs)[np.abs(all_pred_diffs) < limit])
# pred_data = np.append(pred_data,limit)
# pred_p = np.arange(len(pred_data)) / (len(pred_data) - 1)
# pred_cdf_auc = auc(pred_data,pred_p)

# plt.plot(emp_data,emp_p,label="Empirical RT")
# plt.legend()
# plt.xlabel("RT difference")
# plt.ylabel("Fraction of Precursors")
# emp_abs_errors_med = np.median(np.abs(all_emp_diffs-np.median(all_emp_diffs)))
# pred_abs_errors_med = np.median(np.abs(all_pred_diffs-np.median(all_pred_diffs)))


# plt.plot(pred_data,pred_p,label="Predicted RT")
# plt.plot(pred_data,stats.expon.cdf(pred_data,loc=0,scale=pred_abs_errors_med/np.log(2)))
# plt.plot(pred_data,stats.halfnorm.cdf(pred_data,loc=0,scale=np.power(pred_abs_errors_med*1.4826,1)))


# plt.plot(emp_data,emp_p,label="Empirical RT")
# plt.plot(emp_data,stats.expon.cdf(emp_data,loc=0,scale=emp_abs_errors_med/np.log(2)))
# plt.plot(emp_data,stats.halfnorm.cdf(emp_data,loc=0,scale=np.power(emp_abs_errors_med*1.4826,1)))

# plt.hist(np.power(stats.expon.cdf(pred_data,loc=0,scale=pred_abs_errors_med*2)-pred_p,2))
# plt.hist(np.power(stats.halfnorm.cdf(pred_data,loc=0,scale=np.power(pred_abs_errors_med*1.4826,1))-pred_p,2),alpha=.5)

# plt.hist(np.power(stats.expon.cdf(emp_data,loc=0,scale=emp_abs_errors_med*2)-emp_p,2))
# plt.hist(np.power(stats.halfnorm.cdf(emp_data,loc=0,scale=np.power(emp_abs_errors_med*1.4826,1))-emp_p,2),alpha=.5)


# ### laplace
# errors = all_pred_diffs
# # errors = np.random.normal(loc=0,scale=mad*1.4826/2,size=10000)
# mad = np.median(np.abs(errors))
# x=np.linspace(-2,2,100)
# plt.hist(errors,x,density=True)
# plt.plot(x,stats.laplace.pdf(x,loc=0,scale=mad/np.log(2)))
# plt.plot(x,stats.norm.pdf(x,loc=0,scale=mad*1.4826))
# plt.xlim(-2,2)



# mad = np.median(np.abs(errors))
# x=np.linspace(-2,2,100)
# plt.hist(np.abs(errors),x,density=True)
# plt.plot(x,stats.expon.pdf(x,loc=0,scale=mad/np.log(2)))
# plt.plot(x,stats.halfnorm.pdf(x,loc=0,scale=mad*1.4826))
# plt.xlim(-2,2)





##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


                                        



def MZRTfit(dia_spectra,librarySpectra,dino_features,mz_tol,ms1=False,results_folder=None,ms2=False):
    ## for testing
    # mz_tol,ms1,results_folder,ms2 = (config.ms1_tol,False,None,False) 
    # here spectra are both ms1 and ms2 
    
    config.n_most_intense_features = int(1e5) # larger than possible, essentually all
    
    # Calculate scans_per_cycle safely
    if len(dia_spectra.ms1scans) > 0:
        scans_per_cycle = max(1, round(len(dia_spectra.ms2scans)/len(dia_spectra.ms1scans)))
    else:
        scans_per_cycle = 1

    print("Intitial search")
    # print(f"Fitting the {config.n_most_intense} most intense spectra")
    
    ms1spectra = dia_spectra.ms1scans
    ms2spectra = dia_spectra.ms2scans
    
    ms1_rt = np.array([i.RT for i in ms1spectra])
    
    # Adjust partitioning based on available data
    totalIC = np.array([np.sum(i.intens) for i in ms2spectra])
    total_scans = len(totalIC)
    
    # Dynamically adjust number of partitions based on data size
    num_partition = min(10, max(1, total_scans // 10))  # At least 1 partition, at most 10
    
    if num_partition > 0 and total_scans > 0:
        # Calculate desired scans per partition
        desired_per_partition = min(total_scans // num_partition, 
                                   config.n_most_intense // num_partition)
        
        split_size = max(1, int(np.ceil(total_scans/num_partition)))
        split_tic = [totalIC[i*split_size:min(total_scans, (i+1)*split_size)] for i in range(num_partition)]
        
        # Only take as many as available in each partition
        split_top_n = []
        for idx, tics in enumerate(split_tic):
            if len(tics) > 0:  # Only process non-empty partitions
                # Take min of desired or available
                n_to_take = min(len(tics), desired_per_partition)
                if n_to_take > 0:
                    split_top_n.append((np.argsort(-tics)+(idx*split_size))[:n_to_take])
        
        if split_top_n:  # If we have any results
            top_n = np.concatenate(split_top_n)
        else:
            # Fallback if partitioning fails
            top_n = np.random.choice(np.arange(total_scans), 
                                    min(total_scans, config.n_most_intense), 
                                    replace=False)
    else:
        # Fallback for very small datasets
        top_n = np.random.choice(np.arange(total_scans), 
                                min(total_scans, config.n_most_intense), 
                                replace=False)
    
    # Safely calculate top_n_ms1
    if scans_per_cycle > 0:
        top_n_ms1 = top_n // scans_per_cycle
    else:
        top_n_ms1 = top_n

    # top_n = np.argsort(-totalIC)[:config.n_most_intense]
    top_n_ms1 = top_n//scans_per_cycle
    all_keys = list(librarySpectra)
    rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in librarySpectra.values()])

    top_n_spectra = [ms2spectra[i] for i in top_n]
    
    
    if dino_features is None:
    
        ### redefine "top_n_spectra" to evenly span Rt and m/z
        np.random.seed(0)
        #top_n = np.random.choice(np.arange(len(ms2spectra)),config.n_most_intense,replace=False)
        
        
        fit_outputs=[]
        
        frags = []
        for idx in tqdm.trange(len(top_n)):
            if ms2:
                fit_output,frag_errors = fit_to_lib(top_n_spectra[idx],
                                        library=librarySpectra,
                                        rt_mz=rt_mz,
                                        all_keys=all_keys,
                                        dino_features=None,
                                        rt_filter=False,
                                        return_frags=True,
                                        ms1_spectra = ms1spectra,
                                        frac_matched=.8
                                        )
                frags.append(frag_errors)
            else:
                fit_output = fit_to_lib(top_n_spectra[idx],
                                        library=librarySpectra,
                                        rt_mz=rt_mz,
                                        all_keys=all_keys,
                                        dino_features=None,
                                        rt_filter=False,
                                        return_frags=False,
                                        ms1_spectra = ms1spectra,
                                        frac_matched=.8
                                        )
            fit_outputs.append(fit_output)
        fit_outputs1=fit_outputs
    # """
    #################################################################################
    # """
    else:
        all_dia_rt = [i.RT for i in ms2spectra]
        all_dia_windows = np.array([i.ms1window for i in ms2spectra])
        lowest_mz = np.min(all_dia_windows,0)[0] # assume window span is constant over time
        largest_mz = np.max(all_dia_windows,0)[1]
        mz_bins = np.linspace(lowest_mz,largest_mz,6)
        
        ## remove charge 1+ features
        dino_features = dino_features[dino_features["charge"]!=1]
        dino_features = dino_features.reset_index(drop=True)
        sorted_features = np.argsort(-np.array(dino_features.intensityApex))
        sorted_mz = dino_features.mz[sorted_features]
        large_feature_indices = sorted_features[np.array(np.logical_and(sorted_mz>lowest_mz,sorted_mz<largest_mz))][:config.n_most_intense_features] 
        
        sorted_feature_mz_bins = [sorted_features[np.logical_and(sorted_mz>mz_bins[i],sorted_mz<mz_bins[i+1])] for i in range(len(mz_bins)-1)]
        large_feature_indices = [j for i in sorted_feature_mz_bins for j in i[:(config.n_most_intense_features//(len(mz_bins)-1))]]
        
        lf_rt = np.array(dino_features.rtApex[large_feature_indices])
        lf_mz = np.array(dino_features.mz[large_feature_indices])
        # print("Finding correct spectra")
        # lf_spectra = [np.argmin(np.abs(np.array(all_dia_rt)-i)) for i in lf_rt]
        dia_rt_mzwin = np.array([[i.RT,*i.ms1window] for i in ms2spectra])
        lf_spectra = [closest_spec(dia_rt_mzwin,i,j) for i,j in zip(lf_mz,lf_rt)] 
        # mz_int_n = get_largest(dia_spec,tra)
        fit_outputs2=[]
        frags = []
        for idx in tqdm.trange(len(lf_spectra)):
            if ms2:
                fit_output,frag_errors = fit_to_lib(ms2spectra[int(lf_spectra[idx])],
                                                    library=librarySpectra,
                                                    rt_mz=rt_mz,
                                                    all_keys=all_keys,
                                                    dino_features=None,
                                                    rt_filter=False,
                                                    ms1_mz=lf_mz[idx],
                                                    return_frags = True,
                                                    ms1_spectra = ms1spectra,
                                                    frac_matched=.8, ## NB: this may be selcting for smaller peptides
                                                    ms1_tol=config.ms1_tol
                                                    )
                
                frags.append(frag_errors)
            else:
                
                fit_output = fit_to_lib(ms2spectra[int(lf_spectra[idx])],
                                        library=librarySpectra,
                                        rt_mz=rt_mz,
                                        all_keys=all_keys,
                                        dino_features=None,
                                        rt_filter=False,
                                        ms1_mz=lf_mz[idx],
                                        ms1_spectra = ms1spectra,
                                        frac_matched=.8,## NB: this may be selcting for smaller peptides
                                        ms1_tol=config.ms1_tol
                                        )
            fit_outputs2.append(fit_output)
            
        fit_outputs = fit_outputs2
        top_n_spectra = [ms2spectra[i] for i in lf_spectra]
    # """
     ########################################################################
     
     
    dia_rt = []
    lib_rt = []
    output=[]
    max_ids=[]
    lc_frags_errors=[]
    lc_frags=[]
    feature_mzs = []
    feature_idxs = []
    for idx,fit_output in enumerate(fit_outputs):    
        if fit_output[0][0]!=0:
            lib_rt.append([librarySpectra[(i[3],i[4])]["iRT"] for i in fit_output])
            dia_rt.append(top_n_spectra[idx].RT)
            output.append(fit_output)
            max_id = np.argmax([i[0] for i in fit_output])
            max_ids.append(max_id)
            if ms2:
                lc_frags_errors.append(frags[idx][0][max_id])
                lc_frags.append(frags[idx][1][max_id])
            if dino_features is not None:
                feature_mzs.append(lf_mz[idx])
                feature_idxs.append(large_feature_indices[idx])
    # max_ids = [np.argmax([i[0] for i in j]) for j in output]
    ms1windows = [i.ms1window for i in top_n_spectra]
    id_keys = [(i[j][3],i[j][4]) for i,j in zip(output,max_ids)]
    id_mzs = [librarySpectra[i]["prec_mz"] for i in id_keys]
    
    # plt.hist(np.log10([i[j][0] for i,j in zip(output,max_ids)]),np.arange(1,9,.3))
    # plt.xlabel("log10(Coefficients)")
    # plt.ylabel("Frequency")
    
    # plt.scatter(dino_features.mz,dino_features.rtApex,s=.1)
    # plt.ylabel("Retention time")
    # plt.xlabel("m/z")
    
    min_int = 100#np.median([j[0] for i in output for j in i])
    
    all_id_rt = [[(i[j][3],i[j][4]),i[j][6]] for i in output for j in range(len(i)) if i[j][0]>min_int]
    all_coeff = [i[j][0] for i in output for j in range(len(i)) if i[j][0]>min_int]
    all_id_mzs = [librarySpectra[i[0]]["prec_mz"] for i in all_id_rt]
    
    all_hyper = [i[j][19] for i in output for j in range(len(i)) if i[j][0]>min_int]
    
    def max_coeff_rt(outputs):
        max_id = np.argmax([i[0] for i in outputs])
        # if outputs[0][0]==0:
        #     return np.nan
        # else:
        return librarySpectra[(outputs[max_id][3],outputs[max_id][4])]["iRT"]
    
    
    output_rts = np.array([max_coeff_rt(i) for i in output])
    output_coeff = np.array([i[j][0] for i,j in zip(output,max_ids)])
    output_hyper = np.array([i[j][19] for i,j in zip(output,max_ids)])
    all_lib_rts = np.array([librarySpectra[i[0]]["iRT"] for i in all_id_rt])
    
    output_df = pd.DataFrame([i[j] for i,j in zip(output,max_ids)],columns=names[:len(output[0][0])])
    
    all_output_df = pd.DataFrame([j for i in output for j in i],columns=names[:len(output[0][0])])
    
    frag_cosines = np.array([fragment_cor(output_df,i) for i in range(len(output_df))])
    frag_cosines_p = np.array([fragment_cor(output_df,i,fn="p") for i in range(len(output_df))])
    frag_multiply = frag_cosines*frag_cosines_p
    # plt.scatter(all_lib_rts,[i[1] for i in all_id_rt],label="Original_RT",s=1)
    output_df["frag_cosines"] = frag_cosines
    output_df["frag_cosines_p"] = frag_cosines_p
    
    frag_errors = [unstring_floats(mz) for mz in output_df.frag_errors]
    median  = np.median(np.concatenate([i for i in frag_errors]))
    output_df["med_frag_error"] = [np.median(np.abs(median-i)) for i in frag_errors]
    
    output_df["stripped_seq"]=np.array([re.sub("Decoy_","",re.sub("\(.*?\)","",i)) for i in output_df["seq"]])
    output_df["last_aa"]=[i[-1] for i in output_df.stripped_seq]
    
    if results_folder is not None:
        output_df.to_csv(results_folder+"/firstSearch.csv", index=False)
    # output_df = pd.DataFrame([j for i in output for j in i  if j[0]>min_int],columns=names[:len(output[0][0])])
    
    
    cor_filter = np.ones_like(dia_rt,dtype=bool)
    if dino_features is not None:
        cor_limit = 0.8
        hyper_cutoff = np.percentile(output_hyper,50)
        cor_filter = np.logical_and(frag_multiply>cor_limit,output_hyper>hyper_cutoff)
        feature_percentile = 0
        # for feature_percentile in [50,60,70,80,90]: 
        print("Filtering IDs from initial search")
        for feature_percentile in  range(20,80,5):
        
        ## empirically derived cutoffs
            
        
                                    
            cor_filter = np.logical_and.reduce(
                                                [output_df[feat]>np.percentile(output_df[feat],feature_percentile) for feat in ["hyperscore",
                                                                                                          "frag_cosines_p",
                                                                                                          "frag_cosines_p",
                                                                                                          "manhattan_distances",
                                                                                                          ]]
                                                +
                                                [output_df[feat]<np.percentile(output_df[feat],100-feature_percentile) for feat in [
                                                                                                                  "scribe_scores",
                                                                                                                  "gof_stats",
                                                                                                                  "max_matched_residuals",
                                                                                                                  "med_frag_error"]]
                                                                                                                )
            
            # greater_than_feat = {'hyperscore': 10,
            #                          'frag_cosines_p': 0,#0.9,
            #                          'manhattan_distances': 1.3}
            # less_than_feat = {'scribe_scores': 0.02,
            #                      'gof_stats': -1.1,
            #                      'max_matched_residuals': -2.5,
            #                      'med_frag_error': 4.2e-06}
            # cor_filter = np.logical_and.reduce(
            #                                     [output_df[feat]>greater_than_feat[feat] for feat in ["hyperscore",
            #                                                                                               "frag_cosines_p",
            #                                                                                               "frag_cosines_p",
            #                                                                                               "manhattan_distances",
            #                                                                                               ]]
            #                                     +
            #                                     [output_df[feat]<less_than_feat[feat] for feat in [
            #                                                                                                       "scribe_scores",
            #                                                                                                       "gof_stats",
            #                                                                                                       # "manhattan_distances",
            #                                                                                                       "max_matched_residuals",
            #                                                                                                       "med_frag_error"]]
            #                                                                                                     )
            # plt.scatter(output_rts[cor_filter],output_df.rt[cor_filter],s=.01)
            # print(sum(cor_filter))
            # plt.subplots()
            # plt.scatter(np.array(diffs)[cor_filter],output_df.rt[cor_filter],s=1)
            # plt.title(str(feature_percentile))
            
            # plt.subplots()
            # plt.scatter(np.array([(i-mz)/mz for i,mz in zip(feature_mzs,id_mzs)])[cor_filter],output_df.rt[cor_filter],s=1)
            # plt.title(str(feature_percentile))
            
            
            f = lowess_fit(output_rts[cor_filter],output_df.rt[cor_filter],.1)
            plt.subplots()
            plt.scatter(output_rts[cor_filter],output_df.rt[cor_filter],s=1)
            plt.scatter(output_rts[cor_filter],f(output_rts[cor_filter]),s=1)
            plt.title(str(feature_percentile))
            plt.savefig(results_folder+f"/Percentile_{str(feature_percentile)}.png",dpi=600,bbox_inches="tight")
            
            # plt.subplots()
            # f = lowess_fit(np.array([librarySpectra[i]["iRT"] for i in zip(output_df.seq,output_df.z)])[cor_filter],output_df.rt[cor_filter],.1)
            # plt.scatter(np.array([librarySpectra[i]["iRT"] for i in zip(output_df.seq,output_df.z)])[cor_filter],output_df.rt[cor_filter],s=1)
            # plt.scatter(np.array([librarySpectra[i]["iRT"] for i in zip(output_df.seq,output_df.z)])[cor_filter],
            #             f(np.array([librarySpectra[i]["iRT"] for i in zip(output_df.seq,output_df.z)])[cor_filter]),s=1)
            # plt.title(str(feature_percentile))
            
            # r_filter =np.logical_and(output_df.last_aa=="R",cor_filter)
            # fr = lowess_fit(output_rts[r_filter], output_df.rt[r_filter],.1)
            # plt.subplots()
            # plt.scatter(output_rts[r_filter],output_df.rt[r_filter],s=1)
            # x = np.linspace(min(output_rts[r_filter]),max(output_rts[r_filter]),100)
            # plt.scatter(x,fr(x),s=1)
            # plt.title(str(feature_percentile))
            
            
            # k_filter =np.logical_and(output_df.last_aa=="K",cor_filter)
            # fk = lowess_fit(output_rts[k_filter], output_df.rt[k_filter],.1)
            # # plt.subplots()
            # plt.scatter(output_rts[k_filter],output_df.rt[k_filter],s=1)
            # x = np.linspace(min(output_rts[k_filter]),max(output_rts[k_filter]),100)
            # plt.scatter(x,fk(x),s=1)
            # plt.title(str(feature_percentile))
            
         
           
            
            first_rt_diffs = (f(output_rts)-output_df.rt)
            rt_amplitude, rt_mean, rt_stddev = fit_gaussian(first_rt_diffs[cor_filter])
            first_rt_tolerance = 4*np.abs(rt_stddev)
            # rt_mean, rt_stddev = stats.norm.fit(first_rt_diffs[cor_filter])
            # vals,bins,_=plt.hist(first_rt_diffs[cor_filter],np.linspace(-10,10,100),density=True)
            # plt.title(str(feature_percentile))
            # plt.plot(np.linspace(-first_rt_tolerance,first_rt_tolerance,100),gaussian(np.linspace(-first_rt_tolerance,first_rt_tolerance,100), rt_amplitude, rt_mean, rt_stddev))
            # plt.plot(np.linspace(-first_rt_tolerance,first_rt_tolerance,100),stats.norm.pdf(np.linspace(-first_rt_tolerance,first_rt_tolerance,100),loc= rt_mean, scale=rt_stddev))
            
            # plt.vlines([-first_rt_tolerance,first_rt_tolerance],[0]*2,[max(vals)]*2)
            
            bad_IDs  = (np.abs(first_rt_diffs)>np.min([first_rt_tolerance,np.ptp(dia_rt)/5]))[cor_filter]
            outside_ratio = sum(bad_IDs)/len(bad_IDs)
            print(f"Testing Percentile: {feature_percentile}, Ratio: {np.round(outside_ratio,4)}, #IDs: {sum(cor_filter)}")
            if outside_ratio<.05 or (sum(cor_filter)-sum(bad_IDs)<800):
                break
        
        print(feature_percentile,np.round(outside_ratio,4),sum(cor_filter))
                
        
        cor_filter = np.logical_and(cor_filter,np.abs(first_rt_diffs)<first_rt_tolerance)
        # plt.scatter(output_rts[cor_filter],output_df.rt[cor_filter],s=1)
        # plt.scatter(output_rts[cor_filter],f(output_rts[cor_filter]),s=1)
        # plt.title(str(feature_percentile))
        
        # x = f(output_rts[cor_filter])-output_df.rt[cor_filter]
        # x_d = np.linspace(-10,10,100)
        # density = sum(stats.norm(xi).pdf(x_d) for xi in x)
        # # plt.subplots()
        # # plt.plot(x_d,density)
        # # plt.title(str(feature_percentile))
        # print(str(feature_percentile),min(density))
        
        
        emp_rt_spl = lowess_fit(np.array(output_rts)[cor_filter],np.array(dia_rt)[cor_filter],.1)
    else:
        hyper_cutoff = np.percentile(all_hyper,80)
        all_cor_filter = all_hyper>hyper_cutoff
        cor_filter = output_hyper>hyper_cutoff
        emp_rt_spl = initstepfit(np.array(all_lib_rts)[all_cor_filter],np.array([i[1] for i in all_id_rt])[all_cor_filter],1,z=np.array(all_hyper)[all_cor_filter])
        
    
    
    
    
    percentile = config.rt_percentile
    
    limit=3 ## exlcude RT diffs larger than this (outliers)
    
    ###############################################################
    ####### fine tuning
    ###############################################################
    
    if not config.args.use_emp_rt:
        ## filter for only a single channel for each
        print("Trying RT Prediction")
        seq_rt = {}
        for s,rt in zip(np.array(id_keys)[cor_filter],np.array(dia_rt)[cor_filter]):
            key=librarySpectra[(s[0],float(s[1]))]["seq"]
            seq_rt.setdefault(key,[])
            seq_rt[key].append(rt)
        # exclude those with ambiguity (differences between channels/charge states)
        filtered_seq_rt = {s:np.median(seq_rt[s]) for s in seq_rt if np.ptp(seq_rt[s])<1}
            
        ## use observed rt for fine_tuning
        # grouped_df = pd.DataFrame({'Stripped.Sequence':[librarySpectra[(i[0],float(i[1]))]["seq"] for i in id_keys],"RT":[i for i in np.array(dia_rt)]})[cor_filter]
        grouped_df =  pd.DataFrame({'Stripped.Sequence':[s for s in filtered_seq_rt],"RT":[filtered_seq_rt[s] for s in filtered_seq_rt]})
        data_split, models, convertor = fine_tune_rt(grouped_df,qc_plots=True,results_path=results_folder)
        
        
        # plt.scatter(output_rts[cor_filter],np.array(dia_rt)[cor_filter],label="Original_RT",s=.5,c=[len(seq_rt[librarySpectra[(s[0],float(s[1]))]["seq"]]) for s in np.array(id_keys)[cor_filter]]);plt.colorbar()
        # plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
        
        # plt.scatter(all_lib_rts,np.array(np.array([i[1] for i in all_id_rt])),label="Original_RT",s=.5,c=np.log10(all_hyper))
        # plt.scatter(all_lib_rts,rt_spl(all_lib_rts),label="Predicted_RT",s=1)
        # plt.colorbar(label="log coeff")
        # # plt.scatter(output_rts,dia_rt,label="Original_RT",s=1)
        # # plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
        # plt.xlabel("Library RT");plt.ylabel("Observed RT");
        
        # plt.scatter(all_lib_rts,[i[1] for i in all_id_rt],label="Original_RT",s=1)
        # plt.scatter(all_lib_rts,rt_spl(all_lib_rts),label="Predicted_RT",s=1)
        
        
        all_emp_diffs = (emp_rt_spl(output_rts)-np.array(dia_rt))[cor_filter]
        
        
        lib_seqs = [one_hot_encode_sequence(librarySpectra[key]["seq"]) for key in id_keys]
        predicted_rts = convertor(np.mean([model.predict(np.array(lib_seqs)) for model in models],axis=0).flatten())
    
        validation_rts = convertor(np.mean([model.predict(np.array(data_split[1])) for model in models],axis=0).flatten())
        validation_rt_diffs = data_split[3]-validation_rts
        
        pred_rt_spl = lowess_fit(predicted_rts[cor_filter],
                               np.array(dia_rt)[cor_filter] ,frac=.2)
        # plt.scatter(new_lib_rts[cor_filter],np.array(dia_rt)[cor_filter],s=1)
        # plt.scatter(new_lib_rts[cor_filter],pred_rt_spl(new_lib_rts[cor_filter]),s=1)
        
        all_pred_diffs = (pred_rt_spl(predicted_rts) - np.array(dia_rt))[cor_filter]
        
        all_pred_diffs = validation_rt_diffs
        
        
        emp_data = np.sort(np.abs(all_emp_diffs)[np.abs(all_emp_diffs) < limit])
        emp_data = np.append(emp_data,limit)
        emp_p = np.arange(len(emp_data)) / (len(emp_data) - 1)
        emp_cdf_auc = auc(emp_data,emp_p)
        pred_data = np.sort(np.abs(all_pred_diffs)[np.abs(all_pred_diffs) < limit])
        pred_data = np.append(pred_data,limit)
        pred_p = np.arange(len(pred_data)) / (len(pred_data) - 1)
        pred_cdf_auc = auc(pred_data,pred_p)
        
        
        
        # plt.plot(emp_data,emp_p,label="Empirical RT",color=colours[0])
        # plt.plot(pred_data,pred_p,label="Predicted RT",color=colours[1])
        # plt.legend()
        # plt.xlabel("RT difference")
        # plt.ylabel("Fraction of Precursors")
        
        # emp_abs_errors_med = np.median(np.abs(all_emp_diffs[all_emp_diffs<limit]-np.median(all_emp_diffs[all_emp_diffs<limit])))
        # pred_abs_errors_med = np.median(np.abs(all_pred_diffs[all_pred_diffs<limit]-np.median(all_pred_diffs[all_pred_diffs<limit])))
        # plt.plot(emp_data,stats.expon.cdf(emp_data,loc=0,scale=emp_abs_errors_med/np.log(2)),label="Emp Expon",linestyle="--",color=colours[0])
        # plt.scatter([stats.expon.ppf(.999,scale=emp_abs_errors_med/np.log(2))], [.999],c=colours[0])
        # plt.plot(emp_data,stats.halfnorm.cdf(emp_data,loc=0,scale=np.power(emp_abs_errors_med*1.4826,1)),label="Emp Gauss",linestyle=":",color=colours[0])
        # plt.scatter([stats.halfnorm.ppf(.999,scale=emp_abs_errors_med*1.4826)], [.999],c=colours[0])
        # plt.plot(pred_data,stats.expon.cdf(pred_data,loc=0,scale=pred_abs_errors_med/np.log(2)),label="Pred Expon",linestyle="--",color=colours[1])
        # plt.scatter([stats.expon.ppf(.999,scale=pred_abs_errors_med/np.log(2))], [.999],c=colours[1])
        # plt.plot(pred_data,stats.halfnorm.cdf(pred_data,loc=0,scale=np.power(pred_abs_errors_med*1.4826,1)),label="Pred Gauss",linestyle=":",color=colours[1])
        # plt.scatter([stats.halfnorm.ppf(.999,scale=pred_abs_errors_med*1.4826)], [.999],c=colours[1])
        # plt.legend()
        # plt.xlim(0,1)
        # plt.hist(np.power(stats.expon.cdf(pred_data,loc=0,scale=pred_abs_errors_med*2)-pred_p,2))
        # plt.hist(np.power(stats.halfnorm.cdf(pred_data,loc=0,scale=np.power(pred_abs_errors_med*1.4826,1))-pred_p,2),alpha=.5)
        
        # vals,bins,_=plt.hist(np.power(stats.expon.cdf(emp_data,loc=0,scale=emp_abs_errors_med/np.log(2))-emp_p,2),100,alpha=.5)
        # plt.hist(np.power(stats.halfnorm.cdf(emp_data,loc=0,scale=np.power(emp_abs_errors_med*1.4826,1))-emp_p,2),bins,alpha=.5)
        
        
    
        ###Apply the KneeLocator method to find the elbow for empirical CDF
        # kneedle_emp = KneeLocator(emp_data, emp_p, curve="concave", direction="increasing",S=25)
        # elbow_emp_x = kneedle_emp.knee
        # elbow_emp_y = emp_p[np.argmin(np.abs(emp_data - elbow_emp_x))]
        
    
        ###Apply the KneeLocator method to find the elbow for empirical CDF
        # kneedle_emp = KneeLocator(emp_data, emp_p, curve="concave", direction="increasing",S=25)
        # elbow_emp_x = kneedle_emp.knee
        # elbow_emp_y = emp_p[np.argmin(np.abs(emp_data - elbow_emp_x))]
        
    
        
        #plt.show()
        
        
        updatedLibrary = copy.deepcopy(librarySpectra)
        all_lib_keys = list(librarySpectra)
        
        if pred_cdf_auc>emp_cdf_auc: ## Predictions are better
            # boundary = elbow_pred_x
            print("Fine Tuned Library Chosen")
            boundary = fit_errors(all_pred_diffs,limit,percentile)
            rt_spl = pred_rt_spl
            all_lib_seqs = [one_hot_encode_sequence(updatedLibrary[key]["seq"]) for key in all_lib_keys]
            all_new_lib_rts = convertor(np.mean([model.predict(np.array(all_lib_seqs)) for model in models],axis=0).flatten())
            
            for key,rt in zip(all_lib_keys,all_new_lib_rts):
                updatedLibrary[key]["iRT"] = rt
                
        else: ### empirical are better
            # boundary = elbow_emp_x
            print("Empirical Library Chosen")
            boundary = fit_errors(all_emp_diffs,limit,percentile)
            ## keep the library RTs and splines the same
            rt_spl = emp_rt_spl
        
    else:

        print("Using Empirical w/o Fine Tuning")
        updatedLibrary = copy.deepcopy(librarySpectra)
        all_lib_keys = list(librarySpectra)
        rt_spl = emp_rt_spl
        all_emp_diffs = (emp_rt_spl(output_rts)-np.array(dia_rt))[cor_filter]
        
        
        emp_data = np.sort(np.abs(all_emp_diffs)[np.abs(all_emp_diffs) < limit])
        emp_data = np.append(emp_data,limit)
        emp_p = np.arange(len(emp_data)) / (len(emp_data) - 1)
        emp_cdf_auc = auc(emp_data,emp_p)
        boundary = fit_errors(all_emp_diffs,limit,percentile)
    
    new_lib_rt = np.array([updatedLibrary[k]["iRT"] for k in id_keys])
    converted_rt = rt_spl([updatedLibrary[k]["iRT"] for k in id_keys])
    
    rt_amplitude, rt_mean, rt_stddev = fit_gaussian((dia_rt-converted_rt)[cor_filter])
    # rt_spl = twostepfit(all_lib_rts,[i[1] for i in all_id_rt]) # does not work
    
    # mz_spl = twostepfit(id_mzs, diffs)
    # mz_spl = twostepfit(output_rts, diffs)
    
    ## old version (failed)
    # diffs = [get_diff(mz, ms1spectra[ms1_idx].mz, window, mz_tol) for  mz,ms1_idx,window in zip(id_mzs,top_n_ms1,ms1windows)]
    
    
    # all_diffs = [closest_feature(all_id_mzs[i],all_id_rt[i][1],dino_features,1,20*1e-6) for i in range(len(all_id_mzs))]
    # all_coeffs = [closest_feature(all_id_mzs[i],all_id_rt[i][1],dino_features,1,10*1e-6) for i in range(len(all_id_mzs))]
    
    if dino_features is None:
        resp_ms1scans = [closest_ms1spec(dia_rt[i], ms1_rt) for i in range(len(dia_rt))]
        diffs = [closest_peak_diff(mz, ms1spectra[i].mz) for i,mz in zip(resp_ms1scans,id_mzs)]
    else:
        diffs = np.array([(i-mz)/mz for i,mz in zip(feature_mzs,id_mzs)])
    # # diffs = [closest_feature(id_mzs[i],dia_rt[i],dino_features,0,10*1e-6) for i in range(len(id_mzs))]
    
    # # feature_mz = [closest_feature2(id_mzs[i],converted_rt[i],dino_features,1,30*1e-6) for i in range(len(id_mzs))]
    # mz_spl = twostepfit(np.array(id_mzs)[cor_filter],np.array(diffs)[cor_filter],1)
    # # mz_spl = twostepfit(id_mzs,diffs,1,z=np.log10(output_coeff))
    
    # # plt.scatter(all_id_mzs,[i for i in all_diffs],label="Original_RT",s=1)


    # def mz_func(mz):
    #     return mz+(mz_spl(mz)*mz)
    
    # mz_amplitude, mz_mean, mz_stddev = fit_gaussian((diffs-mz_spl(id_mzs))[cor_filter])
    
    
    ################################################
    ########### correct mz errors wrt RT    ########
    ################################################
    
    
    f_rt_mz = lowess_fit(new_lib_rt[cor_filter],np.array(diffs)[cor_filter],.02)
    # plt.subplots()
    # plt.scatter(new_lib_rt[cor_filter],np.array(diffs)[cor_filter],label="Original_MZ",s=.1,alpha=1)
    # plt.scatter(new_lib_rt[cor_filter],f_rt_mz(new_lib_rt)[cor_filter],s=1,alpha=.2)
    # plt.xlim(10,42);plt.ylim(-1.5e-5,1.5e-5)
    # plt.scatter(id_mzs,diffs,label="Original_MZ",s=1,alpha=.1)
    # plt.scatter(id_mzs,diffs-f_rt_mz(dia_rt),label="Original_MZ",s=1,alpha=.1)
    
    # mz_spl = twostepfit(np.array(id_mzs)[rt_filter_bool],(diffs-f_rt_mz(dia_rt))[r t_filter_bool],1)
    mz_spl = lowess_fit(np.array(id_mzs)[cor_filter],(diffs-f_rt_mz(new_lib_rt))[cor_filter])
    # plt.subplots()
    # plt.scatter(np.array(id_mzs)[cor_filter],(diffs-f_rt_mz(new_lib_rt))[cor_filter],label="Original_MZ",s=1,alpha=1)
    # plt.scatter(id_mzs,mz_spl(id_mzs),label="Original_MZ",s=1,alpha=.1)
    # plt.hlines(0,400,900)

    def mz_func(mz,rt):
        return mz+((mz_spl(mz)+f_rt_mz(rt))*mz)
    
    # orig_mzs = id_mzs+(diffs*np.array(id_mzs))
    # plt.hist(((mz_func(id_mzs,rts)-orig_mzs)/id_mzs)[rt_filter_bool],100)
    
    corrected_mz_diffs = (diffs-(f_rt_mz(new_lib_rt)+mz_spl(id_mzs)))[cor_filter]
    mz_amplitude, mz_mean, mz_stddev = fit_gaussian(corrected_mz_diffs)
    
    ### MS2 alignment
    if ms2:
        all_frag_errors = np.concatenate(lc_frags_errors)
        all_frags = np.concatenate(lc_frags)
        ms2_spl = twostepfit(all_frags,all_frag_errors,1)
        def ms2_func(mz):
            return mz+(ms2_spl(mz)*mz)
        
        ms2_amplitude, ms2_mean, ms2_stddev = fit_gaussian(all_frag_errors-ms2_spl(all_frags))
    
    # plt.hist(diffs,50)
    # plt.subplots()
    # plt.scatter(id_mzs,np.array(diffs)*1e6,s=1)
    # plt.xlabel("m/z")
    # plt.ylabel("m/z error (ppm)")
    # plt.subplots()
    # plt.scatter(dia_rt,np.array(diffs)*1e6,s=1)
    # plt.xlabel("RT")
    # plt.ylabel("m/z error (ppm)")
    
    # ## 2D plane fitting
    # function = lin_func
    # parameters = curve_param(output_rts, id_mzs, diffs,func=function)
    
    # def mz_func(mz,rt):
    #     return mz+(function([rt,mz],*parameters)*mz)
    
    # set optimised rt tol to 95th percentile of search (assumes a few outliers)
    # buffer = 1.2
    # config.opt_rt_tol = np.round(np.sort(np.abs(dia_rt-rt_spl(output_rts)))[int(config.n_most_intense*.95)]*buffer,5) 
    
    # new_rt_tol = get_tol(dia_rt-rt_spl(output_rts))
    new_rt_tol = boundary#4*np.abs(rt_stddev) 
    if config.args.user_rt_tol:
        print("Using user specified RT tolerance")
        new_rt_tol = config.args.rt_tol
    print(f"Optimsed RT tolerance: {new_rt_tol}")
    config.opt_rt_tol = np.abs(new_rt_tol)
    
    # set optimised ms2 tol
    # is_real = ~np.isnan(diffs)
    # buffer = 1.2
    # config.opt_ms1_tol = np.round(
    #                         np.sort(
    #                             np.abs(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs
    #                                    )[is_real])[int(sum(is_real)*.95)]*buffer,6+5)#6 for 1e-6 the 5 decimal places

    # new_ms1_tol = get_tol(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs)
    # new_ms1_tol = get_tol(diffs-mz_spl(id_mzs))
    new_ms1_tol = np.abs(4*mz_stddev)
    print(f"Optimsed ms1 tolerance: {new_ms1_tol}")
    
    
    if config.args.ms1_ppm!=0:
        print(f"Using MS1 Tolerance provided: {config.args.ms1_ppm}ppm")
        new_ms1_tol=np.abs(config.args.ms1_ppm*1e-6)
    elif config.min_ms1_tol!=0 and config.min_ms1_tol>new_ms1_tol:
        print(f"Exceeded minimum MS1 tolerance: {np.abs(config.min_ms1_tol)}")
        print(f"Setting new MS1 tolerance: {np.abs(config.min_ms1_tol)}")
        new_ms1_tol=np.abs(config.min_ms1_tol)
        
    config.opt_ms1_tol  = new_ms1_tol
    
    if ms2:
        new_ms2_tol = 5*ms2_stddev
        config.opt_ms2_tol  = new_ms2_tol
    
    if results_folder is not None:
        
        ### Save functions
        with open(results_folder+"/rt_spl","wb") as dill_file:
            dill.dump(rt_spl,dill_file)
            
        with open(results_folder+"/mz_func","wb") as dill_file:
            dill.dump(mz_func,dill_file)
        
        if ms2:
            with open(results_folder+"/ms2_func","wb") as dill_file:
                dill.dump(ms2_func,dill_file)
            
        ##plot RT alignment
        plt.subplots()
        plt.scatter(output_rts[cor_filter],np.array(dia_rt)[cor_filter],label="Original_RT",s=.1)
        plt.scatter(output_rts,emp_rt_spl(output_rts),label="Predicted_RT",s=1)
        # plt.legend()
        plt.xlabel("Library RT")
        plt.ylabel("Observed RT")
        # plt.show()
        plt.savefig(results_folder+"/OriginalRTfit.png",dpi=600,bbox_inches="tight")
        
        
        ##plot RT alignment
        plt.subplots()
        plt.scatter(np.array([updatedLibrary[k]["iRT"] for k in id_keys])[cor_filter],np.array(dia_rt)[cor_filter],label="Original_RT",s=.1)
        plt.scatter([updatedLibrary[k]["iRT"] for k in id_keys],rt_spl([updatedLibrary[k]["iRT"] for k in id_keys]),label="Predicted_RT",s=1)
        # plt.legend()
        plt.xlabel("Updated Library RT")
        plt.ylabel("Observed RT")
        # plt.show()
        plt.savefig(results_folder+"/RTfit.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        plt.scatter(np.array([updatedLibrary[k]["iRT"] for k in id_keys])[cor_filter],
                    (dia_rt-rt_spl([updatedLibrary[k]["iRT"] for k in id_keys]))[cor_filter],label="Original_RT",s=.1)
        min_rt = np.min([updatedLibrary[k]["iRT"] for k in id_keys])
        max_rt = np.max([updatedLibrary[k]["iRT"] for k in id_keys])
        plt.plot([min_rt,max_rt],[0,0],color="r",linestyle="--",alpha=.5)
        plt.plot([min_rt,max_rt],[config.opt_rt_tol,config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
        plt.plot([min_rt,max_rt],[-config.opt_rt_tol,-config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
        # plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
        # plt.legend()
        plt.ylim(-15,15)
        plt.xlabel("Updated Library RT")
        plt.ylabel("RT Residuals")
        # plt.show()
        plt.savefig(results_folder+"/RtResidual.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        vals,bins,_ = plt.hist((dia_rt-emp_rt_spl(output_rts))[cor_filter],100,density=True,alpha=.5,label="Original RT")
        vals,bins,_ = plt.hist((dia_rt-rt_spl([updatedLibrary[k]["iRT"] for k in id_keys]))[cor_filter],100,density=True,alpha=.5,label="Updated RT")
        plt.plot(np.linspace(-config.opt_rt_tol,config.opt_rt_tol,100),gaussian(np.linspace(-config.opt_rt_tol,config.opt_rt_tol,100), rt_amplitude, rt_mean, rt_stddev),label="Updated RT fit")
        plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals),color="r")
        # plt.vlines([-4*rt_stddev,4*rt_stddev],0,max(vals),color="g")
        plt.text(config.opt_rt_tol,max(vals),np.round(config.opt_rt_tol,2))
        plt.xlabel("RT difference")
        plt.ylabel("Frequency")
        plt.legend()
        # plt.show()
        plt.savefig(results_folder+"/RTdiff.png",dpi=600,bbox_inches="tight")
        
        
        

        
        ### Plot the CDFs with elbow points
        
        plt.subplots()
        plt.figure(figsize=(8, 5))
        plt.plot(emp_data, emp_p, label="Original CDF", linestyle='-')
        
        # plt.scatter(elbow_emp_x, elbow_emp_y, color='blue', label=f'Original Elbow at {elbow_emp_x:.2f}', zorder=3)
        # plt.scatter(elbow_pred_x, elbow_pred_y, color='red', label=f'Finetuned Elbow at {elbow_pred_x:.2f}', zorder=3)
        
        
        emp_abs_errors_med = np.median(np.abs(all_emp_diffs[all_emp_diffs<limit]-np.median(all_emp_diffs[all_emp_diffs<limit])))

        plt.plot(emp_data,stats.expon.cdf(emp_data,loc=0,scale=emp_abs_errors_med/np.log(2)),linestyle="--",color=colours[0],label="Emp Expon CDF")
        emp_exp_999 = stats.expon.ppf(percentile,scale=emp_abs_errors_med/np.log(2))
        plt.scatter([emp_exp_999], [percentile],c=colours[0],label=f"Emp Expon {percentile}: {emp_exp_999:.2f}",marker="*")
        plt.plot(emp_data,stats.halfnorm.cdf(emp_data,loc=0,scale=np.power(emp_abs_errors_med*1.4826,1)),linestyle=":",color=colours[0],label="Emp Norm CDF")
        emp_gauss_999 = stats.halfnorm.ppf(percentile,scale=emp_abs_errors_med*1.4826)
        plt.scatter([emp_gauss_999], [percentile],c=colours[0],label=f"Emp Norm {percentile}: {emp_gauss_999:.2f}")
        if not config.args.use_emp_rt:
            plt.plot(pred_data, pred_p, label="Finetuned CDF", linestyle='-')
            pred_abs_errors_med = np.median(np.abs(all_pred_diffs[all_pred_diffs<limit]-np.median(all_pred_diffs[all_pred_diffs<limit])))
            plt.plot(pred_data,stats.expon.cdf(pred_data,loc=0,scale=pred_abs_errors_med/np.log(2)),linestyle="--",color=colours[1],label="Pred Exp CDF")
            pred_exp_999 = stats.expon.ppf(percentile,scale=pred_abs_errors_med/np.log(2))
            plt.scatter([pred_exp_999], [percentile],c=colours[1],label=f"Pred Expon {percentile}: {pred_exp_999:.2f}",marker="*")
            plt.plot(pred_data,stats.halfnorm.cdf(pred_data,loc=0,scale=np.power(pred_abs_errors_med*1.4826,1)),linestyle=":",color=colours[1],label="Pred Norm CDF")
            pred_gauss_999 = stats.halfnorm.ppf(percentile,scale=pred_abs_errors_med*1.4826)
            plt.scatter([pred_gauss_999], [percentile],c=colours[1],label=f"Pred Norm {percentile}: {pred_gauss_999:.2f}")
        
        
        plt.vlines(boundary,0,1,colors="r",linestyle="--",label="Boundary")
        
        plt.xlabel("RT Differences")
        plt.ylabel("Cumulative Probability")
        plt.legend()
        plt.title("Finding an optimal RT library")
        plt.savefig(results_folder+"/RTelbows.png",dpi=600,bbox_inches="tight")
        
        
        ##plot mz rt alignment
        plt.subplots()
        plt.scatter(new_lib_rt[cor_filter],np.array(diffs)[cor_filter],label="Original_MZ",s=1,alpha=min(1,5/((len(np.array(dia_rt)[cor_filter])//1000)+1)))
        plt.scatter(new_lib_rt,f_rt_mz(new_lib_rt),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("Updated RT")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        plt.savefig(results_folder+"/MZrtfit.png",dpi=600,bbox_inches="tight")
        
        

        ##plot mz alignment
        plt.subplots()
        plt.scatter(np.array(id_mzs)[cor_filter],(diffs-f_rt_mz(new_lib_rt))[cor_filter],label="Original_MZ",s=1,alpha=min(1,5/((len(np.array(new_lib_rt)[cor_filter])//1000)+1)))
        plt.scatter(id_mzs,mz_spl(id_mzs),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("m/z")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        plt.savefig(results_folder+"/MZfit.png",dpi=600,bbox_inches="tight")
        
        
        
        ## plot mz diff
        plt.subplots()
        plt.hist(np.array(diffs)[cor_filter],100)
        # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs,100,alpha=.5)
        # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
        vals,bins,_ = plt.hist((diffs-mz_spl(id_mzs)-f_rt_mz(new_lib_rt))[cor_filter],100,alpha=.5)
        plt.vlines([-config.opt_ms1_tol,config.opt_ms1_tol],0,max(vals)*.8,color="r")
        # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
        plt.text(config.opt_ms1_tol,max(vals)*.8,f"{np.round(1e6*config.opt_ms1_tol,2)} ppm")
        plt.xlabel("m/z difference (relative)")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(results_folder+"/MZdiff.png",dpi=600,bbox_inches="tight")
    
    
        if ms2:
            ##plot mz alignment
            plt.subplots()
            plt.scatter(all_frags,all_frag_errors,label="Original_MS2",s=1)
            plt.scatter(all_frags,ms2_spl(all_frags),label="Predicted_MS2",s=1)
            # plt.legend()
            plt.xlabel("m/z")
            plt.ylabel("m/z difference (relative)")
            # plt.show()
            plt.savefig(results_folder+"/MS2fit.png",dpi=600,bbox_inches="tight")
            
            
            ## plot mz alignment
            plt.subplots()
            plt.hist(all_frag_errors,100)
            # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs,100,alpha=.5)
            # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
            plt.hist(all_frag_errors-ms2_spl(all_frags),100,alpha=.5)
            plt.vlines([-config.opt_ms2_tol,config.opt_ms2_tol],0,50,color="r")
            # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
            plt.text(config.opt_ms2_tol,50,np.round(config.opt_ms2_tol,2))
            plt.xlabel("m/z difference (relative)")
            plt.ylabel("Frequency")
            # plt.show()
            plt.savefig(results_folder+"/MS2diff.png",dpi=600,bbox_inches="tight")
    # plt.scatter(id_mzs,diffs)
    # plt.scatter(output_rts,diffs)
    plt.close("all")
    
    if ms2:
        return (rt_spl, mz_func, ms2_func), updatedLibrary
    else:
        return (rt_spl, mz_func), updatedLibrary

###################################################################################################
###################################################################################################
###################################################################################################

def filter_rts_by_dense(rts,n_bins=20):
    """
    Input list of RTs
    Where they are not dense is probably FPs
    return bool of those from dense region
    """
    hist,bins = np.histogram(rts,n_bins)
    med = np.mean(hist[hist!=0])
    where_larger = np.where(hist>med/2)[0]
    smallest,largest = (bins[where_larger[0]],bins[min(where_larger[-1]+1,len(bins)-1)])
    return np.logical_and(np.greater(rts,smallest),np.less(rts,largest))

def MZRTfit_timeplex(dia_spectra,librarySpectra,dino_features,mz_tol,ms1=False,results_folder=None,ms2=False):
    ## for testing
    # mz_tol,ms1,results_folder,ms2 = (config.ms1_tol,False,None,False)
    # here spectra are both ms1 and ms2 
    
    config.n_most_intense_features = int(1e8) # larger than possible, essentually all
    
    scans_per_cycle = round(len(dia_spectra.ms2scans)/len(dia_spectra.ms1scans))
    print("Intitial search")
    # print(f"Fitting the {config.n_most_intense} most intense spectra")
    
    ms1spectra = dia_spectra.ms1scans
    ms2spectra = dia_spectra.ms2scans
    
    ### array of all MS1 RTs
    ms1_rt = np.array([i.RT for i in ms1spectra])
    
    totalIC = np.array([np.sum(i.intens) for i in ms2spectra])
    
    top_n = np.argsort(-totalIC)[:config.n_most_intense]
    top_n_ms1 = top_n//scans_per_cycle
    all_keys = list(librarySpectra)
    rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in librarySpectra.values()])

    
    
    #################################################################################
    all_dia_rt = [i.RT for i in ms2spectra]
    all_dia_windows = np.array([i.ms1window for i in ms2spectra])
    lowest_mz = np.min(all_dia_windows,0)[0] # assume window span is constant over time
    largest_mz = np.max(all_dia_windows,0)[1]
    mz_bins = np.linspace(lowest_mz,largest_mz,6)
    
    
    ## sort identified peptide features by intensity
    sorted_features = np.argsort(-np.array(dino_features.intensityApex))
    sorted_mz = dino_features.mz[sorted_features]
    large_feature_indices = sorted_features[np.array(np.logical_and(sorted_mz>lowest_mz,sorted_mz<largest_mz))][:config.n_most_intense_features] 
    
    # sorted_feature_mz_bins = [sorted_features[np.logical_and(sorted_mz>mz_bins[i],sorted_mz<mz_bins[i+1])] for i in range(len(mz_bins)-1)]
    # large_feature_indices = [j for i in sorted_feature_mz_bins for j in i[:(config.n_most_intense_features//(len(mz_bins)-1))]]
    
    lf_rt = np.array(dino_features.rtApex[large_feature_indices])
    lf_mz = np.array(dino_features.mz[large_feature_indices])
    ## order bt Rt so first column is first in list
    rt_order = np.argsort(lf_rt)
    lf_rt = lf_rt[rt_order]
    lf_mz = lf_mz[rt_order]
    print("Finding correct spectra")
    # lf_spectra = [np.argmin(np.abs(np.array(all_dia_rt)-i)) for i in lf_rt]
    dia_rt_mzwin = np.array([[i.RT,*i.ms1window] for i in ms2spectra])
    lf_spectra = [closest_spec(dia_rt_mzwin,i,j) for i,j in zip(lf_mz,lf_rt)] 
    # mz_int_n = get_largest(dia_spec,tra)
    print("Searching largest Features")
    fit_outputs2=[]
    frags = []
    for idx in tqdm.trange(len(lf_spectra)):
        if ms2:
            fit_output,frag_errors = fit_to_lib(ms2spectra[int(lf_spectra[idx])],
                                                library=librarySpectra,
                                                rt_mz=rt_mz,
                                                all_keys=all_keys,
                                                dino_features=None,
                                                rt_filter=False,
                                                ms1_mz=lf_mz[idx],
                                                return_frags = True,
                                                ms1_spectra = ms1spectra
                                                )
            
            frags.append(frag_errors)
        else:
            
            fit_output = fit_to_lib(ms2spectra[int(lf_spectra[idx])],
                                    library=librarySpectra,
                                    rt_mz=rt_mz,
                                    all_keys=all_keys,
                                    dino_features=None,
                                    rt_filter=False,
                                    ms1_mz=lf_mz[idx],
                                    ms1_spectra = ms1spectra
                                    )
        fit_outputs2.append(fit_output)
        
    fit_outputs = fit_outputs2
    top_n_spectra = [ms2spectra[i] for i in lf_spectra]
     ########################################################################
     
     
    dia_rt = []
    lib_rt = []
    output=[]
    max_ids=[]
    lc_frags_errors=[]
    lc_frags=[]
    feature_mzs = []
    for idx,fit_output in enumerate(fit_outputs):    
        if fit_output[0][0]!=0:
            lib_rt.append([librarySpectra[(i[3],i[4])]["iRT"] for i in fit_output])
            dia_rt.append(top_n_spectra[idx].RT)
            output.append(fit_output)
            max_id = np.argmax([i[0] for i in fit_output])
            max_ids.append(max_id)
            if ms2:
                lc_frags_errors.append(frags[idx][0][max_id])
                lc_frags.append(frags[idx][1][max_id])
            if dino_features is not None:
                feature_mzs.append(lf_mz[idx])
    # max_ids = [np.argmax([i[0] for i in j]) for j in output]
    ms1windows = [i.ms1window for i in top_n_spectra]
    id_keys = [(i[j][3],i[j][4]) for i,j in zip(output,max_ids)]
    id_keys_clean = id_keys#[(re.sub("\(tag6-\d\)","",i[j][3]),i[j][4]) for i,j in zip(output,max_ids)]
    id_mzs = [librarySpectra[i]["prec_mz"] for i in id_keys]
    
    
    all_id_rt = [[(i[j][3],i[j][4]),i[j][6]] for i in output for j in range(len(i)) if i[j][0]>10]
    all_coeff = [i[j][0] for i in output for j in range(len(i)) if i[j][0]>10]
    all_id_frac_lib = [i[j][8] for i in output for j in range(len(i)) if i[j][0]>10]
    all_id_mzs = [librarySpectra[i[0]]["prec_mz"] for i in all_id_rt]
    
    
    def max_coeff_rt(outputs):
        max_id = np.argmax([i[0] for i in outputs])
        # if outputs[0][0]==0:
        #     return np.nan
        # else:
        return librarySpectra[(outputs[max_id][3],outputs[max_id][4])]["iRT"]
    
    
    output_rts = np.array([max_coeff_rt(i) for i in output])
    output_hyper = np.array([i[j][19] for i,j in zip(output,max_ids)])
    output_coeff = np.array([i[j][0] for i,j in zip(output,max_ids)])
    output_frac_lib = np.array([i[j][8] for i,j in zip(output,max_ids)])
    output_seqs = np.array([i[j][3:5] for i,j in zip(output,max_ids)])
    all_lib_rts = np.array([librarySpectra[i[0]]["iRT"] for i in all_id_rt])
    
    output_df = pd.DataFrame([i[j] for i,j in zip(output,max_ids)],columns=names[:len(output[0][0])])
    if results_folder is not None:
        output_df.to_csv(results_folder+"/firstSearch.csv", index=False)
    # output_df = pd.DataFrame([j for i in output for j in i  if j[0]>min_int],columns=names[:len(output[0][0])])
    
    frag_cosines = np.array([fragment_cor(output_df,i) for i in range(len(output_df))])
    frag_cosines_p = np.array([fragment_cor(output_df,i,fn="p") for i in range(len(output_df))])
    frag_multiply = frag_cosines*frag_cosines_p
    # plt.scatter(all_lib_rts,[i[1] for i in all_id_rt],label="Original_RT",s=1)
    output_df["frag_cosines"] = frag_cosines
    output_df["frag_cosines_p"] = frag_cosines_p
    
    frag_errors = [unstring_floats(mz) for mz in output_df.frag_errors]
    median  = np.median(np.concatenate([i for i in frag_errors]))
    output_df["med_frag_error"] = [np.median(np.abs(median-i)) for i in frag_errors]
    
    feature_percentile = 50
    if config.args.user_percentile:
        print("Using user specified feature percentile for first search")
        feature_percentile = config.args.initial_percentile
    def get_df_filter(df,p=50):
        return np.logical_and.reduce([df[feat]>np.percentile(df[feat],feature_percentile) for feat in ["hyperscore",
                                                                                                 "frag_cosines_p",
                                                                                                 "frag_cosines_p",
                                                                                                  "manhattan_distances",
                                                                                                 ]]+
                                       [df[feat]<np.percentile(df[feat],feature_percentile) for feat in [
                                                                                                        "scribe_scores",
                                                                                                        "gof_stats",
                                                                                                        # "manhattan_distances",
                                                                                                        "max_matched_residuals",
                                                                                                        "med_frag_error"
                                                                                                        ]])
    
    
    
    #### create dictionary for each key and it's positions
    key_dict = {}
    for i, key in enumerate(id_keys_clean):
        key_dict.setdefault(key,[])
        key_dict[key].append(i)
        
    ## find keys that appear more than once
    multiples = []
    multiples_idxs = []
    num_multiples = []
    channels = []
    multiples_hyper = []
    multiples_coeff = []
    multiples_seqs = []
    multiples_zs = []
    
    searched = set()
    for key in set(id_keys):
        # break
        # clean_key = (re.sub("\(tag6-\d\)","",key[0]),key[1])
        # orig_key= key
        # key = clean_key
        if key in searched:
            continue
        else:
            key_pos = key_dict[key]##np.where([i==key for i in id_keys])[0]
            if len(key_pos)>1:
                multiple_rts = np.array([dia_rt[i] for i in key_pos])
                multiples_hyper.append(np.array([output_hyper[i] for i in key_pos]))
                multiples_coeff.append(np.array([output_coeff[i] for i in key_pos]))
                order = np.argsort(multiple_rts)
                order = np.arange(len(multiple_rts))
                multiples.append(multiple_rts[order])
                multiples_idxs.append(np.array(key_pos)[order])
                num_multiples.append(len(key_pos))
                # channels.append([re.findall("\(tag6-(\d+)\)",id_keys[i][0])[0] for i in key_pos])
                multiples_seqs.append([id_keys[i][0] for i in key_pos])
                multiples_zs.append([id_keys[i][1] for i in key_pos])
            searched.update(key)
    
    if config.num_timeplex==0:
        timeplex = stats.mode(num_multiples).mode
    else:
        timeplex = config.num_timeplex
        
    # while it may be nice to know, we are assuming that this is not constant and therfore not necessary to know
    time_diffs = np.concatenate([np.diff(i) for i in multiples if len(i)==timeplex])
    # plt.hist(time_diffs,np.linspace(-1,5,40))
    # plt.xlabel("TimePLEX offset")
    
    # plt.scatter(np.concatenate([i[0:2] for i in multiples if len(i)==timeplex]),time_diffs,s=1,edgecolors="none")
    # plt.ylabel("TimePLEX offset")
    # plt.xlabel("RT")
    
    ## for enantiomers
    # plt.scatter([i[np.argsort(j)][0] for i,j in zip(multiples,channels) if len(i)==timeplex and "8" not in j and len(set(j))==2],
    #             np.concatenate([np.diff(i[np.argsort(j)]) for i,j in zip(multiples,channels) if len(i)==timeplex and "8" not in j and len(set(j))==2]),
    #             # c=np.log10([np.sum(i) for i,j in zip(multiples_hyper,channels) if len(i)==timeplex and "8" not in j and len(set(j))==2]),
    #             s=1)
    # plt.xlabel("RT of d0")
    # plt.ylabel("RT d4 - RT d0")
    # plt.ylim(-5,5)
    
    # df = pd.DataFrame([(*i[np.argsort(j)],float(np.diff(i[np.argsort(j)])),re.sub("\(tag6-\d\)","",np.array(k)[np.argsort(j)][0]),np.array(l)[np.argsort(j)][0]) for i,j,k,l in
    #  zip(multiples,channels,multiples_seqs,multiples_zs) 
    #  if len(i)==timeplex and "8" not in  j and len(set(j))==2],columns = ["RTd0","RTd4","RTdelta","seq","z"])
    # df.to_csv("/Volumes/Lab/KMD/timeplex/T6/StereoData/multiples_45min.csv")
    # plt.scatter([i[0] for i in multiples if len(i)==timeplex],[i[1] for i in multiples if len(i)==timeplex],s=1)
        
    # t1 = np.array([[dia_rt[i[0]],output_rts[i[0]]] for i in multiples_idxs])
    # t2 = np.array([[dia_rt[i[1]],output_rts[i[1]]] for i in multiples_idxs])
    
    rt_spls = []
    t_vals = []
    t_seqs = []
    t_dfs = []
    filters = []
    converted_rts = []
    gaussian_fits = []
    for idx in range(timeplex):
        lib_rt_range = [np.percentile(rt_mz[:,0],5),np.percentile(rt_mz[:,0],95)]
        ### array of (obs_rt, lib_rt, hyperscore)
        t1 = np.array([[dia_rt[i[idx]],output_rts[i[idx]],output_hyper[i[idx]],id_mzs[i[idx]],output_coeff[i[idx]],output_frac_lib[i[idx]]] for i in multiples_idxs if len(i)==timeplex and output_rts[i[idx]]>lib_rt_range[0] and output_rts[i[idx]]<lib_rt_range[1]])
        t1_s = [output_seqs[i[idx]] for i in multiples_idxs if len(i)==timeplex and output_rts[i[idx]]>lib_rt_range[0] and output_rts[i[idx]]<lib_rt_range[1]]
        # t1 = np.array([[dia_rt[i[idx]],output_rts[i[idx]],output_hyper[i[idx]]] for i in multiples_idxs if len(i)==timeplex])
        t_df = output_df.iloc[[i[idx] for i in multiples_idxs if len(i)==timeplex and output_rts[i[idx]]>lib_rt_range[0] and output_rts[i[idx]]<lib_rt_range[1]]]
        new_filter = get_df_filter(t_df,50)
        filters.append(new_filter)
        rt_spl = lowess_fit(t1[:,1][new_filter],t1[:,0][new_filter])
        rt_spls.append(rt_spl)
        t_vals.append(t1)
        t_seqs.append(t1_s)
        
        converted_rt = rt_spl(t1[:,1])
        converted_rts.append(converted_rt)
        gaussian_fits.append(fit_gaussian(t1[:,0]-converted_rt))
    
    # for idx in range(timeplex):
    #     plt.scatter(t_vals[idx][:,1][filters[idx]],t_vals[idx][:,0][filters[idx]],s=1,c=colours[idx],edgecolor="none",label=f"T{str(idx)}")
    #     plt.scatter(t_vals[idx][:,1][filters[idx]],rt_spls[idx](t_vals[idx][:,1][filters[idx]]),s=1,c=colours[idx],edgecolor="none",label=f"T{str(idx)}")
    # plt.xlabel("Library RT")
    # plt.ylabel("Observed RT")
    # plt.ylim(0,60)
    ########################################################################################################
    #########################################################################################################
    #########################################################################################################
    
    
    ## only uses peptides within certain tolerance (Assume most of these are true nad exclude incorrect outliers)
    # _bool = np.abs(rt_diffs)<(4*rt_stddev)
    # ### Could also change to those with the expectd offset when observed
    # rt_offsets = np.array([i[0] for i in t_vals[1]])-[i[0] for i in t_vals[0]]
    # rt_offsets2 = np.array([i[0] for i in t_vals[2]])-[i[0] for i in t_vals[1]] ## for column 2 vs 3
    all_rt_offsets = [np.array([i[0] for i in t_vals[idx+1]])-[i[0] for i in t_vals[idx]] 
                      for idx in range(timeplex-1)]
    offset_tolerance = 1 ## 1 minute
    # expected_offset = stats.mode(np.round(all_rt_offsets[0][all_rt_offsets[0]>.5],1)).mode
    exp_offsets = [stats.mode(np.round(rt_off[rt_off>.5],1)).mode for rt_off in all_rt_offsets] ## ensure it's around zero
    diff_bool = np.abs(all_rt_offsets[0]-exp_offsets[0])<offset_tolerance
    all_diff_bools = [np.abs(all_rt_offsets[idx]-exp_offsets[idx])<offset_tolerance for idx in range(timeplex-1)]
    # plt.scatter(np.array([i[1] for i in t_vals[0]])[diff_bool],np.array([i[0] for i in t_vals[0]])[diff_bool],s=1)
    # plt.scatter(np.array([i[1] for i in t_vals[1]])[diff_bool],np.array([i[0] for i in t_vals[1]])[diff_bool],s=1)
    
    # for idx in range(timeplex):
    #     # plt.subplots()
    #     plt.scatter(np.array([i[1] for i in t_vals[idx]])[np.logical_and.reduce([*all_diff_bools])],np.array([i[0] for i in t_vals[idx]])[np.logical_and.reduce([*all_diff_bools])],s=1,c=colours[idx],edgecolor="none",label=f"T{str(idx)}")
    # plt.xlabel("Library RT")
    # plt.ylabel("Observed RT")
    # plt.ylim(0,60)
    
    ## fit to the "zeroth" column
    f = lowess_fit(np.array([i[1] for i in t_vals[1]])[diff_bool],np.array([i[0] for i in t_vals[0]])[diff_bool])
    
    # plt.scatter([i[1] for i in t_vals[0]],[i[0] for i in t_vals[0]],s=1)
    # plt.scatter([i[1] for i in t_vals[0]],f([i[1] for i in t_vals[0]]),s=1)
    
    # rt_diffs = f([i[1] for i in t_vals[1]])-[i[0] for i in t_vals[0]]
    # rt_amplitude, rt_mean, rt_stddev = fit_gaussian(rt_diffs)
    
    # vals,bins,_ = plt.hist(rt_diffs,np.linspace(-10,10,150),density=True)
    # plt.vlines([-4*rt_stddev,4*rt_stddev],0,max(vals),color="g")
    # plt.hist(rt_diffs[np.abs(rt_diffs)<(4*rt_stddev)],50,density=True,alpha=.5)
    
    
    ## use observed rt for fine_tuning
    grouped_df = pd.DataFrame({'Stripped.Sequence':[librarySpectra[(i[0],float(i[1]))]["seq"] for i in t_seqs[0]],"RT":[i[0] for i in t_vals[0]]})[diff_bool]
    data_split, models, convertor = fine_tune_rt(grouped_df,qc_plots=True,results_path=results_folder)
    
    # all_seqs_onehot = [one_hot_encode_sequence(seq) for seq in grouped_df['Stripped.Sequence']]
    # all_seqs_onehot = [one_hot_encode_sequence(i[0]) for i in t_seqs[0]]
    # predictions = np.mean([model.predict(np.array(all_seqs_onehot)) for model in models],axis=0)
    
    
    # plt.scatter(f1(convertor(predictions)).flatten(),[i[0] for i in t_vals[0]],s=1)
    # plt.scatter(f1(convertor(predictions)).flatten(),[i[0] for i in t_vals[1]],s=1)
    
    
    
    
    ### compare original empirical RTs to fintuned RTs
    
    
    ### recalculate RT_spls...
    keys = [(i,float(j)) for i,j in t_seqs[0]]
    
    lib_seqs = [one_hot_encode_sequence(librarySpectra[key]["seq"]) for key in keys]
    new_lib_rts = convertor(np.mean([model.predict(np.array(lib_seqs)) for model in models],axis=0).flatten())
    
    t0_rts = new_lib_rts
    ## take 95% to decrease effect of ends inlfuencing predictions
    # rt_filter_bool = np.logical_and(t0_rts>np.percentile(t0_rts,1),t0_rts<np.percentile(t0_rts,95))
    rt_filter_bool = filter_rts_by_dense(t0_rts,30)
    rt_spls = []
    for idx in range(timeplex):
        # rt_spl = threestepfit([updatedLibrary[key]["iRT"] for key in keys],[i[0] for i in t_vals[0]],1)
        rt_spl = lowess_fit(t0_rts[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],
                            np.array([i[0] for i in t_vals[idx]])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],frac=.2)
        rt_spls.append(rt_spl)
    
    # for idx in range(timeplex):
    #     # plt.subplots()
    #     test_bool = np.logical_and(diff_bool,rt_filter_bool)
    #     plt.scatter(t0_rts[test_bool],np.array([i[0] for i in t_vals[idx]])[test_bool],c=colours[idx],s=1,edgecolor="none") 
    #     plt.scatter(t0_rts[test_bool],rt_spls[idx](t0_rts)[test_bool],c=colours[idx],s=1,alpha=.2) 
    # plt.scatter(np.array([updatedLibrary[key]["iRT"] for key in keys]),np.array([i[0] for i in t_vals[1]]),s=1,alpha=.2)    
    # plt.scatter(np.array([updatedLibrary[key]["iRT"] for key in keys])[diff_bool],np.array([i[0] for i in t_vals[0]])[diff_bool],s=1,alpha=.2)
    # plt.scatter(np.array([updatedLibrary[key]["iRT"] for key in keys])[test_bool],np.array([i[0] for i in t_vals[0]])[test_bool],s=1,alpha=.2)
    # plt.scatter(np.array([updatedLibrary[key]["iRT"] for key in keys])[diff_bool],np.array([i[0] for i in t_vals[1]])[diff_bool],s=1,alpha=.2)
    # # plt.scatter([updatedLibrary[key]["iRT"] for key in keys],rt_spls[0]([updatedLibrary[key]["iRT"] for key in keys]),s=1)
    
    ### compare to empirical fit
    emp_rt_spls = []
    for idx in range(timeplex):
        # rt_spl = threestepfit([updatedLibrary[key]["iRT"] for key in keys],[i[0] for i in t_vals[0]],1)
        rt_spl = lowess_fit(np.array(t_vals[idx][:,1])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],
                            np.array(t_vals[idx][:,0])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],frac=.1)
        emp_rt_spls.append(rt_spl)

    # for idx in range(timeplex):
    #     # plt.subplots()
    #     plt.scatter(np.array(t_vals[idx][:,1])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],
    #                         np.array(t_vals[idx][:,0])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],
    #                         c=colours[idx],s=1,edgecolor="none") 
    #     plt.scatter(np.array(t_vals[idx][:,1])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],
    #                 emp_rt_spls[idx](np.array(t_vals[idx][:,1]))[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],
    #                 c=colours[idx],s=1,alpha=.2) 
    
    
    all_emp_diffs = np.concatenate([emp_rt_spls[i](np.array(t_vals[idx][:,1]))[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])]-np.array(t_vals[i][:,0])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])] for i in range(timeplex)])
    # kde = stats.gaussian_kde(all_emp_diffs,.01)
    # plt.plot(np.linspace(-10,10,200),kde(np.linspace(-10,10,200)))
    # plt.hist(np.abs(all_emp_diffs),np.linspace(-10,10,200))
    # plt.xlabel("Empirical RT error")
    
    all_pred_diffs = np.concatenate([rt_spls[i](t0_rts)[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])]-np.array([i[0] for i in t_vals[i]])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])] for i in range(timeplex)])
    # plt.hist(np.abs(all_pred_diffs),np.linspace(-10,10,200))
    # plt.xlabel("Predicted RT error")
    
    # np.percentile(np.abs(all_emp_diffs), 50)
    # np.percentile(np.abs(all_pred_diffs), 50)
    
    
    # limit=5 ## exlcude RT diffs larger than this (outliers)
    # data = sorted(np.abs(all_emp_diffs)[np.abs(all_emp_diffs)<limit])
    # p = 1. * np.arange(len(data)) / (len(data) - 1)
    # # plt.plot(data,p,label="Empirical RT")
    # emp_cdf_auc = auc(data,p)
    # data = sorted(np.abs(all_pred_diffs)[np.abs(all_pred_diffs)<limit])
    # p = 1. * np.arange(len(data)) / (len(data) - 1)
    # # plt.plot(data,p,label="Predicted RT")
    # pred_cdf_auc = auc(data,p)
    # # plt.legend()
    # # plt.xlabel("RT difference")
    # # plt.ylabel("Fraction of Precursors")
    
    limit=3 ## exlcude RT diffs larger than this (outliers)
    emp_data_filter = all_emp_diffs[np.abs(all_emp_diffs) < limit]
    emp_data = np.sort(np.abs(emp_data_filter))
    emp_data = np.append(emp_data,limit)
    emp_p = np.arange(len(emp_data)) / (len(emp_data) - 1)
    emp_cdf_auc = auc(emp_data,emp_p)
    pred_data_filter = all_pred_diffs[np.abs(all_pred_diffs) < limit]
    pred_data = np.sort(np.abs(pred_data_filter))
    pred_data = np.append(pred_data,limit)
    pred_p = np.arange(len(pred_data)) / (len(pred_data) - 1)
    pred_cdf_auc = auc(pred_data,pred_p)
    

    percentile = config.rt_percentile
    

    # plt.plot(emp_data,emp_p,label="Empirical RT",color=colours[0])
    # plt.plot(pred_data,pred_p,label="Predicted RT",color=colours[1])
    # plt.legend()
    # plt.xlabel("RT difference")
    # plt.ylabel("Fraction of Precursors")
    
    # percentile =.999
    # emp_abs_errors_med = np.median(np.abs(emp_data_filter[emp_data_filter<limit]-np.median(emp_data_filter[emp_data_filter<limit])))
    # pred_abs_errors_med = np.median(np.abs(pred_data_filter[pred_data_filter<limit]-np.median(pred_data_filter[pred_data_filter<limit])))
    # plt.plot(emp_data,stats.expon.cdf(emp_data,loc=0,scale=emp_abs_errors_med/np.log(2)),label="Emp Expon",linestyle="--",color=colours[0])
    # plt.scatter([stats.expon.ppf(percentile,scale=emp_abs_errors_med/np.log(2))], [percentile],c=colours[0],marker="*")
    # plt.plot(emp_data,stats.halfnorm.cdf(emp_data,loc=0,scale=np.power(emp_abs_errors_med*1.4826,1)),label="Emp Gauss",linestyle=":",color=colours[0])
    # plt.scatter([stats.halfnorm.ppf(percentile,scale=emp_abs_errors_med*1.4826)], [percentile],c=colours[0])
    # plt.plot(pred_data,stats.expon.cdf(pred_data,loc=0,scale=pred_abs_errors_med/np.log(2)),label="Pred Expon",linestyle="--",color=colours[1])
    # plt.scatter([stats.expon.ppf(percentile,scale=pred_abs_errors_med/np.log(2))], [percentile],c=colours[1],marker="*")
    # plt.plot(pred_data,stats.halfnorm.cdf(pred_data,loc=0,scale=np.power(pred_abs_errors_med*1.4826,1)),label="Pred Gauss",linestyle=":",color=colours[1])
    # plt.scatter([stats.halfnorm.ppf(percentile,scale=pred_abs_errors_med*1.4826)], [percentile],c=colours[1])
    # plt.legend()
    # plt.xlim(0-limit*.05,limit*1.05)
    
    
    
    ### Note: I need to return an updated library not just the rt_spl
    ### Or else return the models used to predict the RT
    
    
    updatedLibrary = copy.deepcopy(librarySpectra)
    all_lib_keys = list(librarySpectra)
    
    if pred_cdf_auc>emp_cdf_auc: ## Predictions are better

        boundary = fit_errors(all_pred_diffs,limit,percentile)

        all_lib_seqs = [one_hot_encode_sequence(updatedLibrary[key]["seq"]) for key in all_lib_keys]
        all_new_lib_rts = convertor(np.mean([model.predict(np.array(all_lib_seqs)) for model in models],axis=0).flatten())
        
        for key,rt in zip(all_lib_keys,all_new_lib_rts):
            updatedLibrary[key]["iRT"] = rt
            
    else: ### empirical are better
        ## keep the library RTs the same

        boundary = fit_errors(all_emp_diffs,limit,percentile)

        ## update the splines
        rt_spls = emp_rt_spls
    # ## get keys from t_vals and recreate scatter plot
    # keys = [(i,float(j)) for i,j in t_seqs[0]]
    # plt.scatter(f1(convertor([updatedLibrary[key]["iRT"] for key in keys])),[i[0] for i in t_vals[0]],s=1)
    # plt.plot([10,50],[10,50])
    # plt.scatter(f1(convertor([updatedLibrary[key]["iRT"] for key in keys])),[i[0] for i in t_vals[1]],s=1)
    # plt.plot([10,50],[13,53])
    
    
    # ## just use T0
    # export_df = pd.DataFrame({"obs_rt":np.concatenate([t_vals[0][:,0],t_vals[1][:,0]]),
    #                           "lib_rt":np.concatenate([t_vals[0][:,1],t_vals[1][:,1]]),
    #                           "seq":[i[0] for i in t_seqs[0]]+[i[0] for i in t_seqs[1]],
    #                           "charge":[i[1] for i in t_seqs[0]]+[i[1] for i in t_seqs[1]]})
    
    # export_df = pd.DataFrame({"obs_rt_0":t_vals[0][:,0],
    #                           "obs_rt_1":t_vals[1][:,0],
    #                           "lib_rt":t_vals[0][:,1],
    #                           "seq":[i[0] for i in t_seqs[0]],
    #                           "charge":[i[1] for i in t_seqs[0]]})
    # export_df.to_csv("/Volumes/Lab/KMD/For_JD/T6doublets.csv")
    
    
    ## combined gausian fit
    # rt_amplitude, rt_mean, rt_stddev = fit_gaussian(np.concatenate([t[:,0]-c_rt for t,c_rt in zip(t_vals,converted_rts)]))
    # f = lowess_fit([i[1] for i in t_vals[0]],[i[0] for i in t_vals[0]])
    # f1 = lowess_fit(convertor(predictions).flatten(),[i[0] for i in t_vals[0]],frac=.4)
    rt_amplitude, rt_mean, rt_stddev = fit_gaussian(rt_spls[0]([updatedLibrary[key]["iRT"] for key in keys])[diff_bool]-np.array([i[0] for i in t_vals[0]])[diff_bool],bin_n=100)
    
    emp_rt_amplitude, emp_rt_mean, emp_rt_stddev = fit_gaussian(emp_rt_spls[0](np.array(t_vals[0][:,1]))[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])]-np.array(t_vals[0][:,0])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],bin_n=100)
    
    
    
    # vals,bins,_ = plt.hist((f([i[1] for i in t_vals[0]])-[i[0] for i in t_vals[0]])[np.logical_and(diff_bool,rt_filter_bool)],np.linspace(-10,10,150),density=True,label="Old RT")
    # vals,bins,_ = plt.hist((rt_spls[0]([updatedLibrary[key]["iRT"] for key in keys])-[i[0] for i in t_vals[0]])[np.logical_and(diff_bool,rt_filter_bool)],bins,alpha=.5,density=True,label="New RT")
    # plt.plot(np.linspace(-5,5,100),gaussian(np.linspace(-5,5,100), rt_amplitude, rt_mean, rt_stddev),label="New RT fit")
    # # plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals))
    # # plt.legend()
    # ### vals,bins,_ = plt.hist(np.abs(rt_spls[0]([updatedLibrary[key]["iRT"] for key in keys])-[i[0] for i in t_vals[0]])[np.logical_and(diff_bool,rt_filter_bool)],bins,alpha=.5,density=True,label="New RT")
    
    # vals,bins,_ = plt.hist((f([i[1] for i in t_vals[0]])-[i[0] for i in t_vals[0]])[np.logical_and(diff_bool,rt_filter_bool)],np.linspace(-10,10,150),density=True,label="Old RT")
    # vals,bins,_ = plt.hist(emp_rt_spls[0](np.array(t_vals[0][:,1]))[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])]-np.array(t_vals[0][:,0])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],bins,alpha=.5,density=True,label="New RT")
    # plt.plot(np.linspace(-5,5,100),gaussian(np.linspace(-5,5,100),  emp_rt_amplitude, emp_rt_mean, emp_rt_stddev),label="New RT fit")
    # plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals))
    
   
    ## NB: Only for timeplex=2
    ## computes differences between the fit lines of both plexes
    prediction_diffs = np.abs(rt_spls[1]([updatedLibrary[key]["iRT"] for key in keys])-rt_spls[0]([updatedLibrary[key]["iRT"] for key in keys]))
    all_prediction_diffs = []
    for idx in range(timeplex-1):
        all_prediction_diffs.append(np.abs(rt_spls[idx+1]([updatedLibrary[key]["iRT"] for key in keys])-rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys])))
    #####  Assume that the mz error is independent of timeplex
    resp_ms1scans = [closest_ms1spec(dia_rt[i], ms1_rt) for i in range(len(dia_rt))]
    diffs = [closest_peak_diff(mz, ms1spectra[i].mz) for i,mz in zip(resp_ms1scans,id_mzs)]
    
    mz_spl = twostepfit(id_mzs,diffs,1)
    
    
    
    
    
    ################################################
    ########### correct mz errors wrt RT    ########
    ################################################
    
    rts = np.array([updatedLibrary[(i[0],float(i[1]))]["iRT"] for i in output_seqs])#np.array([i[0] for i in t_vals[0]])
    # rt_filter_bool = filter_rts_by_dense(rts,30)
    # rt_filter_bool = np.logical_and(rts>15,rts<30)
    rt_mz_filter_bool = np.array(output_frac_lib)>.9 # use as proxy for correct IDs
    f_rt_mz = lowess_fit(rts[rt_mz_filter_bool],np.array(diffs)[rt_mz_filter_bool],.2)
    # plt.scatter(rts[rt_filter_bool],np.array(diffs)[rt_filter_bool],label="Original_MZ",s=1,alpha=.1)
    # plt.scatter(dia_rt,f_rt_mz(dia_rt),s=1,alpha=.2)
    
    # plt.scatter(id_mzs,diffs,label="Original_MZ",s=1,alpha=.1)
    # plt.scatter(id_mzs,diffs-f_rt_mz(dia_rt),label="Original_MZ",s=1,alpha=.1)
    
    # mz_spl = twostepfit(np.array(id_mzs)[rt_filter_bool],(diffs-f_rt_mz(dia_rt))[r t_filter_bool],1)
    mz_spl = lowess_fit(np.array(id_mzs)[rt_mz_filter_bool],(diffs-f_rt_mz(dia_rt))[rt_mz_filter_bool])
    # plt.scatter(id_mzs,diffs-f_rt_mz(dia_rt),label="Original_MZ",s=1,alpha=.1)
    # plt.scatter(id_mzs,mz_spl(id_mzs),label="Original_MZ",s=1,alpha=.1)
    # plt.hlines(0,400,900)

    def mz_func(mz,rt):
        return mz+((mz_spl(mz)+f_rt_mz(rt))*mz)
    
    # orig_mzs = id_mzs+(diffs*np.array(id_mzs))
    # plt.hist(((mz_func(id_mzs,rts)-orig_mzs)/id_mzs)[rt_filter_bool],100)
    
    corrected_mz_diffs = (diffs-(f_rt_mz(rts)+mz_spl(id_mzs)))[rt_mz_filter_bool]
    mz_amplitude, mz_mean, mz_stddev = fit_gaussian(corrected_mz_diffs)
    
    # plt.hist(np.array(diffs)[rt_filter_bool],100,density=True)
    # vals,bins,_ = plt.hist(corrected_mz_diffs,100,alpha=.5,density=True)
    # plt.plot(bins,gaussian(bins, mz_amplitude, mz_mean, mz_stddev))
    # plt.vlines(0,0,max(vals))
    
    ### MS2 alignment
    if ms2:
        all_frag_errors = np.concatenate(lc_frags_errors)
        all_frags = np.concatenate(lc_frags)
        ms2_spl = twostepfit(all_frags,all_frag_errors,1)
        def ms2_func(mz):
            return mz+(ms2_spl(mz)*mz)
        
        ms2_amplitude, ms2_mean, ms2_stddev = fit_gaussian(all_frag_errors-ms2_spl(all_frags))
    
   
    
    # new_rt_tol = get_tol(dia_rt-rt_spl(output_rts))
    new_rt_tol =np.abs(boundary)# 4*np.abs(rt_stddev)
    if config.args.user_rt_tol:
        print("Using user specified RT tolerance")
        new_rt_tol = np.abs(config.args.rt_tol)
    print(f"Optimsed RT tolerance: {new_rt_tol}")
    
    # ## ensure there is no overlap
    # obs_rt_range = [min(dia_rt),max(dia_rt)]
    # ## range that captures middle 90% of library
    # lib_rt_range = [np.percentile(rt_mz[:,0],5),np.percentile(rt_mz[:,0],95)]
    # sample_rts = np.linspace(lib_rt_range[0],lib_rt_range[1],100)
    # # plt.scatter(sample_rts,rt_spls[0](sample_rts),s=1)
    # # plt.scatter(sample_rts,rt_spls[1](sample_rts),s=1)
    # ## differnce in 
    # model_diffs = np.abs(rt_spls[0](sample_rts)-rt_spls[1](sample_rts))
    # rt_tol_spl = InterpolatedUnivariateSpline(sample_rts,model_diffs)
    # # plt.plot(rt_spls[0](sample_rts),model_diffs)
    # # plt.scatter(rt_spls[1](t_vals[1][:,1]),np.ones_like(rt_spls[1](t_vals[1][:,1]))*new_rt_tol*2,s=1)
    
    # def rt_tol_fn(obs_rt):
    #     return np.maximum(np.minimum(new_rt_tol,(rt_tol_spl(lib_rt)/2)*.99),0)
    
    # config.rt_tol_spl = rt_tol_fn
    
    # ensure there is no overlap
    # if new_rt_tol>np.median(time_diffs)/2:
    min_prediction_diff = np.min(prediction_diffs)
    min_prediction_diff = np.min([np.min(i) for i in prediction_diffs])
    
    if new_rt_tol>np.abs(min_prediction_diff/2):
        print("Warning; Library RTs overlapping")
        new_rt_tol = np.abs(min_prediction_diff/2)*.99 # ensure no overlap
        print(f"Reseting tolerance to {new_rt_tol}")
    

    
    config.opt_rt_tol = new_rt_tol
    
    
    
    # set optimised ms2 tol
    # is_real = ~np.isnan(diffs)
    # buffer = 1.2
    # config.opt_ms1_tol = np.round(
    #                         np.sort(
    #                             np.abs(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs
    #                                    )[is_real])[int(sum(is_real)*.95)]*buffer,6+5)#6 for 1e-6 the 5 decimal places

    # new_ms1_tol = get_tol(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs)
    # new_ms1_tol = get_tol(diffs-mz_spl(id_mzs))
    new_ms1_tol = np.abs(4*mz_stddev)
    print(f"Optimsed ms1 tolerance: {new_ms1_tol}")
    
    config.opt_ms1_tol  = new_ms1_tol
    
    if ms2:
        new_ms2_tol = 4*ms2_stddev
        config.opt_ms2_tol  = new_ms2_tol
    
    if results_folder is not None:
        
        ### Save functions
        for idx in range(timeplex):
            with open(results_folder+f"/rt_spl{idx}","wb") as dill_file:
                dill.dump(rt_spls[idx],dill_file)
            
        with open(results_folder+"/mz_func","wb") as dill_file:
            dill.dump(mz_func,dill_file)
        
        if ms2:
            with open(results_folder+"/ms2_func","wb") as dill_file:
                dill.dump(ms2_func,dill_file)
            
        ##plot RT alignment
        filter_bool = np.logical_and.reduce([*all_diff_bools,rt_filter_bool])
        
        plt.subplots()
        for idx in range(timeplex):
            plt.scatter(np.array(t_vals[idx][:,1])[filter_bool],
                        np.array(t_vals[idx][:,0])[filter_bool],
                        s=1,c=colours[idx], alpha=.2,label=f"T{str(idx)}")
            # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1]),s=1,label=f"T{str(idx)}",c=colours[idx])
            # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])+config.opt_rt_tol,s=.1,c=colours[idx],alpha=.1)
            # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])-config.opt_rt_tol,s=.1,c=colours[idx],alpha=.1)
            # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])+config.rt_tol_spl(t_vals[idx][:,1]),s=.1,c=colours[idx],alpha=.1)
            # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])-config.rt_tol_spl(t_vals[idx][:,1]),s=.1,c=colours[idx],alpha=.1)
        plt.legend(markerscale=10)
        plt.xlabel("Library RT")
        plt.ylabel("Observed RT")
        # plt.show()
        plt.savefig(results_folder+"/OriginalRTfit.png",dpi=600,bbox_inches="tight")
            
        ### want this later
        plt.subplots()
        for idx in range(timeplex):
            plt.scatter(np.array([updatedLibrary[key]["iRT"] for key in keys])[filter_bool],np.array([i[0] for i in t_vals[idx]])[filter_bool],s=1,label=f"T{str(idx)}",alpha=.2)
            # plt.scatter([updatedLibrary[key]["iRT"] for key in keys],rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys]),s=1,label=f"T{str(idx)}",c=colours[idx])
            plt.scatter([updatedLibrary[key]["iRT"] for key in keys],rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys])+config.opt_rt_tol,s=.1,c=colours[idx],alpha=.1)
            plt.scatter([updatedLibrary[key]["iRT"] for key in keys],rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys])-config.opt_rt_tol,s=.1,c=colours[idx],alpha=.1)
            # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])+config.rt_tol_spl(t_vals[idx][:,1]),s=.1,c=colours[idx],alpha=.1)
            # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])-config.rt_tol_spl(t_vals[idx][:,1]),s=.1,c=colours[idx],alpha=.1)
        plt.legend(markerscale=10)
        plt.xlabel("Updated Library RT")
        plt.ylabel("Observed RT")
        plt.savefig(results_folder+"/RTfit.png",dpi=600,bbox_inches="tight")
        
        plt.subplots()
        for idx in range(timeplex):
            vals,bins,_ =plt.hist(np.array(t_vals[idx][:,0]-rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys]))[filter_bool],100,alpha=.5,label=f"T{str(idx)}")
            # rt_stddev = gaussian_fits[idx][-1]
        x_scale = np.diff(plt.xlim())[0]
        plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals),color="r")
        plt.text(config.opt_rt_tol+x_scale/100,max(vals)*.8,np.round(config.opt_rt_tol,2))
        plt.legend()  
        plt.xlabel("RT difference")
        plt.ylabel("Frequency") 
        # plt.show()
        plt.savefig(results_folder+"/RTdiff.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        for idx in range(timeplex):
            if idx!=0:
                offset = all_prediction_diffs[idx-1]
            else:
                offset = 0
            vals,bins,_ =plt.hist(np.array(t_vals[idx][:,0]-rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys])+offset)[filter_bool],100,alpha=.5,label=f"T{str(idx)}")
            # rt_stddev = gaussian_fits[idx][-1]
            plt.vlines([-config.opt_rt_tol+np.median(offset),config.opt_rt_tol+np.median(offset)],0,max(vals),color="r")
        x_scale = np.diff(plt.xlim())[0]
        # plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals),color="r")
        plt.text(config.opt_rt_tol+x_scale/100,max(vals)*.8,np.round(config.opt_rt_tol,2))
        plt.legend()  
        plt.xlabel("RT difference")
        plt.ylabel("Frequency") 
        plt.savefig(results_folder+"/Rterrors.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        vals,bins,_ = plt.hist((f([i[1] for i in t_vals[0]])-[i[0] for i in t_vals[0]])[filter_bool],np.linspace(-10,10,150),density=True,label="Original RT")
        plt.hist((rt_spls[0]([updatedLibrary[key]["iRT"] for key in keys])-[i[0] for i in t_vals[0]])[filter_bool],bins,alpha=.5,density=True,label="Updated RT")
        plt.plot(np.linspace(-5,5,100),gaussian(np.linspace(-5,5,100), rt_amplitude, rt_mean, rt_stddev),label="Updated RT fit")
        plt.legend()
        plt.xlabel("RT alignment errors")
        plt.savefig(results_folder+"/RtAlignmentErrors.png",dpi=600,bbox_inches="tight")
        # plt.show()
        
        fig, ax = plt.subplots(nrows = timeplex, figsize=(7.2, 3.6*timeplex))        
        for idx,row in enumerate(ax):
            row.scatter(np.array(t_vals[idx][:,1])[filter_bool],np.array(t_vals[idx][:,0]-rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys]))[filter_bool],label="Original_RT",s=.1)
            row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[0,0],color="r",linestyle="--",alpha=.5)
            row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[config.opt_rt_tol,config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
            row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[-config.opt_rt_tol,-config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
            row.set_ylabel(f"RT Residuals (T{idx})")
            row.set_ylim(-5,5)
        # plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
        # plt.legend()
        plt.xlabel("Updated Library RT")
        # plt.ylabel("RT Residuals")
        # plt.show()
        plt.savefig(results_folder+"/RtResidual.png",dpi=600,bbox_inches="tight")
        
        
        # Plot the CDFs with elbow points
        plt.subplots()
        plt.figure(figsize=(8, 5))
        plt.plot(emp_data, emp_p, label="Original CDF", linestyle='-')
        plt.plot(pred_data, pred_p, label="Finetuned CDF", linestyle='-')
        # plt.scatter(elbow_emp_x, elbow_emp_y, color='blue', label=f'Original Elbow at {elbow_emp_x:.2f}', zorder=3)
        # plt.scatter(elbow_pred_x, elbow_pred_y, color='red', label=f'Finetuned Elbow at {elbow_pred_x:.2f}', zorder=3)
        
        emp_abs_errors_med = np.median(np.abs(all_emp_diffs[all_emp_diffs<limit]-np.median(all_emp_diffs[all_emp_diffs<limit])))

        plt.plot(emp_data,stats.expon.cdf(emp_data,loc=0,scale=emp_abs_errors_med/np.log(2)),linestyle="--",color=colours[0],label="Emp Expon CDF")
        emp_exp_999 = stats.expon.ppf(percentile,scale=emp_abs_errors_med/np.log(2))
        plt.scatter([emp_exp_999], [percentile],c=colours[0],label=f"Emp Expon {percentile}: {emp_exp_999:.2f}",marker="*")
        plt.plot(emp_data,stats.halfnorm.cdf(emp_data,loc=0,scale=np.power(emp_abs_errors_med*1.4826,1)),linestyle=":",color=colours[0],label="Emp Norm CDF")
        emp_gauss_999 = stats.halfnorm.ppf(percentile,scale=emp_abs_errors_med*1.4826)
        plt.scatter([emp_gauss_999], [percentile],c=colours[0],label=f"Emp Norm {percentile}: {emp_gauss_999:.2f}")
        
        pred_abs_errors_med = np.median(np.abs(all_pred_diffs[all_pred_diffs<limit]-np.median(all_pred_diffs[all_pred_diffs<limit])))
        plt.plot(pred_data,stats.expon.cdf(pred_data,loc=0,scale=pred_abs_errors_med/np.log(2)),linestyle="--",color=colours[1],label="Pred Exp CDF")
        pred_exp_999 = stats.expon.ppf(percentile,scale=pred_abs_errors_med/np.log(2))
        plt.scatter([pred_exp_999], [percentile],c=colours[1],label=f"Pred Expon {percentile}: {pred_exp_999:.2f}",marker="*")
        plt.plot(pred_data,stats.halfnorm.cdf(pred_data,loc=0,scale=np.power(pred_abs_errors_med*1.4826,1)),linestyle=":",color=colours[1],label="Pred Norm CDF")
        pred_gauss_999 = stats.halfnorm.ppf(percentile,scale=pred_abs_errors_med*1.4826)
        plt.scatter([pred_gauss_999], [percentile],c=colours[1],label=f"Pred Norm {percentile}: {pred_gauss_999:.2f}")

        plt.vlines(boundary,0,1,colors="r",linestyle="--",label="Boundary")
        
        plt.xlabel("RT Differences")
        plt.ylabel("Cumulative Probability")
        plt.legend()
        plt.title("Finding an optimal RT library")
        plt.savefig(results_folder+"/RTelbows.png",dpi=600,bbox_inches="tight")
        
        
        
        ##plot mz alignment
        plt.subplots()
        plt.scatter(rts,diffs,label="Original_MZ",s=1,alpha=min(1,5/((len(dia_rt)//1000)+1)))
        plt.scatter(rts,f_rt_mz(rts),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("Updated RT")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        plt.savefig(results_folder+"/MZrtfit.png",dpi=600,bbox_inches="tight")
        
        ##plot mz alignment
        plt.subplots()
        plt.scatter(id_mzs,diffs-f_rt_mz(rts),label="Original_MZ",s=1,alpha=min(1,5/((len(dia_rt)//1000)+1)))
        plt.scatter(id_mzs,mz_spl(id_mzs),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("m/z")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        plt.savefig(results_folder+"/MZfit.png",dpi=600,bbox_inches="tight")
        
        
        ## plot mz alignment
        plt.subplots()
        plt.hist(np.array(diffs)[rt_mz_filter_bool],100)
        # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs,100,alpha=.5)
        # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
        vals,bins,_ = plt.hist((diffs-mz_spl(id_mzs)-f_rt_mz(rts))[rt_mz_filter_bool],100,alpha=.5)
        plt.vlines([-config.opt_ms1_tol,config.opt_ms1_tol],0,max(vals)*.8,color="r")
        # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
        plt.text(config.opt_ms1_tol,max(vals)*.8,f"{np.round(1e6*config.opt_ms1_tol,2)} ppm")
        plt.xlabel("m/z difference (relative)")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(results_folder+"/MZdiff.png",dpi=600,bbox_inches="tight")
    
    
        if ms2:
            ##plot mz alignment
            plt.subplots()
            plt.scatter(all_frags,all_frag_errors,label="Original_MS2",s=1)
            plt.scatter(all_frags,ms2_spl(all_frags),label="Predicted_MS2",s=1)
            # plt.legend()
            plt.xlabel("m/z")
            plt.ylabel("m/z difference (relative)")
            # plt.show()
            plt.savefig(results_folder+"/MS2fit.png",dpi=600,bbox_inches="tight")
            
            
            ## plot mz alignment
            plt.subplots()
            plt.hist(all_frag_errors,100)
            # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs,100,alpha=.5)
            # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
            plt.hist(all_frag_errors-ms2_spl(all_frags),100,alpha=.5)
            plt.vlines([-config.opt_ms2_tol,config.opt_ms2_tol],0,50,color="r")
            # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
            plt.text(config.opt_ms2_tol,50,np.round(config.opt_ms2_tol,2))
            plt.xlabel("m/z difference (relative)")
            plt.ylabel("Frequency")
            # plt.show()
            plt.savefig(results_folder+"/MS2diff.png",dpi=600,bbox_inches="tight")
    # plt.scatter(id_mzs,diffs)
    # plt.scatter(output_rts,diffs)
    
    if ms2:
        return (rt_spls, mz_func, ms2_func), updatedLibrary
    else:
        return (rt_spls, mz_func), updatedLibrary



