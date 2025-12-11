"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

import numpy as np
import pandas as pd
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm
import src.config as config
from src.spectral_fitting import fit_to_lib

from scipy.interpolate import LSQUnivariateSpline as spline
from scipy.stats import norm
#from scipy.optimize import isotonic_regression
from statistics import quantiles
from src.utils.misc_functions import within_tol
from scipy import signal
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import auc
import dill
dill.settings['recurse'] = True
import copy

from scipy.interpolate import interp1d
import statsmodels.api as sm


#from src.mass_tags import tag_library, mTRAQ, mTRAQ_678, mTRAQ_02468, diethyl_6plex, tag6

from src.utils.misc_functions import within_tol,moving_average, \
    closest_ms1spec, closest_peak_diff, unstring_floats, fragment_cor


from src.finetune_funs import fine_tune_rt, one_hot_encode_sequence
from src.utils.io.read_output import names, dtypes

from src.logger import logger

colours = ["tab:blue","tab:orange","tab:green","tab:red",
'tab:purple',
'tab:brown',
'tab:pink',
'tab:gray',
'tab:olive',
'tab:cyan']


def twostepfit(x,y,n_knots=2,z=None,k1=1):
    """
    Get spline that maps x to y in 2 steps. Outliers are removed after first step

    Parameters
    ----------
    x : array
        Series of x values.
    y : array
        Series of x values.
    n_knots : int, optional
        How many knots in the spline. The default is 2.
    z : array, optional
        If present, attributes used to weight the spline fitting. The default is None.
    k1 : int, optional
        Degree of spline. The default is 1.

    Returns
    -------
    spl2 : scipy.interpolate.UnivariateSpline
        Spline mapping x to y.

    """
    
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

# def threestepfit(x,y,n_knots=2,z=None,k1=1):
#     """
#     Get spline that maps x to y in 3 steps. Outliers are removed after each step

#     Parameters
#     ----------
#     x : array
#         Series of x values.
#     y : array
#         Series of x values.
#     n_knots : int, optional
#         How many knots in the spline. The default is 2.
#     z : array, optional
#         If present, attributes used to weight the spline fitting. The default is None.
#     k1 : int, optional
#         Degree of spline. The default is 1.

#     Returns
#     -------
#     spl2 : scipy.interpolate.UnivariateSpline
#         Spline mapping x to y.

#     """
#     if z is None:
#         z= np.ones_like(x)
#     y_exists = np.isfinite(y)
#     x_exists = np.isfinite(x)*y_exists
#     x=np.array(x)[x_exists]
#     y=np.array(y)[x_exists]
#     z=np.array(z)[x_exists]
#     y_range = np.max(y)-np.min(y)
#     sorted_idxs = np.argsort(x)
#     sort_x = np.array(x)[sorted_idxs]
#     sort_y = np.array(y)[sorted_idxs]
#     sort_z = np.array(z)[sorted_idxs]
#     knots = quantiles(sort_x,n=n_knots)
#     spl = spline(sort_x,sort_y,knots,w=sort_z,k=1)
#     # poly = np.polyfit(sort_x, sort_y, w=sort_z, deg=5)
#     # sort_x+=np.arange(len(sort_x))*1e-7
#     # spl  = InterpolatedUnivariateSpline(sort_x,sort_y,w=np.log10(sort_z),k=5)
#     # plt.plot(sort_x,np.polyval(poly, sort_x))
#     # plt.scatter(x,y,s=1)
#     # plt.scatter(x,spl(x),s=1)
#     # find outliers and remove; points over 1/4 of the y range away from prediction
#     _bool = abs(spl(sort_x)-sort_y)<(y_range/4)
#     # knots = quantiles(sort_x,n=4)
#     spl2 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
#     # spl2 = UnivariateSpline(sort_x,sort_y)
#     # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
#     # plt.scatter(x,spl2(x),s=1)
    
#     _bool = abs(spl2(sort_x)-sort_y)<(y_range/8)
    
#     # knots = quantiles(sort_x,n=n_knots)
#     spl3 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
#     # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
#     # plt.scatter(sort_x[_bool],spl3(sort_x[_bool]),s=1)
    
#     return spl3



# def initstepfit(x,y,n_knots=2,z=None,k1=1):
#     """
#     Get spline that maps x to y in 3 steps. Outliers are removed after each step.
#     SIinitial guess is a straight line from [min_x,min_] to [max_x,max_y]
    
#     Parameters
#     ----------
#     x : array
#         Series of x values.
#     y : array
#         Series of x values.
#     n_knots : int, optional
#         How many knots in the spline. The default is 2.
#     z : array, optional
#         If present, attributes used to weight the spline fitting. The default is None.
#     k1 : int, optional
#         Degree of spline. The default is 1.

#     Returns
#     -------
#     spl2 : scipy.interpolate.UnivariateSpline
#         Spline mapping x to y.

#     """
#     ### like above but initial guess is just a straight line from [min_x,min_] to [max_x,max_y]
#     if z is None:
#         z= np.ones_like(x)
#     y_exists = np.isfinite(y)
#     x_exists = np.isfinite(x)*y_exists
#     x=np.array(x)[x_exists]
#     y=np.array(y)[x_exists]
#     z=np.array(z)[x_exists]
#     y_range = np.max(y)-np.min(y)
#     x_range = np.max(x)-np.min(x)
#     sorted_idxs = np.argsort(x)
#     sort_x = np.array(x)[sorted_idxs]
#     sort_y = np.array(y)[sorted_idxs]
#     sort_z = np.array(z)[sorted_idxs]
#     knots = quantiles(sort_x,n=n_knots)
#     # spl = spline(sort_x,sort_y,knots,w=sort_z,k=1)
#     # plt.scatter(x,y,s=1)
#     # plt.scatter(x,spl(x),s=1)
#     # plt.plot(x,((y_range/x_range)*x)+min(y)-((y_range/x_range)*min(x)))
#     _bool = np.abs((((y_range/x_range)*sort_x)+min(y)-((y_range/x_range)*min(x)))-sort_y)<(y_range/4)
#     # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
#     # find outliers and remove; points over 1/4 of the y range away from prediction
#     # _bool = np.abs(spl(sort_x)-sort_y)<(y_range/4)
#     # knots = quantiles(sort_x,n=4)
#     spl2 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
#     # spl2 = UnivariateSpline(sort_x,sort_y)
#     # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
#     # plt.scatter(x,spl2(x),s=1)
    
#     _bool = np.abs(spl2(sort_x)-sort_y)<(y_range/8)
    
#     # knots = quantiles(sort_x,n=n_knots)
#     spl3 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
#     # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
#     # plt.scatter(sort_x[_bool],spl3(sort_x[_bool]),s=1)
    
#     return spl3



def lowess_fit(x,y,frac=.2, it=3):
    
    """    
    Fit LOWESS regression line mapping x to y
    
    Parameters
    ----------
    x : array
        Series of x values.
    y : array
        Series of y values.
    frac : float, optional
        Fraction of the data used for each estimate . The default is .2.
    it : int, optional
        Number of rewightings to perform. The default is 3.

    Returns
    -------
    f: scipy.interpolate.interp1d
    Mapping from x to y

    """
    # plt.scatter(x,y,s=1)
    
    lowess = sm.nonparametric.lowess(y, x, frac=frac,it=it)
    
    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]
    
    # run scipy's interpolation. There is also extrapolation I believe
    f = interp1d(lowess_x, lowess_y, bounds_error=False,fill_value=(min(lowess_y),max(lowess_y)))
    
    return f


    
    
def closest_spec(dia_rt_mzwin, mz, rt):
    """
    Find which window is closest to the desired m/z and RT 

    Parameters
    ----------
    dia_rt_mzwin : array
        3D Array of bottom of mz window, top of mz window, mz window retention time.
    mz : float
        Desired mz.
    rt : float
        Desired retention time.

    Returns
    -------
    int
        Index of the closest window/spectrum.

    """
    contender_idxs = np.where((dia_rt_mzwin[:,1] < mz) & (mz < dia_rt_mzwin[:,2]))[0]
    
    if contender_idxs.size == 0:  # More efficient size check
        return 0
    
    contenders = dia_rt_mzwin[contender_idxs, 0]  # Extract only necessary column
    closest_idx = contender_idxs[np.argmin(np.abs(contenders - rt))]
    
    return closest_idx
    
def gaussian(x, amplitude, mean, stddev):
    """
    Get y values for gaussian distribution

    Parameters
    ----------
    x : array
        Array of x values.
    amplitude : float
        Scale factor for amplitude of distribution.
    mean : float
        Mean of gaussian distribution.
    stddev : float
        Standard deviation of gaussia distribution.

    Returns
    -------
    array
        y values correspond to the input x values for the gaussian with parameters given.

    """
    return (amplitude/ (np.abs(stddev) * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stddev) ** 2)


# NB: Need to make the folowing changes to this code
## if there is a background uniform distribution, it will not fit the gaussian well
# therefore we can subtract the min val in all bins fram all bins and then fit
### This seems to work for some data but need to robustly test
def fit_gaussian(data,init_std=None,bin_n=50):
    """
    Given a distribution of data points, try to fit a gaussian, then return the parameters that define it

    Parameters
    ----------
    data : array
        Distribution of points that aproximate a gaussian distribution.
    init_std : float, optional
        Initial guess at the standard deviation of the gaussian. The default is None.
    bin_n : int, optional
        Number of bins used to fit the distribution. The default is 50.

    Returns
    -------
    list
        Parameters defing the fitted gaussian; amplitude, mean, standard deviation

    """
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



def fit_errors(errors,limit=10,percentile=.999):
    """
    Given a distribution of errors, find if if is best explained by gaussian/exponential.
    When best, return the boundary that defines the provided percentile.

    Parameters
    ----------
    errors : array
        Array of errors.
    limit : float, optional
        Maximum error that is considered. The default is 10.
    percentile : float [0,1], optional
        Percentile to define the boundary. The default is .999.

    Returns
    -------
    boundary : float
        Value below which 'percentile' errors fall.

    """
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
        logger.info("Fitted Exponential to RT errors")
    else:
        scale_param = mad*1.4826
        boundary = stats.halfnorm.ppf(percentile,loc=0,scale=scale_param)
        logger.info("Fitted Gaussian to RT errors")
        
    # logger.info(boundary)
    return boundary

    
    




##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


# def fit_without_features(dia_spectra, librarySpectra):

#     all_keys = list(librarySpectra)
#     rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in librarySpectra.values()])
    
#     # Adjust partitioning based on available data
#     totalIC = np.array([np.sum(i.intens) for i in dia_spectra.ms2scans])
#     total_scans = len(totalIC)
    
#     # Dynamically adjust number of partitions based on data size
#     num_partition = min(10, max(1, total_scans // 10))  # At least 1 partition, at most 10
    
#     if num_partition > 0 and total_scans > 0:
#         # Calculate desired scans per partition
#         desired_per_partition = min(total_scans // num_partition, 
#                                    config.n_most_intense // num_partition)
        
#         split_size = max(1, int(np.ceil(total_scans/num_partition)))
#         split_tic = [totalIC[i*split_size:min(total_scans, (i+1)*split_size)] for i in range(num_partition)]
        
#         # Only take as many as available in each partition
#         split_top_n = []
#         for idx, tics in enumerate(split_tic):
#             if len(tics) > 0:  # Only process non-empty partitions
#                 # Take min of desired or available
#                 n_to_take = min(len(tics), desired_per_partition)
#                 if n_to_take > 0:
#                     split_top_n.append((np.argsort(-tics)+(idx*split_size))[:n_to_take])
        
#         if split_top_n:  # If we have any results
#             top_n = np.concatenate(split_top_n)
#         else:
#             # Fallback if partitioning fails
#             top_n = np.random.choice(np.arange(total_scans), 
#                                     min(total_scans, config.n_most_intense), 
#                                     replace=False)
#     else:
#         # Fallback for very small datasets
#         top_n = np.random.choice(np.arange(total_scans), 
#                                 min(total_scans, config.n_most_intense), 
#                                 replace=False)
   
    

    

#     top_n_spectra = [dia_spectra.ms2scans[i] for i in top_n]



#     ### redefine "top_n_spectra" to evenly span Rt and m/z
#     np.random.seed(0)
#     #top_n = np.random.choice(np.arange(len(ms2spectra)),config.n_most_intense,replace=False)
    
    
#     fit_outputs=[]
    
    frags = []
    for idx in tqdm.trange(len(top_n)):
            fit_output = fit_to_lib(top_n_spectra[idx],
                                    library=librarySpectra,
                                    rt_mz=rt_mz,
                                    all_keys=all_keys,
                                    dino_features=None,
                                    rt_filter=False,
                                    return_frags=False,
                                    ms1_spectra = dia_spectra.ms1scans,
                                    frac_matched=.8,
                                    rt_tol = config.rt_tol,
                                    ms1_tol = config.ms1_tol,
                                    mz_tol = config.mz_tol,
                                    )
            fit_outputs.append(fit_output)
    
#     return fit_outputs                                    

def fit_with_features(dia_spectra, librarySpectra, dino_features):
    
    all_keys = list(librarySpectra)
    rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in librarySpectra.values()])
    
    all_dia_rt = [i.RT for i in dia_spectra.ms2scans]
    all_dia_windows = np.array([i.ms1window for i in dia_spectra.ms2scans])
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
    # logger.info("Finding correct spectra")
    # lf_spectra = [np.argmin(np.abs(np.array(all_dia_rt)-i)) for i in lf_rt]
    dia_rt_mzwin = np.array([[i.RT,*i.ms1window] for i in dia_spectra.ms2scans])
    lf_spectra = [closest_spec(dia_rt_mzwin,i,j) for i,j in zip(lf_mz,lf_rt)] 
    
    fit_outputs=[]
    frags = []
    GUI_print_idxs = [int(((len(lf_spectra)-1)/10)*y) for y in range(1,11)]
    for idx in tqdm.trange(len(lf_spectra)):
        if config.ran_from_GUI:
            if idx in GUI_print_idxs:
                frac_done = (GUI_print_idxs.index(idx)+1) * 10
                logger.info(f"Initial Seach - {frac_done}%")
            
        fit_output = fit_to_lib(dia_spectra.ms2scans[int(lf_spectra[idx])],
                                    library=librarySpectra,
                                    rt_mz=rt_mz,
                                    all_keys=all_keys,
                                    dino_features=None,
                                    rt_filter=False,
                                    ms1_mz=lf_mz[idx],
                                    ms1_spectra = dia_spectra.ms1scans,
                                    frac_matched=.8, ## NB: this may be selcting for smaller peptides
                                    rt_tol = config.rt_tol,
                                    ms1_tol = config.ms1_tol,
                                    mz_tol = config.mz_tol
                                    )
        fit_outputs.append(fit_output)
        
    top_n_spectra = [dia_spectra.ms2scans[i] for i in lf_spectra]
    
    return fit_outputs, top_n_spectra, large_feature_indices, lf_mz


def process_prelim_search(fit_outputs,
                          librarySpectra,
                          top_n_spectra,
                          dino_features,
                          large_feature_indices,
                          lf_mz
                          ):
    
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
            # if ms2:
            #     lc_frags_errors.append(frags[idx][0][max_id])
            #     lc_frags.append(frags[idx][1][max_id])
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
    
    
    all_output_df = pd.DataFrame([j for i in output for j in i if j[0]>min_int],columns=names[:len(output[0][0])])
    all_output_df["lib_rt"] = np.array([librarySpectra[i[0]]["iRT"] for i in all_id_rt])
    
    all_frag_cosines = np.array([fragment_cor(all_output_df,i) for i in range(len(all_output_df))])
    all_frag_cosines_p = np.array([fragment_cor(all_output_df,i,fn="p") for i in range(len(all_output_df))])
    # plt.scatter(all_lib_rts,[i[1] for i in all_id_rt],label="Original_RT",s=1)
    all_output_df["frag_cosines"] = all_frag_cosines
    all_output_df["frag_cosines_p"] = all_frag_cosines_p

    all_frag_errors = [unstring_floats(mz) for mz in all_output_df.frag_errors]
    all_median  = np.median(np.concatenate([i for i in all_frag_errors]))
    all_output_df["med_frag_error"] = [np.median(np.abs(all_median-i)) for i in all_frag_errors]
    all_output_df["stripped_seq"]=np.array([re.sub("Decoy_","",re.sub("\(.*?\)","",i)) for i in all_output_df["seq"]])
    all_output_df["last_aa"]=[i[-1] for i in all_output_df.stripped_seq]

    
    output_df = pd.DataFrame([i[j] for i,j in zip(output,max_ids)],columns=names[:len(output[0][0])])
    output_df["lib_rt"] = np.array([max_coeff_rt(i) for i in output])
    
    frag_cosines = np.array([fragment_cor(output_df,i) for i in range(len(output_df))])
    frag_cosines_p = np.array([fragment_cor(output_df,i,fn="p") for i in range(len(output_df))])

    # plt.scatter(all_lib_rts,[i[1] for i in all_id_rt],label="Original_RT",s=1)
    output_df["frag_cosines"] = frag_cosines
    output_df["frag_cosines_p"] = frag_cosines_p

    frag_errors = [unstring_floats(mz) for mz in output_df.frag_errors]
    median  = np.median(np.concatenate([i for i in frag_errors]))
    output_df["med_frag_error"] = [np.median(np.abs(median-i)) for i in frag_errors]
    
    output_df["stripped_seq"]=np.array([re.sub("Decoy_","",re.sub("\(.*?\)","",i)) for i in output_df["seq"]])
    output_df["last_aa"]=[i[-1] for i in output_df.stripped_seq]

    return output_df, all_output_df, id_keys, feature_mzs

def empirical_fit(output_df,results_folder=None):
    """
    Filter data by confidence then fit LOWESS to empirical RT

    Parameters
    ----------
    output_df : pd.DataFrame
        Dataframe of IDs from the preliminary search.

    Returns
    -------
    cor_filter : np.ndarray(bool)
        DESCRIPTION.
    emp_rt_spl : TYPE
        DESCRIPTION.

    """

    logger.info("")
    logger.info("Filtering IDs from initial search")

    for feature_percentile in range(20, 80, 5):

        cor_filter = np.logical_and.reduce(
            [output_df[feat] > np.percentile(output_df[feat], feature_percentile)
             for feat in [
                 "hyperscore",
                 "frag_cosines_p",
                 "frag_cosines_p",
                 "manhattan_distances",
             ]]
            +
            [output_df[feat] < np.percentile(output_df[feat], 100 - feature_percentile)
             for feat in [
                 "scribe_scores",
                 "gof_stats",
                 "max_matched_residuals",
                 "med_frag_error",
             ]]
        )

        f = lowess_fit(output_df.lib_rt[cor_filter],
                       output_df.rt[cor_filter],
                       .1)

        plt.subplots()
        plt.scatter(output_df.lib_rt[cor_filter],
                    output_df.rt[cor_filter], s=1)
        plt.scatter(output_df.lib_rt[cor_filter],
                    f(output_df.lib_rt[cor_filter]), s=1)
        plt.title(str(feature_percentile))
        if results_folder is not None:
            plt.savefig(results_folder + f"/Percentile_{feature_percentile}.png",
                        dpi=600, bbox_inches="tight")
        plt.close()

        first_rt_diffs = f(output_df.lib_rt) - output_df.rt

        rt_amplitude, rt_mean, rt_stddev = fit_gaussian(first_rt_diffs[cor_filter])
        first_rt_tolerance = 4 * np.abs(rt_stddev)

        bad_IDs = (
                np.abs(first_rt_diffs) >
                np.min([first_rt_tolerance, np.ptp(output_df.rt) / 5])
        )[cor_filter]
        outside_ratio = bad_IDs.sum() / len(bad_IDs)

        # GMM on residuals from current filter
        res = first_rt_diffs[cor_filter]
        weights, sigmas = fit_zero_mean_gmm_1d(res, n_components=2)

        order = np.argsort(sigmas)
        sigmas = sigmas[order]
        weights = weights[order]

        num = weights * norm.pdf(0.0, loc=0.0, scale=sigmas)
        pep0 = num[0] / num.sum()

        # rt error + mixture plot, saved per percentile
        if results_folder is not None:
            plot_rt_residuals_mixture(
                residuals=res,
                feat=feature_percentile,
                weights=weights,
                sigmas=sigmas,
                results_folder=results_folder
            )

        logger.info(
            f"Testing Percentile: {feature_percentile}, "
            f"Ratio: {outside_ratio:.4f}, #IDs: {cor_filter.sum()}, PEP(0): {pep0:.4f}"
        )

        # new + existing stopping criteria
        if (pep0 >= 0.95 or
                outside_ratio < 0.05 or
                (cor_filter.sum() - bad_IDs.sum()) < 800):
            break

    logger.debug(f"{feature_percentile} {np.round(outside_ratio, 4)} {cor_filter.sum()}")

    cor_filter = np.logical_and(
        cor_filter,
        np.abs(first_rt_diffs) < first_rt_tolerance
    )

    emp_rt_spl = lowess_fit(
        np.array(output_df.lib_rt)[cor_filter],
        np.array(output_df.rt)[cor_filter],
        .02
    )

    return cor_filter, emp_rt_spl


def fit_zero_mean_gmm_1d(x, n_components=2, max_iter=200, tol=1e-6):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.shape[0]

    # init by splitting on |x| to define narrow and wide components
    abs_x = np.abs(x)
    split = np.median(abs_x)
    resp = np.zeros((n, n_components))
    resp[:, 0] = abs_x <= split
    resp[:, 1] = abs_x > split
    resp /= resp.sum(axis=1, keepdims=True)

    weights = resp.mean(axis=0)
    variances = np.array([
        np.sum(resp[:, k] * x**2) / np.sum(resp[:, k])
        for k in range(n_components)
    ])
    sigmas = np.sqrt(variances)

    def log_likelihood():
        pdfs = np.stack(
            [weights[k] * norm.pdf(x, loc=0.0, scale=sigmas[k])
             for k in range(n_components)],
            axis=1,
        )
        return np.sum(np.log(pdfs.sum(axis=1) + 1e-300))

    prev_ll = log_likelihood()

    # iterate until convergence
    for _ in range(max_iter):
        pdfs = np.stack(
            [weights[k] * norm.pdf(x, loc=0.0, scale=sigmas[k])
             for k in range(n_components)],
            axis=1,
        )
        denom = pdfs.sum(axis=1, keepdims=True) + 1e-300
        resp = pdfs / denom

        weights = resp.mean(axis=0)
        variances = np.array([
            np.sum(resp[:, k] * x**2) / np.sum(resp[:, k])
            for k in range(n_components)
        ])
        sigmas = np.sqrt(variances)

        ll = log_likelihood()
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return weights, sigmas


def plot_rt_residuals_mixture(residuals,
                              feat,
                              weights=None,
                              sigmas=None,
                              results_folder=None):
    """
    Make the RT residual histogram + 2-comp zero-mean GMM overlay and PEP(0) text.
    """
    res = np.asarray(residuals, dtype=float)
    res = res[~np.isnan(res)]

    # sort so comp 0 is narrow
    order = np.argsort(sigmas)
    sigmas = sigmas[order]
    weights = weights[order]

    # posterior at x=0 for the narrow component
    num = weights * norm.pdf(0.0, loc=0.0, scale=sigmas)
    pep0 = num[0] / num.sum()

    max_abs = np.percentile(np.abs(res), 99)
    bins = np.linspace(-max_abs, max_abs, 80)

    plt.figure(figsize=(6, 4))

    counts, edges, _ = plt.hist(
        res,
        bins=bins,
        density=True,
        alpha=0.5,
        edgecolor="black",
        linewidth=0.5,
    )

    x = np.linspace(edges[0], edges[-1], 500)

    comp_pdfs = [
        weights[k] * norm.pdf(x, loc=0.0, scale=sigmas[k])
        for k in range(2)
    ]
    mixture_pdf = comp_pdfs[0] + comp_pdfs[1]

    plt.plot(x, mixture_pdf, label="mixture", linewidth=2)
    plt.plot(x, comp_pdfs[0], linestyle="--",
             label=f"comp 1 (σ={sigmas[0]:.3f})")
    plt.plot(x, comp_pdfs[1], linestyle="--",
             label=f"comp 2 (σ={sigmas[1]:.3f})")

    ax = plt.gca()
    ax.axvline(0, color="black", linewidth=1)

    ax.text(
        0.98, 0.95,
        f"PEP(0) = {pep0:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )

    plt.legend()
    plt.xlabel("fit_rt - rt")
    plt.ylabel("density")
    plt.title(f"RT Residuals After Empirical Alignment {feat}")
    plt.tight_layout()

    if results_folder is not None:
        plt.savefig(
            results_folder + f"/rt_residuals_p{feat}.png",
            dpi=600,
            bbox_inches="tight",
        )

    else:
        plt.close()

    return pep0


def cdf_data(rt_diffs,limit=3):
    """
    Convert differnces to format for cdf cacluation    

    Parameters
    ----------
    rt_diffs : array
        Retention time errors.
    limit : float, optional
        Exclude errors greater than this value. The default is 3.

    Returns
    -------
    data : array
        Filtered and ordered retention time errors.
    p : array
        Corresponding proportions less than each error.
    cdf_auc : float
        Area under the CDF described by data and p.

    """
    data = np.sort(np.abs(rt_diffs)[np.abs(rt_diffs) < limit])
    data = np.append(data,limit)
    p = np.arange(len(data)) / (len(data) - 1)
    cdf_auc = auc(data,p)
    
    return data, p, cdf_auc
        

def alignment_plots(filtered_output, 
                    orig_spl, 
                    rt_spl,
                    f_rt_mz,
                    mz_spl,
                    rt_dist_params,
                    results_folder=None):
        ##plot RT alignment
        plt.subplots()
        plt.scatter(filtered_output.lib_rt,np.array(filtered_output.rt),label="Original_RT",s=.1)
        plt.scatter(filtered_output.lib_rt,orig_spl(filtered_output.lib_rt),label="Predicted_RT",s=1)
        # plt.legend()
        plt.xlabel("Library RT")
        plt.ylabel("Observed RT")
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/OriginalRTfit.png",dpi=600,bbox_inches="tight")
        
        
        ##plot RT alignment
        plt.subplots()
        plt.scatter(filtered_output.updated_lib_rt,np.array(filtered_output.rt),label="Original_RT",s=.1)
        plt.scatter(filtered_output.updated_lib_rt,rt_spl(filtered_output.updated_lib_rt),label="Predicted_RT",s=1)
        # plt.legend()
        plt.xlabel("Updated Library RT")
        plt.ylabel("Observed RT")
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/RTfit.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        plt.scatter(filtered_output.updated_lib_rt,
                    (filtered_output.rt-rt_spl(filtered_output.updated_lib_rt)),label="Original_RT",s=.1)
        min_rt = np.min(filtered_output.updated_lib_rt)
        max_rt = np.max(filtered_output.updated_lib_rt)
        plt.plot([min_rt,max_rt],[0,0],color="r",linestyle="--",alpha=.5)
        plt.plot([min_rt,max_rt],[config.opt_rt_tol,config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
        plt.plot([min_rt,max_rt],[-config.opt_rt_tol,-config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
        # plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
        # plt.legend()
        lims = plt.ylim()
        y_lim = min(10,np.max(np.abs(lims)))
        # plt.ylim(-y_lim,y_lim)
        plt.xlabel("Updated Library RT")
        plt.ylabel("RT Residuals")
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/RtResidual.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        vals,bins,_ = plt.hist((filtered_output.rt-orig_spl(filtered_output.lib_rt)),100,density=True,alpha=.5,label="Original RT")
        vals,bins,_ = plt.hist((filtered_output.rt-rt_spl(filtered_output.updated_lib_rt)),100,density=True,alpha=.5,label="Updated RT")
        plt.plot(np.linspace(-config.opt_rt_tol,config.opt_rt_tol,100),gaussian(np.linspace(-config.opt_rt_tol,config.opt_rt_tol,100), *rt_dist_params),label="Updated RT fit")
        plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals),color="r")
        # plt.vlines([-4*rt_stddev,4*rt_stddev],0,max(vals),color="g")
        plt.text(config.opt_rt_tol,max(vals),np.round(config.opt_rt_tol,2))
        plt.xlabel("RT difference")
        plt.ylabel("Frequency")
        plt.legend()
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/RTdiff.png",dpi=600,bbox_inches="tight")
        
        
        ##plot mz rt alignment
        plt.subplots()
        plt.scatter(filtered_output.updated_lib_rt,np.array(filtered_output.mz_diffs),label="Original_MZ",s=1,alpha=min(1,5/((len(np.array(filtered_output.rt))//1000)+1)))
        plt.scatter(filtered_output.updated_lib_rt,f_rt_mz(filtered_output.updated_lib_rt),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("Updated RT")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/MZrtfit.png",dpi=600,bbox_inches="tight")
        
    
        ##plot mz alignment
        plt.subplots()
        plt.scatter(np.array(filtered_output.mz),(filtered_output.mz_diffs-f_rt_mz(filtered_output.updated_lib_rt)),label="Original_MZ",s=1,alpha=min(1,5/((len(np.array(filtered_output.updated_lib_rt))//1000)+1)))
        plt.scatter(filtered_output.mz,mz_spl(filtered_output.mz),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("m/z")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/MZfit.png",dpi=600,bbox_inches="tight")
        
        
        
        ## plot mz diff
        plt.subplots()
        plt.hist(np.array(filtered_output.mz_diffs),100,label="Theoretical m/z")
        # plt.hist(((np.array(id_mzs)+np.array(filtered_output.mz_diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs,100,alpha=.5)
        # plt.hist(((np.array(id_mzs)+np.array(filtered_output.mz_diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
        vals,bins,_ = plt.hist((filtered_output.mz_diffs-mz_spl(filtered_output.mz)-f_rt_mz(filtered_output.updated_lib_rt)),100,alpha=.5,label="Updated m/z")
        plt.vlines([-config.opt_ms1_tol,config.opt_ms1_tol],0,max(vals)*.8,color="r")
        # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
        plt.text(config.opt_ms1_tol,max(vals)*.8,f"{np.round(1e6*config.opt_ms1_tol,2)} ppm")
        plt.xlabel("m/z difference (relative)")
        plt.ylabel("Frequency")
        plt.legend()
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/MZdiff.png",dpi=600,bbox_inches="tight")
    
    
        plt.close("all")
        
        
        
def cdf_plots(emp_data,emp_p,percentile,boundary,pred_data=None,pred_p=None,results_folder=None):  
        ### Plot the CDFs with elbow points
        
        plt.subplots()
        plt.figure(figsize=(8, 5))
        plt.plot(emp_data, emp_p, label="Original CDF", linestyle='-')
        
        # plt.scatter(elbow_emp_x, elbow_emp_y, color='blue', label=f'Original Elbow at {elbow_emp_x:.2f}', zorder=3)
        # plt.scatter(elbow_pred_x, elbow_pred_y, color='red', label=f'Finetuned Elbow at {elbow_pred_x:.2f}', zorder=3)
        
        
        # emp_abs_errors_med = np.median(np.abs(all_emp_diffs[all_emp_diffs<limit]-np.median(all_emp_diffs[all_emp_diffs<limit])))
        emp_abs_errors_med = np.median(emp_data)
        plt.plot(emp_data,stats.expon.cdf(emp_data,loc=0,scale=emp_abs_errors_med/np.log(2)),linestyle="--",color=colours[0],label="Emp Expon CDF")
        emp_exp_999 = stats.expon.ppf(percentile,scale=emp_abs_errors_med/np.log(2))
        plt.scatter([emp_exp_999], [percentile],c=colours[0],label=f"Emp Expon {percentile}: {emp_exp_999:.2f}",marker="*")
        plt.plot(emp_data,stats.halfnorm.cdf(emp_data,loc=0,scale=np.power(emp_abs_errors_med*1.4826,1)),linestyle=":",color=colours[0],label="Emp Norm CDF")
        emp_gauss_999 = stats.halfnorm.ppf(percentile,scale=emp_abs_errors_med*1.4826)
        plt.scatter([emp_gauss_999], [percentile],c=colours[0],label=f"Emp Norm {percentile}: {emp_gauss_999:.2f}")
        if pred_data is not None:
            plt.plot(pred_data, pred_p, label="Finetuned CDF", linestyle='-')
            # pred_abs_errors_med = np.median(np.abs(all_pred_diffs[all_pred_diffs<limit]-np.median(all_pred_diffs[all_pred_diffs<limit])))
            pred_abs_errors_med = np.median(pred_data)
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
        if results_folder is not None:
            plt.savefig(results_folder+"/RTelbows.png",dpi=600,bbox_inches="tight")
        
        plt.close("all")

def MZRTfit(dia_spectra,librarySpectra,dino_features,mz_tol,ms1=False,results_folder=None,ms2=False):
    """
    Perform a preliminary search of the specrta to align the library mz and RT values

    Parameters
    ----------
    dia_spectra : src.utisl.io.load_files.SpectrumFile
        Spectra to align the library to.
    librarySpectra : dict
        Spectrum library.
    dino_features : pd.DataFrame
        Dataframe of features identified using Biosaur2.
    mz_tol : float
        MS1 mz tolerance.
    ms1 : bool, optional
        DESCRIPTION. The default is False.
    results_folder : String, optional
        If provided, where to save the logs/figures. The default is None.
    ms2 : bool, optional
        (Not active) Whether to align at MS2 level. The default is False.

    Returns
    -------
    (rt_spl, mz_func), updatedLibrary
    
        rt_spl: Spline fitting library retention time to observed values
        
        mz_func: Function that aligns library precuror m/z to observed values
        
        updatedLibrary: Copy of the library with updated Retention time if fine-tuning
        

    """
    
    ## for testing
    # mz_tol,ms1,results_folder,ms2 = (config.ms1_tol,False,None,False) 

    
    config.n_most_intense_features = int(1e5) 
    
    # Calculate scans_per_cycle safely
    if len(dia_spectra.ms1scans) > 0:
        scans_per_cycle = max(1, round(len(dia_spectra.ms2scans)/len(dia_spectra.ms1scans)))
    else:
        scans_per_cycle = 1

    logger.info("")
    logger.info("Starting Initial Search")
    # print(f"Fitting the {config.n_most_intense} most intense spectra")
    
    ms1spectra = dia_spectra.ms1scans
    ms2spectra = dia_spectra.ms2scans
    # """ 
    ms1_rt = np.array([i.RT for i in ms1spectra])
    
    all_keys = list(librarySpectra)
    rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in librarySpectra.values()])
    

    # if dino_features is None:
    #     fit_outputs = fit_without_features(dia_spectra, librarySpectra)
    # 
    # else:
    fit_outputs, top_n_spectra, large_feature_indices, lf_mz = fit_with_features(dia_spectra, librarySpectra, dino_features)
        
    
    #################################################################################
    
    ########################################################################
     
    output_df, all_output_df, id_keys, feature_mzs =  process_prelim_search(fit_outputs,
                                                                              librarySpectra,
                                                                              top_n_spectra,
                                                                              dino_features,
                                                                              large_feature_indices,
                                                                              lf_mz
                                                                              )
    

    if results_folder is not None:
        output_df.to_csv(results_folder+"/firstSearch.csv", index=False)
    # output_df = pd.DataFrame([j for i in output for j in i  if j[0]>min_int],columns=names[:len(output[0][0])])
    
    
    cor_filter = np.ones_like(output_df.rt,dtype=bool)
    if dino_features is not None:
        
        cor_filter, emp_rt_spl = empirical_fit(output_df,results_folder=results_folder)
    else:
        
        hyper_cutoff = np.percentile(all_output_df.hyperscore,80)
        all_cor_filter = all_output_df.hyperscore>hyper_cutoff
        cor_filter = output_df.hyperscore>hyper_cutoff
        # emp_rt_spl = initstepfit(np.array(all_output_df.lib_rt)[all_cor_filter],np.array(all_output_df.rt)[all_cor_filter],1,z=np.array(all_output_df.hyperscore)[all_cor_filter])
        emp_rt_spl = lowess_fit(np.array(all_output_df.lib_rt)[all_cor_filter],np.array(all_output_df.rt)[all_cor_filter])
        
    
    
    
    
    percentile = config.rt_percentile
    
    limit=3 ## exlcude RT diffs larger than this (outliers)
    
    ###############################################################
    ####### fine tuning
    ###############################################################
    
    if not config.args.use_emp_rt:
        ## filter for only a single channel for each
        logger.info("")
        logger.info("Trying RT Prediction")
        seq_rt = {}
        for s,rt in zip(np.array(id_keys)[cor_filter],np.array(output_df.rt)[cor_filter]):
            key=librarySpectra[(s[0],float(s[1]))]["seq"]
            seq_rt.setdefault(key,[])
            seq_rt[key].append(rt)
        # exclude those with ambiguity (differences between channels/charge states)
        filtered_seq_rt = {s:np.median(seq_rt[s]) for s in seq_rt if np.ptp(seq_rt[s])<1}
            
        ## use observed rt for fine_tuning
        # grouped_df = pd.DataFrame({'Stripped.Sequence':[librarySpectra[(i[0],float(i[1]))]["seq"] for i in id_keys],"RT":[i for i in np.array(dia_rt)]})[cor_filter]
        grouped_df =  pd.DataFrame({'Stripped.Sequence':[s for s in filtered_seq_rt],"RT":[filtered_seq_rt[s] for s in filtered_seq_rt]})
        data_split, models, convertor = fine_tune_rt(grouped_df,qc_plots=True,results_path=results_folder)
        
        
        all_emp_diffs = (emp_rt_spl(output_df.lib_rt)-np.array(output_df.rt))[cor_filter]
        
        
        lib_seqs = [one_hot_encode_sequence(librarySpectra[key]["seq"]) for key in id_keys]
        predicted_rts = convertor(np.mean([model.predict(np.array(lib_seqs)) for model in models],axis=0).flatten())
    
        validation_rts = convertor(np.mean([model.predict(np.array(data_split[1])) for model in models],axis=0).flatten())
        validation_rt_diffs = data_split[3]-validation_rts
        
        pred_rt_spl = lowess_fit(predicted_rts[cor_filter],
                               np.array(output_df.rt)[cor_filter] ,frac=.2)
        
        all_pred_diffs = (pred_rt_spl(predicted_rts) - np.array(output_df.rt))[cor_filter]
        
        all_pred_diffs = validation_rt_diffs
        
        emp_data, emp_p, emp_cdf_auc = cdf_data(all_emp_diffs,limit=limit)
        pred_data, pred_p, pred_cdf_auc = cdf_data(all_pred_diffs,limit=limit)
        
        
        updatedLibrary = copy.deepcopy(librarySpectra)
        all_lib_keys = list(librarySpectra)
        
        
        ###### Check if fine-tuning iproves alignment
        
        if pred_cdf_auc>emp_cdf_auc: ## Predictions are better
            # boundary = elbow_pred_x
            logger.info("Fine Tuned Library Chosen")
            boundary = fit_errors(all_pred_diffs,limit,percentile)
            rt_spl = pred_rt_spl
            all_lib_seqs = [one_hot_encode_sequence(updatedLibrary[key]["seq"]) for key in all_lib_keys]
            all_new_lib_rts = convertor(np.mean([model.predict(np.array(all_lib_seqs)) for model in models],axis=0).flatten())
            
            for key,rt in zip(all_lib_keys,all_new_lib_rts):
                updatedLibrary[key]["iRT"] = rt
                
        else: ### empirical are better
            # boundary = elbow_emp_x
            logger.info("Empirical Library Chosen")
            boundary = fit_errors(all_emp_diffs,limit,percentile)
            ## keep the library RTs and splines the same
            rt_spl = emp_rt_spl
        
        
        
    ###############################################################
    ####### NO fine tuning
    ###############################################################
    
    else:

        logger.info("Using Empirical w/o Fine Tuning")
        updatedLibrary = copy.deepcopy(librarySpectra)
        all_lib_keys = list(librarySpectra)
        rt_spl = emp_rt_spl
        all_emp_diffs = (emp_rt_spl(output_df.lib_rt)-np.array(output_df.rt))[cor_filter]
        
        pred_data = pred_p = None
        emp_data = np.sort(np.abs(all_emp_diffs)[np.abs(all_emp_diffs) < limit])
        emp_data = np.append(emp_data,limit)
        emp_p = np.arange(len(emp_data)) / (len(emp_data) - 1)
        emp_cdf_auc = auc(emp_data,emp_p)
        boundary = fit_errors(all_emp_diffs,limit,percentile)
    
    new_lib_rt = np.array([updatedLibrary[k]["iRT"] for k in id_keys])
    converted_rt = rt_spl([updatedLibrary[k]["iRT"] for k in id_keys])
    
    rt_amplitude, rt_mean, rt_stddev = fit_gaussian((output_df.rt-converted_rt)[cor_filter])
   
    
   
    ################################################
    ########### correct mz errors wrt RT    ########
    ################################################
    
    
    
    if dino_features is None:
        resp_ms1scans = [closest_ms1spec(output_df.rt[i], ms1_rt) for i in range(len(output_df.rt))]
        diffs = [closest_peak_diff(mz, ms1spectra[i].mz) for i,mz in zip(resp_ms1scans,output_df.mz)]
    else:
        diffs = np.array([(i-mz)/mz for i,mz in zip(feature_mzs,output_df.mz)])
    

    
    f_rt_mz = lowess_fit(new_lib_rt[cor_filter],np.array(diffs)[cor_filter],.02)
    
    # mz_spl = twostepfit(np.array(id_mzs)[rt_filter_bool],(diffs-f_rt_mz(dia_rt))[r t_filter_bool],1)
    mz_spl = lowess_fit(np.array(output_df.mz)[cor_filter],(diffs-f_rt_mz(new_lib_rt))[cor_filter])


    def mz_func(mz,rt):
        return mz+((mz_spl(mz)+f_rt_mz(rt))*mz)
    
    # orig_mzs = id_mzs+(diffs*np.array(id_mzs))
    # plt.hist(((mz_func(id_mzs,rts)-orig_mzs)/id_mzs)[rt_filter_bool],100)
    
    corrected_mz_diffs = (diffs-(f_rt_mz(new_lib_rt)+mz_spl(output_df.mz)))[cor_filter]
    mz_amplitude, mz_mean, mz_stddev = fit_gaussian(corrected_mz_diffs)
    
    # ### MS2 alignment
    # if ms2:
    #     all_frag_errors = np.concatenate(lc_frags_errors)
    #     all_frags = np.concatenate(lc_frags)
    #     ms2_spl = twostepfit(all_frags,all_frag_errors,1)
    #     def ms2_func(mz):
    #         return mz+(ms2_spl(mz)*mz)
        
    #     ms2_amplitude, ms2_mean, ms2_stddev = fit_gaussian(all_frag_errors-ms2_spl(all_frags))
    
    
    # ## 2D plane fitting
    # function = lin_func
    # parameters = curve_param(output_rts, id_mzs, diffs,func=function)
    
    # def mz_func(mz,rt):
    #     return mz+(function([rt,mz],*parameters)*mz)
    
    
    ################################################
    ########### Set Optimal Limits   ########
    ################################################
    
    
    new_rt_tol = boundary#4*np.abs(rt_stddev) 
    if config.args.user_rt_tol:
        logger.info("Using user specified RT tolerance")
        new_rt_tol = config.args.rt_tol
    logger.info(f"Optimized RT tolerance: {new_rt_tol}")
    config.opt_rt_tol = np.abs(new_rt_tol)
    
    
    new_ms1_tol = np.abs(4*mz_stddev)
    logger.info(f"Optimized MS1 tolerance: {new_ms1_tol}")
    logger.info("")
    
    
    if config.args.ms1_ppm!=0:
        logger.info(f"Using MS1 Tolerance provided: {config.args.ms1_ppm}ppm")
        new_ms1_tol=np.abs(config.args.ms1_ppm*1e-6)
    elif config.min_ms1_tol!=0 and config.min_ms1_tol>new_ms1_tol:
        logger.info(f"Exceeded minimum MS1 tolerance: {np.abs(config.min_ms1_tol)}")
        logger.info(f"Setting new MS1 tolerance: {np.abs(config.min_ms1_tol)}")
        new_ms1_tol=np.abs(config.min_ms1_tol)
        
    config.opt_ms1_tol  = new_ms1_tol
    
    # if ms2:
    #     new_ms2_tol = 5*ms2_stddev
    #     config.opt_ms2_tol  = new_ms2_tol
    
    
    ################################################################
    ########### Save the functions and Plot the alignment   ########
    ################################################################
    
    if results_folder is not None:
        
        ### Save functions
        with open(results_folder+"/rt_spl","wb") as dill_file:
            dill.dump(rt_spl,dill_file)
            
        with open(results_folder+"/mz_func","wb") as dill_file:
            dill.dump(mz_func,dill_file)
        
        # if ms2:
        #     with open(results_folder+"/ms2_func","wb") as dill_file:
        #         dill.dump(ms2_func,dill_file)
        
        output_df["updated_lib_rt"] = [updatedLibrary[k]["iRT"] for k in id_keys]
        output_df["mz_diffs"] = diffs
        filtered_output = output_df[cor_filter]
        alignment_plots(filtered_output, 
                            emp_rt_spl, 
                            rt_spl,
                            f_rt_mz,
                            mz_spl,
                            rt_dist_params=(rt_amplitude,rt_mean,rt_stddev),
                            results_folder=results_folder)
        
        cdf_plots(emp_data,emp_p,percentile,boundary,pred_data,pred_p,results_folder=results_folder)
        
        
        """
        ##plot RT alignment
        plt.subplots()
        plt.scatter(output_df.lib_rt[cor_filter],np.array(output_df.rt)[cor_filter],label="Original_RT",s=.1)
        plt.scatter(output_df.lib_rt,emp_rt_spl(output_df.lib_rt),label="Predicted_RT",s=1)
        # plt.legend()
        plt.xlabel("Library RT")
        plt.ylabel("Observed RT")
        # plt.show()
        plt.savefig(results_folder+"/OriginalRTfit.png",dpi=600,bbox_inches="tight")
        
        
        ##plot RT alignment
        plt.subplots()
        plt.scatter(np.array([updatedLibrary[k]["iRT"] for k in id_keys])[cor_filter],np.array(output_df.rt)[cor_filter],label="Original_RT",s=.1)
        plt.scatter([updatedLibrary[k]["iRT"] for k in id_keys],rt_spl([updatedLibrary[k]["iRT"] for k in id_keys]),label="Predicted_RT",s=1)
        # plt.legend()
        plt.xlabel("Updated Library RT")
        plt.ylabel("Observed RT")
        # plt.show()
        plt.savefig(results_folder+"/RTfit.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        plt.scatter(np.array([updatedLibrary[k]["iRT"] for k in id_keys])[cor_filter],
                    (output_df.rt-rt_spl([updatedLibrary[k]["iRT"] for k in id_keys]))[cor_filter],label="Original_RT",s=.1)
        min_rt = np.min([updatedLibrary[k]["iRT"] for k in id_keys])
        max_rt = np.max([updatedLibrary[k]["iRT"] for k in id_keys])
        plt.plot([min_rt,max_rt],[0,0],color="r",linestyle="--",alpha=.5)
        plt.plot([min_rt,max_rt],[config.opt_rt_tol,config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
        plt.plot([min_rt,max_rt],[-config.opt_rt_tol,-config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
        # plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
        # plt.legend()
        lims = plt.ylim()
        y_lim = min(10,np.max(np.abs(lims)))
        # plt.ylim(-y_lim,y_lim)
        plt.xlabel("Updated Library RT")
        plt.ylabel("RT Residuals")
        # plt.show()
        plt.savefig(results_folder+"/RtResidual.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        vals,bins,_ = plt.hist((output_df.rt-emp_rt_spl(output_df.lib_rt))[cor_filter],100,density=True,alpha=.5,label="Original RT")
        vals,bins,_ = plt.hist((output_df.rt-rt_spl([updatedLibrary[k]["iRT"] for k in id_keys]))[cor_filter],100,density=True,alpha=.5,label="Updated RT")
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
        plt.scatter(new_lib_rt[cor_filter],np.array(diffs)[cor_filter],label="Original_MZ",s=1,alpha=min(1,5/((len(np.array(output_df.rt)[cor_filter])//1000)+1)))
        plt.scatter(new_lib_rt,f_rt_mz(new_lib_rt),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("Updated RT")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        plt.savefig(results_folder+"/MZrtfit.png",dpi=600,bbox_inches="tight")
        
        

        ##plot mz alignment
        plt.subplots()
        plt.scatter(np.array(output_df.mz)[cor_filter],(diffs-f_rt_mz(new_lib_rt))[cor_filter],label="Original_MZ",s=1,alpha=min(1,5/((len(np.array(new_lib_rt)[cor_filter])//1000)+1)))
        plt.scatter(output_df.mz,mz_spl(output_df.mz),label="Predicted_MZ",s=1)
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
        vals,bins,_ = plt.hist((diffs-mz_spl(output_df.mz)-f_rt_mz(new_lib_rt))[cor_filter],100,alpha=.5)
        plt.vlines([-config.opt_ms1_tol,config.opt_ms1_tol],0,max(vals)*.8,color="r")
        # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
        plt.text(config.opt_ms1_tol,max(vals)*.8,f"{np.round(1e6*config.opt_ms1_tol,2)} ppm")
        plt.xlabel("m/z difference (relative)")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(results_folder+"/MZdiff.png",dpi=600,bbox_inches="tight")
    
    plt.close("all")
    # """
    
    # if ms2:
    #     return (rt_spl, mz_func, ms2_func), updatedLibrary
    # else:
    return (rt_spl, mz_func), updatedLibrary

###################################################################################################
###################################################################################################
###################################################################################################

def filter_rts_by_dense(rts,n_bins=20):
    """
    Given a list of retention times, trim the ends if these areas are not dense (e.g. wash)    

    Parameters
    ----------
    rts : array
        Input list of RTs.
    n_bins : int, optional
        Number of bins to consider. The default is 20.

    Returns
    -------
    Boolean array
        Which entries to consider; part of the dense region.

    """

    hist,bins = np.histogram(rts,n_bins)
    med = np.mean(hist[hist!=0])
    where_larger = np.where(hist>med/2)[0]
    smallest,largest = (bins[where_larger[0]],bins[min(where_larger[-1]+1,len(bins)-1)])
    return np.logical_and(np.greater(rts,smallest),np.less(rts,largest))


def get_multiples(id_keys, output_df):
    key_dict = {}
    for i, key in enumerate(id_keys):
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
                multiple_rts = np.array([output_df.rt[i] for i in key_pos])
                multiples_hyper.append(np.array([output_df.hyperscore[i] for i in key_pos]))
                multiples_coeff.append(np.array([output_df.coeff[i] for i in key_pos]))
                order = np.argsort(multiple_rts)
                order = np.arange(len(multiple_rts))
                multiples.append(multiple_rts[order])
                multiples_idxs.append(np.array(key_pos)[order])
                num_multiples.append(len(key_pos))
                # channels.append([re.findall("\(tag6-(\d+)\)",id_keys[i][0])[0] for i in key_pos])
                multiples_seqs.append([id_keys[i][0] for i in key_pos])
                multiples_zs.append([id_keys[i][1] for i in key_pos])
            searched.update(key)
    
    return multiples, num_multiples, multiples_idxs

# feature_percentile = 50
# if config.args.user_percentile:
#     print("Using user specified feature percentile for first search")
#     feature_percentile = config.args.initial_percentile
def get_df_filter(df,p=50):
    return np.logical_and.reduce([df[feat]>np.percentile(df[feat],p) for feat in ["hyperscore",
                                                                                             "frag_cosines_p",
                                                                                             "frag_cosines_p",
                                                                                              "manhattan_distances",
                                                                                             ]]+
                                   [df[feat]<np.percentile(df[feat],100-p) for feat in [
                                                                                                    "scribe_scores",
                                                                                                    "gof_stats",
                                                                                                    # "manhattan_distances",
                                                                                                    "max_matched_residuals",
                                                                                                    "med_frag_error"
                                                                                                    ]])

def split_timePlex(output_df,n_timeplex,rt_mz, id_keys, multiples_idxs):
    
    rt_spls = []
    t_vals = []
    t_seqs = []
    t_dfs = []
    filters = []
    converted_rts = []
    gaussian_fits = []
    for idx in range(n_timeplex):
        lib_rt_range = [np.percentile(rt_mz[:,0],5),np.percentile(rt_mz[:,0],95)]
        ### array of (obs_rt, lib_rt, hyperscore)
        t1 = np.array([[output_df.rt[i[idx]],output_df.lib_rt[i[idx]],output_df.hyperscore[i[idx]],output_df.mz[i[idx]],output_df.coeff[i[idx]],output_df.frac_lib_int[i[idx]]] for i in multiples_idxs if len(i)==n_timeplex and output_df.lib_rt[i[idx]]>lib_rt_range[0] and output_df.lib_rt[i[idx]]<lib_rt_range[1]])
        t1_s = [id_keys[i[idx]] for i in multiples_idxs if len(i)==n_timeplex and output_df.lib_rt[i[idx]]>lib_rt_range[0] and output_df.lib_rt[i[idx]]<lib_rt_range[1]]
        # t1 = np.array([[output_df.rt[i[idx]],output_df.lib_rt[i[idx]],output_hyper[i[idx]]] for i in multiples_idxs if len(i)==n_timeplex])
        t_df = output_df.iloc[[i[idx] for i in multiples_idxs if len(i)==n_timeplex and output_df.lib_rt[i[idx]]>lib_rt_range[0] and output_df.lib_rt[i[idx]]<lib_rt_range[1]]]
        new_filter = get_df_filter(t_df,50)
        filters.append(new_filter)
        rt_spl = lowess_fit(t1[:,1][new_filter],t1[:,0][new_filter])
        rt_spls.append(rt_spl)
        t_vals.append(t1)
        t_seqs.append(t1_s)
        
        converted_rt = rt_spl(t1[:,1])
        converted_rts.append(converted_rt)
        gaussian_fits.append(fit_gaussian(t1[:,0]-converted_rt))
        
    return rt_spls, t_vals, t_seqs, filters

"""
def timeplex_algnment_plots(n_timeplex, t_vals, results_folder = None):
    ##plot RT alignment
    filter_bool = np.logical_and.reduce([*all_diff_bools,rt_filter_bool])
    
    plt.subplots()
    for idx in range(n_timeplex):
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
    if results_folder is not None:
        plt.savefig(results_folder+"/OriginalRTfit.png",dpi=600,bbox_inches="tight")
        
    ### want this later
    plt.subplots()
    for idx in range(n_timeplex):
        plt.scatter(np.array([updatedLibrary[key]["iRT"] for key in keys])[filter_bool],np.array([i[0] for i in t_vals[idx]])[filter_bool],s=1,label=f"T{str(idx)}",alpha=.2)
        # plt.scatter([updatedLibrary[key]["iRT"] for key in keys],rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys]),s=1,label=f"T{str(idx)}",c=colours[idx])
        plt.scatter([updatedLibrary[key]["iRT"] for key in keys],rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys])+config.opt_rt_tol,s=.1,c=colours[idx],alpha=.1)
        plt.scatter([updatedLibrary[key]["iRT"] for key in keys],rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys])-config.opt_rt_tol,s=.1,c=colours[idx],alpha=.1)
        # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])+config.rt_tol_spl(t_vals[idx][:,1]),s=.1,c=colours[idx],alpha=.1)
        # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])-config.rt_tol_spl(t_vals[idx][:,1]),s=.1,c=colours[idx],alpha=.1)
    plt.legend(markerscale=10)
    plt.xlabel("Updated Library RT")
    plt.ylabel("Observed RT")
    if results_folder is not None:
        plt.savefig(results_folder+"/RTfit.png",dpi=600,bbox_inches="tight")
    
    plt.subplots()
    for idx in range(n_timeplex):
        vals,bins,_ =plt.hist(np.array(t_vals[idx][:,0]-rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys]))[filter_bool],100,alpha=.5,label=f"T{str(idx)}")
        # rt_stddev = gaussian_fits[idx][-1]
    x_scale = np.diff(plt.xlim())[0]
    plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals),color="r")
    plt.text(config.opt_rt_tol+x_scale/100,max(vals)*.8,np.round(config.opt_rt_tol,2))
    plt.legend()  
    plt.xlabel("RT difference")
    plt.ylabel("Frequency") 
    # plt.show()
    if results_folder is not None:
        plt.savefig(results_folder+"/RTdiff.png",dpi=600,bbox_inches="tight")
    
    
    plt.subplots()
    for idx in range(n_timeplex):
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
    if results_folder is not None:
        plt.savefig(results_folder+"/Rterrors.png",dpi=600,bbox_inches="tight")
    
    
    plt.subplots()
    vals,bins,_ = plt.hist((f([i[1] for i in t_vals[0]])-[i[0] for i in t_vals[0]])[filter_bool],np.linspace(-10,10,150),density=True,label="Original RT")
    plt.hist((rt_spls[0]([updatedLibrary[key]["iRT"] for key in keys])-[i[0] for i in t_vals[0]])[filter_bool],bins,alpha=.5,density=True,label="Updated RT")
    plt.plot(np.linspace(-5,5,100),gaussian(np.linspace(-5,5,100), rt_amplitude, rt_mean, rt_stddev),label="Updated RT fit")
    plt.legend()
    plt.xlabel("RT alignment errors")
    plt.savefig(results_folder+"/RtAlignmentErrors.png",dpi=600,bbox_inches="tight")
    # plt.show()
    
    fig, ax = plt.subplots(nrows = n_timeplex, figsize=(7.2, 3.6*n_timeplex))        
    for idx,row in enumerate(ax):
        row.scatter(np.array(t_vals[idx][:,1])[filter_bool],np.array(t_vals[idx][:,0]-rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys]))[filter_bool],label="Original_RT",s=.1)
        row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[0,0],color="r",linestyle="--",alpha=.5)
        row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[config.opt_rt_tol,config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
        row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[-config.opt_rt_tol,-config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
        row.set_ylabel(f"RT Residuals (T{idx})")
        row.set_ylim(-5,5)
    # plt.scatter(output_df.lib_rt,rt_spl(output_df.lib_rt),label="Predicted_RT",s=1)
    # plt.legend()
    plt.xlabel("Updated Library RT")
    # plt.ylabel("RT Residuals")
    # plt.show()
    if results_folder is not None:
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
    if results_folder is not None:
        plt.savefig(results_folder+"/RTelbows.png",dpi=600,bbox_inches="tight")
    
    
    
    ##plot mz alignment
    plt.subplots()
    plt.scatter(rts,diffs,label="Original_MZ",s=1,alpha=min(1,5/((len(output_df.rt)//1000)+1)))
    plt.scatter(rts,f_rt_mz(rts),label="Predicted_MZ",s=1)
    # plt.legend()
    plt.xlabel("Updated RT")
    plt.ylabel("m/z difference (relative)")
    # plt.show()
    plt.savefig(results_folder+"/MZrtfit.png",dpi=600,bbox_inches="tight")
    
    ##plot mz alignment
    plt.subplots()
    plt.scatter(id_mzs,diffs-f_rt_mz(rts),label="Original_MZ",s=1,alpha=min(1,5/((len(output_df.rt)//1000)+1)))
    plt.scatter(id_mzs,mz_spl(id_mzs),label="Predicted_MZ",s=1)
    # plt.legend()
    plt.xlabel("m/z")
    plt.ylabel("m/z difference (relative)")
    # plt.show()
    if results_folder is not None:
        plt.savefig(results_folder+"/MZfit.png",dpi=600,bbox_inches="tight")
    
    
    ## plot mz alignment
    plt.subplots()
    plt.hist(np.array(diffs)[rt_mz_filter_bool],100)
    # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_df.lib_rt))/id_mzs,100,alpha=.5)
    # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
    vals,bins,_ = plt.hist((diffs-mz_spl(id_mzs)-f_rt_mz(rts))[rt_mz_filter_bool],100,alpha=.5)
    plt.vlines([-config.opt_ms1_tol,config.opt_ms1_tol],0,max(vals)*.8,color="r")
    # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
    plt.text(config.opt_ms1_tol,max(vals)*.8,f"{np.round(1e6*config.opt_ms1_tol,2)} ppm")
    plt.xlabel("m/z difference (relative)")
    plt.ylabel("Frequency")
    # plt.show()
    if results_folder is not None:
        plt.savefig(results_folder+"/MZdiff.png",dpi=600,bbox_inches="tight")
"""

def MZRTfit_timeplex(dia_spectra,librarySpectra,dino_features,mz_tol,ms1=False,results_folder=None,ms2=False):
    """
    Perform a preliminary search of the timeplex spectra to align the library mz and RT values

    Parameters
    ----------
    dia_spectra : src.utisl.io.load_files.SpectrumFile
        Spectra to align the library to.
    librarySpectra : dict
        Spectrum library.
    dino_features : pd.DataFrame
        Dataframe of features identified using Biosaur2.
    mz_tol : float
        MS1 mz tolerance.
    ms1 : bool, optional
        DESCRIPTION. The default is False.
    results_folder : String, optional
        If provided, where to save the logs/figures. The default is None.
    ms2 : bool, optional
        (Not active) Whether to align at MS2 level. The default is False.

    Returns
    -------
    (rt_spl, mz_func), updatedLibrary
    
        rt_spl: Spline fitting library retention time to observed values
        
        mz_func: Function that aligns library precuror m/z to observed values
        
        updatedLibrary: Copy of the library with duplicates of each precursor for each timePlex and each with their own specific retention time
                        
        

    """
    ## for testing
    # mz_tol,ms1,results_folder,ms2 = (config.ms1_tol,False,None,False)
    # here spectra are both ms1 and ms2 
    
    config.n_most_intense_features = int(1e8) # larger than possible, essentually all
    
    scans_per_cycle = round(len(dia_spectra.ms2scans)/len(dia_spectra.ms1scans))
    logger.info("Initial search")
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
    
    fit_outputs, top_n_spectra, large_feature_indices, lf_mz = fit_with_features(dia_spectra, librarySpectra, dino_features)
    
    output_df, all_output_df, id_keys, feature_mzs =  process_prelim_search(fit_outputs,
                                                                              librarySpectra,
                                                                              top_n_spectra,
                                                                              dino_features,
                                                                              large_feature_indices,
                                                                              lf_mz
                                                                              )
    
   
    
    #### create dictionary for each key and it's positions
    
    multiples, num_multiples, multiples_idxs = get_multiples(id_keys, output_df)
    
    if config.num_timeplex==0:
        n_timeplex = stats.mode(num_multiples).mode
    else:
        n_timeplex = config.num_timeplex
        
    # while it may be nice to know, we are assuming that this is not constant and therfore not necessary to know
    time_diffs = np.concatenate([np.diff(i) for i in multiples if len(i)==n_timeplex])
    # plt.hist(time_diffs,np.linspace(-1,5,40))
    # plt.xlabel("TimePLEX offset")
    
    # plt.scatter(np.concatenate([i[0:2] for i in multiples if len(i)==timeplex]),time_diffs,s=1,edgecolors="none")
    # plt.ylabel("TimePLEX offset")
    # plt.xlabel("RT")
    
    
    rt_spls, t_vals, t_seqs, filters = split_timePlex(output_df,n_timeplex,rt_mz, id_keys, multiples_idxs)
    
    # for idx in range(n_timeplex):
    #     plt.scatter(t_vals[idx][:,1][filters[idx]],t_vals[idx][:,0][filters[idx]],s=1,c=colours[idx],edgecolor="none",label=f"T{str(idx)}")
    #     plt.plot(sorted(t_vals[idx][:,1][filters[idx]]),rt_spls[idx](sorted(t_vals[idx][:,1][filters[idx]])),color=colours[idx],label=f"T{str(idx)}")
    # plt.xlabel("Library RT")
    # plt.ylabel("Observed RT")
    # plt.ylim(0,60)
    
    
    ### this may be useful to implement (requires more testing)
    # for percentile in [20, 40, 60, 80]:
    #     rt_spls = []
    #     t_vals = []
    #     t_seqs = []
    #     t_dfs = []
    #     filters = []
    #     converted_rts = []
    #     gaussian_fits = []
    #     for idx in range(timeplex):
    #         lib_rt_range = [np.percentile(rt_mz[:,0],5),np.percentile(rt_mz[:,0],95)]
    #         ### array of (obs_rt, lib_rt, hyperscore)
    #         t1 = np.array([[output_df.rt[i[idx]],output_df.lib_rt[i[idx]],output_df.hyperscore[i[idx]],output_df.mz[i[idx]],output_df.coeff[i[idx]],output_df.frac_lib_int[i[idx]]] for i in multiples_idxs if len(i)==timeplex and output_df.lib_rt[i[idx]]>lib_rt_range[0] and output_df.lib_rt[i[idx]]<lib_rt_range[1]])
    #         t1_s = [id_keys[i[idx]] for i in multiples_idxs if len(i)==timeplex and output_df.lib_rt[i[idx]]>lib_rt_range[0] and output_df.lib_rt[i[idx]]<lib_rt_range[1]]
    #         # t1 = np.array([[output_df.rt[i[idx]],output_df.lib_rt[i[idx]],output_hyper[i[idx]]] for i in multiples_idxs if len(i)==timeplex])
    #         t_df = output_df.iloc[[i[idx] for i in multiples_idxs if len(i)==timeplex and output_df.lib_rt[i[idx]]>lib_rt_range[0] and output_df.lib_rt[i[idx]]<lib_rt_range[1]]]
    #         new_filter = get_df_filter(t_df,percentile)
    #         rt_spl = lowess_fit(t1[:,1][new_filter],t1[:,0][new_filter])
    #         filters.append(new_filter)
    #         rt_spls.append(rt_spl)
    #         t_vals.append(t1)
    #         t_seqs.append(t1_s)
    #         converted_rt = rt_spl(t1[:,1])
    #         converted_rts.append(converted_rt)
    #         gaussian_fits.append(fit_gaussian(t1[:,0]-converted_rt))
            
    #     plt.subplots()
    #     for idx in range(timeplex):
    #         # print(np.sum(filters[idx]))
    #         plt.scatter(t_vals[idx][:,1][filters[idx]],t_vals[idx][:,0][filters[idx]],s=1,c=colours[idx],edgecolor="none",label=f"T{str(idx)}")
    #         plt.plot(sorted(t_vals[idx][:,1][filters[idx]]),rt_spls[idx](sorted(t_vals[idx][:,1][filters[idx]])),color=colours[idx],label=f"T{str(idx)}")
    #     plt.xlabel("Library RT")
    #     plt.ylabel("Observed RT")
    #     plt.title(f"Percentile: {percentile}")
    #     plt.show()
    
    ########################################################################################################
    #########################################################################################################
    #########################################################################################################
    
    
    ## only uses peptides within certain tolerance (Assume most of these are true nad exclude incorrect outliers)
    # _bool = np.abs(rt_diffs)<(4*rt_stddev)
    # ### Could also change to those with the expectd offset when observed
    # rt_offsets = np.array([i[0] for i in t_vals[1]])-[i[0] for i in t_vals[0]]
    # rt_offsets2 = np.array([i[0] for i in t_vals[2]])-[i[0] for i in t_vals[1]] ## for column 2 vs 3
    all_rt_offsets = [np.array([i[0] for i in t_vals[idx+1]])-[i[0] for i in t_vals[idx]] 
                      for idx in range(n_timeplex-1)]
    offset_tolerance = 1 ## 1 minute
    # expected_offset = stats.mode(np.round(all_rt_offsets[0][all_rt_offsets[0]>.5],1)).mode
    exp_offsets = [stats.mode(np.round(rt_off[rt_off>.5],1)).mode for rt_off in all_rt_offsets] ## ensure it's around zero
    diff_bool = np.abs(all_rt_offsets[0]-exp_offsets[0])<offset_tolerance
    all_diff_bools = [np.abs(all_rt_offsets[idx]-exp_offsets[idx])<offset_tolerance for idx in range(n_timeplex-1)]
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
    
    
    t0_rts = np.array(t_vals[0][:,1])
    ## exclude regions where there are no IDs
    rt_filter_bool = filter_rts_by_dense(t0_rts,30)
    
    emp_rt_spls = []
    for idx in range(n_timeplex):
        # rt_spl = threestepfit([updatedLibrary[key]["iRT"] for key in keys],[i[0] for i in t_vals[0]],1)
        rt_spl = lowess_fit(np.array(t_vals[idx][:,1])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],
                            np.array(t_vals[idx][:,0])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],frac=.1)
        emp_rt_spls.append(rt_spl)
    
    all_emp_diffs = np.concatenate([emp_rt_spls[i](np.array(t_vals[idx][:,1]))[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])]-np.array(t_vals[i][:,0])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])] for i in range(n_timeplex)])
    
    # for idx in range(n_timeplex):
    #     # plt.subplots()
    #     test_bool = np.logical_and(diff_bool,rt_filter_bool)
    #     plt.scatter(t0_rts[test_bool],np.array([i[0] for i in t_vals[idx]])[test_bool],c=colours[idx],s=1,edgecolor="none") 
    #     plt.scatter(t0_rts[test_bool],rt_spls[idx](t0_rts)[test_bool],c=colours[idx],s=1,alpha=.2) 
    
    
    percentile = config.rt_percentile
    limit = 3 ## exlcude RT diffs larger than this (outliers)
    
    
    ###############################################################
    ####### fine tuning
    ###############################################################
    # """
    if not config.args.use_emp_rt:
            
            
        ## use observed rt for fine_tuning
        grouped_df = pd.DataFrame({'Stripped.Sequence':[librarySpectra[(i[0],float(i[1]))]["seq"] for i in t_seqs[0]],"RT":[i[0] for i in t_vals[0]]})[diff_bool]
        data_split, models, convertor = fine_tune_rt(grouped_df,qc_plots=True,results_path=results_folder)

        
        ### recalculate RT_spls...
        keys = [(i,float(j)) for i,j in t_seqs[0]]
        
        lib_seqs = [one_hot_encode_sequence(librarySpectra[key]["seq"]) for key in keys]
        new_lib_rts = convertor(np.mean([model.predict(np.array(lib_seqs)) for model in models],axis=0).flatten())
        
        t0_rts = new_lib_rts
        ## exclude regions where there are no IDs
        rt_filter_bool = filter_rts_by_dense(t0_rts,30)
        rt_spls = []
        for idx in range(n_timeplex):
            # rt_spl = threestepfit([updatedLibrary[key]["iRT"] for key in keys],[i[0] for i in t_vals[0]],1)
            rt_spl = lowess_fit(t0_rts[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],
                                np.array([i[0] for i in t_vals[idx]])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])],frac=.2)
            rt_spls.append(rt_spl)
        
        # for idx in range(n_timeplex):
        #     # plt.subplots()
        #     test_bool = np.logical_and(diff_bool,rt_filter_bool)
        #     plt.scatter(t0_rts[test_bool],np.array([i[0] for i in t_vals[idx]])[test_bool],c=colours[idx],s=1,edgecolor="none") 
        #     plt.scatter(t0_rts[test_bool],rt_spls[idx](t0_rts)[test_bool],c=colours[idx],s=1,alpha=.2) 
        
        all_pred_diffs = np.concatenate([rt_spls[i](t0_rts)[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])]-np.array([i[0] for i in t_vals[i]])[np.logical_and.reduce([*all_diff_bools,rt_filter_bool])] for i in range(n_timeplex)])
        
        
        
        emp_data, emp_p, emp_cdf_auc = cdf_data(all_emp_diffs,limit=limit)
        pred_data, pred_p, pred_cdf_auc = cdf_data(all_pred_diffs,limit=limit)
        
    
        updatedLibrary = copy.deepcopy(librarySpectra)
        all_lib_keys = list(librarySpectra)
        
        
        
        ### compare original empirical RTs to fintuned RTs
        
        if pred_cdf_auc>emp_cdf_auc: ## Predictions are better
            logger.info("Fine Tuned Library Chosen")
    
            boundary = fit_errors(all_pred_diffs,limit,percentile)
    
            all_lib_seqs = [one_hot_encode_sequence(updatedLibrary[key]["seq"]) for key in all_lib_keys]
            all_new_lib_rts = convertor(np.mean([model.predict(np.array(all_lib_seqs)) for model in models],axis=0).flatten())
            
            for key,rt in zip(all_lib_keys,all_new_lib_rts):
                updatedLibrary[key]["iRT"] = rt
                
        else: ### empirical are better
            ## keep the library RTs the same
    
            logger.info("Empirical Library Chosen")
            
            boundary = fit_errors(all_emp_diffs,limit,percentile)
    
            ## update the splines
            rt_spls = emp_rt_spls
            
    
        
    ###############################################################
    ####### NO fine tuning
    ###############################################################
     
    else:
    # """
        logger.info("Using Empirical w/o Fine Tuning")
        updatedLibrary = copy.deepcopy(librarySpectra)
        all_lib_keys = list(librarySpectra)
        
        keys = [(i,float(j)) for i,j in t_seqs[0]]
        boundary = fit_errors(all_emp_diffs,limit,percentile)
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
    
   
    ## NB: Only for n_timeplex=2
    ## computes differences between the fit lines of first 2 plexes
    prediction_diffs = np.abs(rt_spls[1]([updatedLibrary[key]["iRT"] for key in keys])-rt_spls[0]([updatedLibrary[key]["iRT"] for key in keys]))
    all_prediction_diffs = []
    for idx in range(n_timeplex-1):
        all_prediction_diffs.append(np.abs(rt_spls[idx+1]([updatedLibrary[key]["iRT"] for key in keys])-rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys])))
    #####  Assume that the mz error is independent of n_timeplex
    resp_ms1scans = [closest_ms1spec(output_df.rt[i], ms1_rt) for i in range(len(output_df.rt))]
    diffs = [closest_peak_diff(mz, ms1spectra[i].mz) for i,mz in zip(resp_ms1scans,output_df.mz)]
    
    mz_spl = twostepfit(output_df.mz,diffs,1)
    
    
    
    
    
    ################################################
    ########### correct mz errors wrt RT    ########
    ################################################
    
    rts = np.array([updatedLibrary[(i[0],float(i[1]))]["iRT"] for i in id_keys])#np.array([i[0] for i in t_vals[0]])
    # rt_filter_bool = filter_rts_by_dense(rts,30)
    # rt_filter_bool = np.logical_and(rts>15,rts<30)
    rt_mz_filter_bool = np.array(output_df.frac_lib_int)>.9 # use as proxy for correct IDs
    f_rt_mz = lowess_fit(rts[rt_mz_filter_bool],np.array(diffs)[rt_mz_filter_bool],.2)
    # plt.scatter(rts[rt_mz_filter_bool],np.array(diffs)[rt_mz_filter_bool],label="Original_MZ",s=1,alpha=.1)
    # plt.scatter(output_df.rt,f_rt_mz(output_df.rt),s=1,alpha=.2)
    
    # plt.scatter(id_mzs,diffs,label="Original_MZ",s=1,alpha=.1)
    # plt.scatter(id_mzs,diffs-f_rt_mz(output_df.rt),label="Original_MZ",s=1,alpha=.1)
    
    # mz_spl = twostepfit(np.array(id_mzs)[rt_mz_filter_bool],(diffs-f_rt_mz(output_df.rt))[rt_mz_filter_bool],1)
    mz_spl = lowess_fit(np.array(output_df.mz)[rt_mz_filter_bool],(diffs-f_rt_mz(output_df.rt))[rt_mz_filter_bool])
    # plt.scatter(id_mzs,diffs-f_rt_mz(output_df.rt),label="Original_MZ",s=1,alpha=.1)
    # plt.scatter(id_mzs,mz_spl(id_mzs),label="Original_MZ",s=1,alpha=.1)
    # plt.hlines(0,400,900)

    def mz_func(mz,rt):
        return mz+((mz_spl(mz)+f_rt_mz(rt))*mz)
    
    # orig_mzs = id_mzs+(diffs*np.array(id_mzs))
    # plt.hist(((mz_func(id_mzs,rts)-orig_mzs)/id_mzs)[rt_filter_bool],100)
    
    corrected_mz_diffs = (diffs-(f_rt_mz(rts)+mz_spl(output_df.mz)))[rt_mz_filter_bool]
    mz_amplitude, mz_mean, mz_stddev = fit_gaussian(corrected_mz_diffs)
    
    # plt.hist(np.array(diffs)[rt_mz_filter_bool],100,density=True)
    # vals,bins,_ = plt.hist(corrected_mz_diffs,100,alpha=.5,density=True)
    # plt.plot(bins,gaussian(bins, mz_amplitude, mz_mean, mz_stddev))
    # plt.vlines(0,0,max(vals))
    
    ### MS2 alignment
    # if ms2:
    #     all_frag_errors = np.concatenate(lc_frags_errors)
    #     all_frags = np.concatenate(lc_frags)
    #     ms2_spl = twostepfit(all_frags,all_frag_errors,1)
    #     def ms2_func(mz):
    #         return mz+(ms2_spl(mz)*mz)
        
    #     ms2_amplitude, ms2_mean, ms2_stddev = fit_gaussian(all_frag_errors-ms2_spl(all_frags))
    
   
    
    new_rt_tol =np.abs(boundary)# 4*np.abs(rt_stddev)
    if config.args.user_rt_tol:
        logger.info("Using user specified RT tolerance")
        new_rt_tol = np.abs(config.args.rt_tol)
    logger.info(f"Optimized RT tolerance: {new_rt_tol}")
    
    # ## ensure there is no overlap
    # obs_rt_range = [min(output_df.rt),max(output_df.rt)]
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
        logger.warning("Warning; Library RTs overlapping")
        new_rt_tol = np.abs(min_prediction_diff/2)*.99 # ensure no overlap
        logger.warning(f"Reseting tolerance to {new_rt_tol}")
    

    
    config.opt_rt_tol = new_rt_tol
    
    
    
    # set optimised ms2 tol
    # is_real = ~np.isnan(diffs)
    # buffer = 1.2
    # config.opt_ms1_tol = np.round(
    #                         np.sort(
    #                             np.abs(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_df.lib_rt))/id_mzs
    #                                    )[is_real])[int(sum(is_real)*.95)]*buffer,6+5)#6 for 1e-6 the 5 decimal places


    new_ms1_tol = np.abs(4*mz_stddev)
    logger.info(f"Optimized MS1 tolerance: {new_ms1_tol}")
    logger.info("")
    
    config.opt_ms1_tol  = new_ms1_tol
    
    # if ms2:
    #     new_ms2_tol = 4*ms2_stddev
    #     config.opt_ms2_tol  = new_ms2_tol
    
    if results_folder is not None:
        
        ### Save functions
        for idx in range(n_timeplex):
            with open(results_folder+f"/rt_spl{idx}","wb") as dill_file:
                dill.dump(rt_spls[idx],dill_file)
            
        with open(results_folder+"/mz_func","wb") as dill_file:
            dill.dump(mz_func,dill_file)
        
        # if ms2:
        #     with open(results_folder+"/ms2_func","wb") as dill_file:
        #         dill.dump(ms2_func,dill_file)
            
        ##plot RT alignment
        filter_bool = np.logical_and.reduce([*all_diff_bools,rt_filter_bool])
        
        plt.subplots()
        for idx in range(n_timeplex):
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
        for idx in range(n_timeplex):
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
        for idx in range(n_timeplex):
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
        for idx in range(n_timeplex):
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
        
        fig, ax = plt.subplots(nrows = n_timeplex, figsize=(7.2, 3.6*n_timeplex))        
        for idx,row in enumerate(ax):
            row.scatter(np.array(t_vals[idx][:,1])[filter_bool],np.array(t_vals[idx][:,0]-rt_spls[idx]([updatedLibrary[key]["iRT"] for key in keys]))[filter_bool],label="Original_RT",s=.1)
            row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[0,0],color="r",linestyle="--",alpha=.5)
            row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[config.opt_rt_tol,config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
            row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[-config.opt_rt_tol,-config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
            row.set_ylabel(f"RT Residuals (T{idx})")
            row.set_ylim(-5,5)
        # plt.scatter(output_df.lib_rt,rt_spl(output_df.lib_rt),label="Predicted_RT",s=1)
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
        plt.scatter(rts,diffs,label="Original_MZ",s=1,alpha=min(1,5/((len(output_df.rt)//1000)+1)))
        plt.scatter(rts,f_rt_mz(rts),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("Updated RT")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        plt.savefig(results_folder+"/MZrtfit.png",dpi=600,bbox_inches="tight")
        
        ##plot mz alignment
        plt.subplots()
        plt.scatter(output_df.mz,diffs-f_rt_mz(rts),label="Original_MZ",s=1,alpha=min(1,5/((len(output_df.rt)//1000)+1)))
        plt.scatter(output_df.mz,mz_spl(output_df.mz),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("m/z")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        plt.savefig(results_folder+"/MZfit.png",dpi=600,bbox_inches="tight")
        
        
        ## plot mz alignment
        plt.subplots()
        plt.hist(np.array(diffs)[rt_mz_filter_bool],100)
        # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_df.lib_rt))/id_mzs,100,alpha=.5)
        # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
        vals,bins,_ = plt.hist((diffs-mz_spl(output_df.mz)-f_rt_mz(rts))[rt_mz_filter_bool],100,alpha=.5)
        plt.vlines([-config.opt_ms1_tol,config.opt_ms1_tol],0,max(vals)*.8,color="r")
        # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
        plt.text(config.opt_ms1_tol,max(vals)*.8,f"{np.round(1e6*config.opt_ms1_tol,2)} ppm")
        plt.xlabel("m/z difference (relative)")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(results_folder+"/MZdiff.png",dpi=600,bbox_inches="tight")

        plt.close("all")
    
    
    # if ms2:
    #     return (rt_spls, mz_func, ms2_func), updatedLibrary
    # else:
    return (rt_spls, mz_func), updatedLibrary



