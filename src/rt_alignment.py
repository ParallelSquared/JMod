"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""
import sys
import os
from itertools import combinations

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
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from scipy import stats
from sklearn.metrics import auc
from threadpoolctl import threadpool_limits
import dill
dill.settings['recurse'] = True
import copy

from scipy.interpolate import interp1d
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

#from src.mass_tags import tag_library, mTRAQ, mTRAQ_678, mTRAQ_02468, diethyl_6plex, tag6

from src.utils.misc_functions import within_tol,moving_average, \
    closest_ms1spec, closest_peak_diff, unstring_floats, fragment_cor


from src.finetune_funs import fine_tune_rt, one_hot_encode_sequence, \
    bagged_finetune_channel, rt_model_path
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


def fast_modal_lowess(x, y,
                      local_frac=0.2,
                      grid_size=100,
                      anchors=200,
                      post_smooth_frac=0.1):

    x = np.asarray(x)
    y = np.asarray(y)

    # Sort
    # order = np.argsort(x)
    order = np.lexsort((y, x))
    x = x[order]
    y = y[order]
    n = len(x)

    # window size
    w = max(5, int(local_frac * n))

    # choose anchor positions
    anchor_idx = np.linspace(0, n - 1, anchors).astype(int)

    modal_vals = np.zeros(len(anchor_idx))

    # Silverman bandwidth (fixed)
    bw = 1.06 * np.std(y) * n ** (-1 / 5)

    from scipy.stats import norm

    for k, i in enumerate(anchor_idx):

        start = max(0, i - w // 2)
        end = min(n, i + w // 2)
        y_win = y[start:end]

        y_min, y_max = y_win.min(), y_win.max()
        y_range = y_max - y_min

        # small grid
        grid = np.linspace(y_min - 0.05 * y_range,
                           y_max + 0.05 * y_range,
                           grid_size)

        # fast KDE
        diff = (grid[None, :] - y_win[:, None]) / bw
        density = norm.pdf(diff).sum(axis=0)

        modal_vals[k] = grid[np.argmax(density)]

    # interpolate modal values to full x-grid
    modal_full = np.interp(x, x[anchor_idx], modal_vals)

    # Post smooth
    smooth = lowess(modal_vals, x[anchor_idx], frac=post_smooth_frac, it=0)[:, 1]

    # Build interpolator
    itp = interp1d(
        x[anchor_idx], smooth,
        bounds_error=False,
        fill_value=(modal_full.min(), modal_full.max())
    )
    # Expose raw anchors for diagnostic plotting (pre-smoothing modal estimates)
    itp.anchor_x = x[anchor_idx]
    itp.anchor_y = modal_vals
    return itp

    
    
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

    # Use single-threaded execution for reproducibility
    with threadpool_limits(limits=1):
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



def empirical_fit(output_df, results_folder=None):
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

    for feature_percentile in range(20, 100, 5):

        cor_filter = np.logical_and.reduce(
            [output_df[feat] > np.percentile(output_df[feat], feature_percentile)
             for feat in [
                "scribe_score",
             ]
             ]
        )
        
        ## Only fit when fewer than this many peaks
        if sum(cor_filter)>config.max_num_prelim_search:
            continue
        
        f = fast_modal_lowess(output_df.lib_rt[cor_filter],
                        output_df.rt[cor_filter],
                        .01,
                        anchors=1000,
                        grid_size=1000,
                        post_smooth_frac=0.01)

        plt.subplots()
        plt.scatter(output_df.lib_rt[cor_filter],
                    output_df.rt[cor_filter], s=1,alpha=.2)
        plt.scatter(output_df.lib_rt[cor_filter],
                    f(output_df.lib_rt[cor_filter]),edgecolor="none", s=1)
        plt.scatter(f.anchor_x, f.anchor_y, color="red", s=8, edgecolor="none")
        plt.title(str(feature_percentile))
        if results_folder is not None:
            plt.savefig(results_folder + f"/first_search/Percentile_{feature_percentile}.png",
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

        k = 3.29 * sigmas[0] # middle 99.9%
        p_in = 2.0 * norm.cdf(k / sigmas) - 1.0
        num = weights * p_in
        partial_posterior = num[0] / num.sum()

        # rt error + mixture plot, saved per percentile
        # if results_folder is not None:
        plot_rt_residuals_mixture(
            residuals=res,
            feat=feature_percentile,
            weights=weights,
            sigmas=sigmas,
            results_folder=results_folder
        )
        

        logger.info(
            f"Testing Percentile: {feature_percentile}, "
            f"Ratio: {outside_ratio:.4f}, #IDs: {cor_filter.sum()}, Partial Posterior: {partial_posterior:.4f}"
        )

        # new + existing stopping criteria
        if (partial_posterior >= 0.99 or
                outside_ratio < 0.05 or
                (cor_filter.sum() - bad_IDs.sum()) < 800):
            break

    logger.debug(f"{feature_percentile} {np.round(outside_ratio, 4)} {cor_filter.sum()}")

    cor_filter = np.logical_and(
        cor_filter,
        np.abs(first_rt_diffs) < first_rt_tolerance
    )

    emp_rt_spl = fast_modal_lowess(
        np.array(output_df.lib_rt)[cor_filter],
        np.array(output_df.rt)[cor_filter],
        .01,
        anchors=1000,
        grid_size=1000,
        post_smooth_frac=0.01
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
        f"PEP(0) = {pep0:.3f}", #TODO THIS IS NOT THE RIGHT VALUE
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )

    plt.legend()
    plt.xlabel("fit_rt - rt")
    plt.ylabel("density")
    plt.title(f"RT Residuals After Alignment {feat}")
    plt.tight_layout()

    if results_folder is not None:
        if feat == "_empirical_final":
            plt.savefig(
            results_folder + f"/first_search/rt_residuals{feat}.png",
            dpi=600,
            bbox_inches="tight",
            )
            plt.close()
        else:
            plt.savefig(
                results_folder + f"/first_search/rt_residuals_p{feat}.png",
                dpi=600,
                bbox_inches="tight",
            )
            plt.close()

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
                    results_folder=None,
                    channels=None,
                    rt_spls=None,
                    offsets=None):
        if channels is not None:
            lib = filtered_output["lib_rt"].to_numpy()
            ulr = filtered_output["updated_lib_rt"].to_numpy()
            obs = filtered_output["rt"].to_numpy()
            xo = np.linspace(lib.min(), lib.max(), 500)
            xu = np.linspace(ulr.min(), ulr.max(), 500)
            tol = config.opt_rt_tol

            plt.subplots()
            for k in range(len(rt_spls)):
                m = channels == k
                plt.scatter(lib[m], obs[m], s=.5, c=colours[k], alpha=.3, label=f"T{k}")
                plt.plot(xo, orig_spl(xo) + offsets[k], c=colours[k])
            plt.xlabel("Library RT"); plt.ylabel("Observed RT"); plt.legend(markerscale=10)
            if results_folder is not None:
                plt.savefig(results_folder+"/first_search/OriginalRTfit.png",dpi=600,bbox_inches="tight"); plt.close()

            plt.subplots()
            for k in range(len(rt_spls)):
                m = channels == k
                plt.scatter(ulr[m], obs[m], s=.5, c=colours[k], alpha=.3, label=f"T{k}")
                plt.plot(xu, rt_spls[k](xu), c=colours[k])
            plt.xlabel("Updated Library RT"); plt.ylabel("Observed RT"); plt.legend(markerscale=10)
            if results_folder is not None:
                plt.savefig(results_folder+"/first_search/RTfit.png",dpi=600,bbox_inches="tight"); plt.close()

            plt.subplots()
            for k in range(len(rt_spls)):
                m = channels == k
                plt.scatter(ulr[m], obs[m] - rt_spls[k](ulr[m]), s=.5, c=colours[k], alpha=.3, label=f"T{k}")
            for yv, col in ((0,"r"), (tol,"g"), (-tol,"g")):
                plt.plot([ulr.min(), ulr.max()], [yv, yv], color=col, linestyle="--", alpha=.5)
            plt.ylim(-5, 5); plt.xlabel("Updated Library RT"); plt.ylabel("RT Residuals"); plt.legend(markerscale=10)
            if results_folder is not None:
                plt.savefig(results_folder+"/first_search/RtResidual.png",dpi=600,bbox_inches="tight"); plt.close()

            plt.subplots()
            for k in range(len(rt_spls)):
                m = channels == k
                plt.hist(obs[m] - rt_spls[k](ulr[m]), 100, density=True, alpha=.5, label=f"T{k}")
            plt.vlines([-tol, tol], 0, plt.ylim()[1], color="r")
            plt.xlabel("RT difference"); plt.ylabel("Frequency"); plt.legend()
            if results_folder is not None:
                plt.savefig(results_folder+"/first_search/RTdiff.png",dpi=600,bbox_inches="tight"); plt.close()
        else:
            ##plot RT alignment
            plt.subplots()
            plt.scatter(filtered_output.lib_rt,np.array(filtered_output.rt),label="Original_RT",s=.1)
            plt.scatter(filtered_output.lib_rt,orig_spl(filtered_output.lib_rt),label="Predicted_RT",s=1)
            # plt.legend()
            plt.xlabel("Library RT")
            plt.ylabel("Observed RT")
            # plt.show()
            if results_folder is not None:
                plt.savefig(results_folder+"/first_search/OriginalRTfit.png",dpi=600,bbox_inches="tight")
                plt.close()
        
        
            ##plot RT alignment
            plt.subplots()
            plt.scatter(filtered_output.updated_lib_rt,np.array(filtered_output.rt),label="Original_RT",s=.1)
            plt.scatter(filtered_output.updated_lib_rt,rt_spl(filtered_output.updated_lib_rt),label="Predicted_RT",s=1)
            # plt.legend()
            plt.xlabel("Updated Library RT")
            plt.ylabel("Observed RT")
            # plt.show()
            if results_folder is not None:
                plt.savefig(results_folder+"/first_search/RTfit.png",dpi=600,bbox_inches="tight")
                plt.close()
        
        
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
                plt.savefig(results_folder+"/first_search/RtResidual.png",dpi=600,bbox_inches="tight")
                plt.close()
        
        
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
                plt.savefig(results_folder+"/first_search/RTdiff.png",dpi=600,bbox_inches="tight")
                plt.close()
        

        ##plot mz rt alignment
        plt.subplots()
        plt.scatter(filtered_output.updated_lib_rt,np.array(filtered_output.mz_diffs),label="Original_MZ",s=1,alpha=min(1,5/((len(np.array(filtered_output.rt))//1000)+1)))
        plt.scatter(filtered_output.updated_lib_rt,f_rt_mz(filtered_output.updated_lib_rt),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("Updated RT")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/first_search/MZrtfit.png",dpi=600,bbox_inches="tight")
            plt.close()
        

        ##plot mz alignment
        plt.subplots()
        plt.scatter(np.array(filtered_output.mz),(filtered_output.mz_diffs-f_rt_mz(filtered_output.updated_lib_rt)),label="Original_MZ",s=1,alpha=min(1,5/((len(np.array(filtered_output.updated_lib_rt))//1000)+1)))
        plt.scatter(filtered_output.mz,mz_spl(filtered_output.mz),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("m/z")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/first_search/MZfit.png",dpi=600,bbox_inches="tight")
            plt.close()
        
        
        
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
            plt.savefig(results_folder+"/first_search/MZdiff.png",dpi=600,bbox_inches="tight")
            plt.close()
    
    
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
            plt.savefig(results_folder+"/first_search/RTelbows.png",dpi=600,bbox_inches="tight")
        
        plt.close()
        
        plt.close("all")

def MZRTfit(dia_spectra,librarySpectra,dino_features,mz_tol,ms1=False,results_folder=None,ms2=False, mass_tag=None, SILAC=None, return_rt_models=False):
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
    

    # Run a preliminary search returning results at the PSM level
    import src.preliminary_search as preliminary_search
    output_df = preliminary_search.fit_with_features(dia_spectra, librarySpectra, mass_tag, SILAC, ms1_ppm_error=20, ms2_ppm_error=10)

    # Calculate the elution width and add cluster_size column
    import src.elution_analysis as elution_analysis
    fwhm, elution_sd, output_df = elution_analysis.calculate_elution_width(output_df)
    logger.info("Mean elution width: FWHM {fwhm:.4f}, SD {elution_sd:.4f}".format(fwhm=fwhm, elution_sd=elution_sd))

    # Vote sigma for MS1 quant Gaussian apex voting: elution SD expressed in MS1
    # cycles. Floored at 0.5 so the Gaussian doesn't collapse to a delta when MS1
    # cycles are coarse relative to the peak.
    ms1_rts = np.array([s.RT for s in dia_spectra.ms1scans])
    if len(ms1_rts) >= 2:
        ms1_cycle_time = float(np.median(np.diff(np.sort(ms1_rts))))
        vote_sigma = max(0.5, elution_sd / ms1_cycle_time) if ms1_cycle_time > 0 else 1.0
        logger.info(f"MS1 cycle time: {ms1_cycle_time:.4f} min, vote sigma: {vote_sigma:.2f} cycles")
    else:
        vote_sigma = 1.0
        logger.info("vote sigma: 1.0 cycles (fallback — fewer than 2 MS1 scans)")

    # Collapse to most intense MS1 per peptide ion
    output_df = output_df.sort("closest_peak_intensity_ms1", descending=True).unique(subset=["seq", "z"], keep="first")
    import polars as pl
    output_df = output_df.filter(pl.col("cluster_size") >= 1)

    # Convert to pandas for downstream processing
    output_df = output_df.to_pandas()

    id_keys = list(zip(output_df["seq"], output_df["z"]))

    #
    """
    if dino_features is None:
        fit_outputs = fit_without_features(dia_spectra, librarySpectra)

    else:
        fit_outputs, top_n_spectra, large_feature_indices, lf_mz = fit_with_features(dia_spectra, librarySpectra, dino_features)
        #fit_outputs, top_n_spectra, large_feature_indices, lf_mz = fit_with_features(dia_spectra, librarySpectra)
    """
    
    #################################################################################
    
    ########################################################################

    """
    output_df, all_output_df, id_keys, feature_mzs =  process_prelim_search(fit_outputs,
                                                                              librarySpectra,
                                                                              top_n_spectra,
                                                                              dino_features,
                                                                              large_feature_indices,
                                                                              lf_mz
                                                                              )
    

    """


    if results_folder is not None:
        output_df.to_csv(results_folder+"/first_search/firstSearch.tsv", index=False,sep='\t')
    # output_df = pd.DataFrame([j for i in output for j in i  if j[0]>min_int],columns=names[:len(output[0][0])])


    cor_filter, emp_rt_spl = empirical_fit(output_df, results_folder=results_folder)
        
    
    
    
    
    percentile = config.rt_percentile
    #percentile = 0.99
    
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
        
        pred_rt_spl = fast_modal_lowess(predicted_rts[cor_filter],
                               np.array(output_df.rt)[cor_filter],
                               local_frac=.01,
                               anchors=1000,
                               grid_size=1000,
                               post_smooth_frac=0.01)
        
        all_pred_diffs = (pred_rt_spl(predicted_rts) - np.array(output_df.rt))[cor_filter]
        
        all_pred_diffs = validation_rt_diffs
        
        emp_data, emp_p, emp_cdf_auc = cdf_data(all_emp_diffs,limit=limit)
        pred_data, pred_p, pred_cdf_auc = cdf_data(all_pred_diffs,limit=limit)

        # TODO: deepcopy is expensive — we only need to compare predicted vs
        # empirical iRTs, then write the winner's values. No need to duplicate
        # the entire library; just compute both iRT arrays and pick one.
        updatedLibrary = copy.deepcopy(librarySpectra)
        all_lib_keys = list(librarySpectra)


        ###### Check if fine-tuning improves alignment
        use_predictions = pred_cdf_auc > emp_cdf_auc or return_rt_models

        if use_predictions:
            if pred_cdf_auc > emp_cdf_auc:
                logger.info("Fine Tuned Library Chosen")
            else:
                logger.info(f"Fine Tuned Library Chosen (forced for decoy prediction; empirical AUC={emp_cdf_auc:.4f} vs pred AUC={pred_cdf_auc:.4f})")
            # Fit 2-component zero-mean GMM to residuals
            weights, sigmas = fit_zero_mean_gmm_1d(all_pred_diffs, n_components=2)
            order = np.argsort(sigmas)
            sigmas = sigmas[order]

            plot_rt_residuals_mixture(
                residuals=all_pred_diffs,
                feat="_fine_tuned_final",
                weights=weights,
                sigmas=sigmas,
                results_folder=results_folder
            )

            # Combine elution width and GMM sigma, take 4th standard deviation
            boundary = 4 * sigmas[0] + 8 * elution_sd
            rt_spl = pred_rt_spl
            # Deduplicate sequences to avoid redundant predictions
            unique_seqs = list(set(updatedLibrary[key]["seq"] for key in all_lib_keys))
            unique_encoded = np.array([one_hot_encode_sequence(s) for s in unique_seqs], dtype=np.float32)
            unique_rts = convertor(np.mean([model.predict(unique_encoded, batch_size=4096) for model in models], axis=0).flatten())
            seq_to_rt = dict(zip(unique_seqs, unique_rts))

            for key in all_lib_keys:
                updatedLibrary[key]["iRT"] = seq_to_rt[updatedLibrary[key]["seq"]]

        else:
            logger.info("Empirical Library Chosen")
            # Fit 2-component zero-mean GMM to residuals
            weights, sigmas = fit_zero_mean_gmm_1d(all_emp_diffs, n_components=2)
            order = np.argsort(sigmas)
            sigmas = sigmas[order]

            plot_rt_residuals_mixture(
                residuals=all_emp_diffs,
                feat="_empirical_final",
                weights=weights,
                sigmas=sigmas,
                results_folder=results_folder
            )

            # Combine elution width and GMM sigma, take 4th standard deviation
            boundary = 4 * sigmas[0] + 8 * elution_sd
            rt_spl = emp_rt_spl
        
        
        
    ###############################################################
    ####### NO fine tuning
    ###############################################################

    else:

        logger.info("Using Empirical w/o Fine Tuning")
        # TODO: no iRT mutation in this branch — deepcopy is unnecessary here
        updatedLibrary = copy.deepcopy(librarySpectra)
        all_lib_keys = list(librarySpectra)
        rt_spl = emp_rt_spl

        all_emp_diffs = (emp_rt_spl(output_df.lib_rt)-np.array(output_df.rt))[cor_filter]

        pred_data = pred_p = None
        emp_data = np.sort(np.abs(all_emp_diffs)[np.abs(all_emp_diffs) < limit])
        emp_data = np.append(emp_data,limit)
        emp_p = np.arange(len(emp_data)) / (len(emp_data) - 1)
        emp_cdf_auc = auc(emp_data,emp_p)
        # Fit 2-component zero-mean GMM to residuals
        weights, sigmas = fit_zero_mean_gmm_1d(all_emp_diffs, n_components=2)
        plot_rt_residuals_mixture(
            residuals=all_emp_diffs,
            feat="- Empirical Final",
            weights=weights,
            sigmas=sigmas,
            results_folder=results_folder
        )
        order = np.argsort(sigmas)
        sigmas = sigmas[order]
        # Combine elution width and GMM sigma, take 4th standard deviation
        boundary = 4 * sigmas[0] + 8 * elution_sd
        #boundary = fit_errors(all_emp_diffs, limit, percentile)
    
    new_lib_rt = np.array([updatedLibrary[k]["iRT"] for k in id_keys])
    converted_rt = rt_spl([updatedLibrary[k]["iRT"] for k in id_keys])
    
    rt_amplitude, rt_mean, rt_stddev = fit_gaussian((output_df.rt-converted_rt)[cor_filter])
   
    
   
    ################################################
    ########### correct mz errors wrt RT    ########
    ################################################
    
    
    """
    if dino_features is None:
        resp_ms1scans = [closest_ms1spec(output_df.rt[i], ms1_rt) for i in range(len(output_df.rt))]
        diffs = [closest_peak_diff(mz, ms1spectra[i].mz) for i,mz in zip(resp_ms1scans,output_df.mz)]
    else:
        diffs = np.array([(i-mz)/mz for i,mz in zip(feature_mzs,output_df.mz)])
    """
    diffs = output_df["relative_error_ms1"].to_numpy()


    
    f_rt_mz = fast_modal_lowess(new_lib_rt[cor_filter],np.array(diffs)[cor_filter],
                                local_frac=.01,
                                anchors=1000,
                                grid_size=1000,
                                post_smooth_frac=0.01)
    
    # mz_spl = twostepfit(np.array(id_mzs)[rt_filter_bool],(diffs-f_rt_mz(dia_rt))[r t_filter_bool],1)
    mz_spl = fast_modal_lowess(np.array(output_df.mz)[cor_filter],(diffs-f_rt_mz(new_lib_rt))[cor_filter],)


    def mz_func(mz,rt):
        return mz+((mz_spl(mz)+f_rt_mz(rt))*mz)
    
    # orig_mzs = id_mzs+(diffs*np.array(id_mzs))
    # plt.hist(((mz_func(id_mzs,rts)-orig_mzs)/id_mzs)[rt_filter_bool],100)
    
    corrected_mz_diffs = (diffs-(f_rt_mz(new_lib_rt)+mz_spl(output_df.mz)))[cor_filter]
    mz_weights, mz_sigmas = fit_zero_mean_gmm_1d(corrected_mz_diffs, n_components=2)
    mz_order = np.argsort(mz_sigmas)
    mz_sigmas = mz_sigmas[mz_order]
    mz_boundary = 4 * mz_sigmas[0]
    
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


    new_ms1_tol = np.abs(mz_boundary)
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

    ################################################################
    ########### Save the functions and Plot the alignment   ########
    ################################################################
    
    if results_folder is not None:
        
        ### Save functions
        with open(results_folder+"/first_search/rt_spl","wb") as dill_file:
            dill.dump(rt_spl,dill_file)
            
        with open(results_folder+"/first_search/mz_func","wb") as dill_file:
            dill.dump(mz_func,dill_file)
        
        # if ms2:
        #     with open(results_folder+"/ms2_func","wb") as dill_file:
        #         dill.dump(ms2_func,dill_file)

        # TODO double check input for tokens that aren't in the original library
        output_df["updated_lib_rt"] = [updatedLibrary[k]["iRT"] for k in id_keys]
        output_df["mz_diffs"] = diffs
        ms1_rts = np.array([s.RT for s in dia_spectra.ms1scans])
        ms1_tics = np.array([s.TIC for s in dia_spectra.ms1scans])
        output_df["ms1_tic"] = ms1_tics[np.searchsorted(ms1_rts, output_df.rt).clip(0, len(ms1_rts)-1)]
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
    if return_rt_models and not config.args.use_emp_rt:
        return (rt_spl, mz_func), updatedLibrary, (models, convertor), fwhm, vote_sigma
    return (rt_spl, mz_func), updatedLibrary, None, fwhm, vote_sigma

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
        ### array of (obs_rt, lib_rt)
        t1 = np.array([[output_df.rt[i[idx]],output_df.lib_rt[i[idx]]] for i in multiples_idxs if len(i)==n_timeplex and output_df.lib_rt[i[idx]]>lib_rt_range[0] and output_df.lib_rt[i[idx]]<lib_rt_range[1]])
        t1_s = [id_keys[i[idx]] for i in multiples_idxs if len(i)==n_timeplex and output_df.lib_rt[i[idx]]>lib_rt_range[0] and output_df.lib_rt[i[idx]]<lib_rt_range[1]]
        # Quality-gate the spline points on scribe score, matching the non-timeplex
        # first search (empirical_fit's cor_filter). Higher scribe = better.
        t1_scribe = np.array([output_df.scribe_score[i[idx]] for i in multiples_idxs if len(i)==n_timeplex and output_df.lib_rt[i[idx]]>lib_rt_range[0] and output_df.lib_rt[i[idx]]<lib_rt_range[1]])
        if len(t1_scribe) > 0:
            new_filter = t1_scribe > np.percentile(t1_scribe, 50)
        else:
            new_filter = np.ones(len(t1), dtype=bool)
        filters.append(new_filter)
        rt_spl = fast_modal_lowess(t1[:,1][new_filter],t1[:,0][new_filter], 
                               local_frac=.01,
                               anchors=1000,
                               grid_size=1000,
                               post_smooth_frac=0.01)
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

# --- timeplex ridge-tracker constants ---
RIDGE_X_BINS = 140
RIDGE_Y_BINS = 240
RIDGE_SIGMA = (1.0, 2.5)     # (x, y) gaussian smoothing of the density image
RIDGE_PEAK_PROM_FRAC = 0.05  # min peak prominence as a fraction of the column max
RIDGE_WIGGLE = 0.30          # max per-channel deviation from the joint line, as a fraction of d
RIDGE_CORE_PCT = 2           # trim this %% of points off each tail before fitting (dense core)


def _extrap_spline(xg, curve, data_x, edge_frac=0.05):
    """
    Interpolate the modal-LOESS curve within the channel's data range; beyond it,
    extrapolate linearly with a slope least-squares-fit over the outer ``edge_frac``
    of the DATA POINTS (by count, so the window tracks point density rather than the
    x-axis), clamped non-negative so RT stays monotone. Keeps the body intact —
    unlike isotonic, the sparse tails don't reshape it.
    """
    xg = np.asarray(xg, float)
    curve = np.asarray(curve, float)
    dx = np.asarray(data_x, float)
    inner = interp1d(xg, curve, bounds_error=False, fill_value=(curve[0], curve[-1]))
    if dx.size < 2:
        return inner
    lo_x, hi_x = float(dx.min()), float(dx.max())
    lo_edge = np.percentile(dx, 100 * edge_frac)        # first edge_frac of points
    hi_edge = np.percentile(dx, 100 * (1 - edge_frac))  # last  edge_frac of points

    def slope(a, b):
        mask = (xg >= a) & (xg <= b)
        return max(0.0, np.polyfit(xg[mask], curve[mask], 1)[0]) if mask.sum() >= 2 else 0.0

    lo_slope, hi_slope = slope(lo_x, lo_edge), slope(hi_edge, hi_x)
    y_lo, y_hi = float(inner(lo_x)), float(inner(hi_x))

    def f(r):
        r = np.asarray(r, float)
        y = inner(r)
        y = np.where(r < lo_x, y_lo + lo_slope * (r - lo_x), y)
        return np.where(r > hi_x, y_hi + hi_slope * (r - hi_x), y)
    return f


def _relax_channels(x, y, shape, offsets, d, assign, wiggle=RIDGE_WIGGLE):
    """
    Let each channel wiggle into its own local mode. The joint fit
    (shape + offsets[k]) is the anchor: using the supplied per-point channel
    assignment (gated at half-spacing so an outlier can't reshape a channel), fit
    a per-channel modal LOESS to those points, then soft-clamp the result so it
    can only deviate ±wiggle*d from the joint line (tanh — linear for small
    deviations, saturating smoothly toward the cap). Returns a list of per-channel
    callables mapping the x-axis (library or predicted iRT) to observed RT.
    """
    K = len(offsets)
    xg = np.linspace(float(np.min(x)), float(np.max(x)), 500)
    base = np.asarray(shape(xg))
    shape_x = np.asarray(shape(x))
    within = np.abs(y - (shape_x + offsets[assign])) <= 0.5 * d   # gate outliers
    cap = wiggle * d
    rt_spls = []
    for k in range(K):
        anchor = base + offsets[k]
        m = (assign == k) & within
        # Fit only the dense core of the channel's points: the sparse off-diagonal
        # tails (e.g. CNN iRT-prediction errors after fine-tuning) otherwise hijack
        # the modal LOESS and fold the curve back. Tails are handled by extrapolation.
        xm, ym = x[m], y[m]
        if xm.size >= 20:
            lo = np.percentile(xm, RIDGE_CORE_PCT)
            hi = np.percentile(xm, 100 - RIDGE_CORE_PCT)
            c = (xm >= lo) & (xm <= hi)
            xm, ym = xm[c], ym[c]
        if xm.size >= 20:
            res = min(100, int(xm.size))
            itp = fast_modal_lowess(xm, ym, local_frac=0.01, anchors=res,
                                    grid_size=res, post_smooth_frac=max(0.01, 5 / res))
            curve = np.asarray(itp(xg))
        else:
            curve = anchor.copy()
        curve = anchor + cap * np.tanh((curve - anchor) / cap)    # soft-limit the wiggle
        # Body kept as-is; tails get a robust linear extrapolation (see _extrap_spline)
        # so out-of-core / out-of-range iRTs don't flat-clamp or chase noise.
        rt_spls.append(_extrap_spline(xg, curve, xm))
        logger.info(f"Ridge channel {k}: {int(m.sum())} pts, "
                    f"max wiggle {np.max(np.abs(curve - anchor)):.3f} (cap {cap:.3f})")
    return rt_spls


def _timeplex_scribe_filter(output_df, results_folder=None):
    """
    Scribe-score percentile sweep for the timeplex first search — a standalone
    copy of empirical_fit's stepping so the non-timeplex path is left untouched.
    Steps the scribe cutoff up (20→95) until the alignment residuals are clean
    and returns the scribe-only cor_filter (no single-curve RT tolerance, which
    would collapse the channel structure the ridge tracker needs).
    """
    for feature_percentile in range(20, 100, 5):

        cor_filter = np.logical_and.reduce(
            [output_df[feat] > np.percentile(output_df[feat], feature_percentile)
             for feat in [
                "scribe_score",
             ]
             ]
        )

        ## Only fit when fewer than this many peaks
        if sum(cor_filter)>config.max_num_prelim_search:
            continue

        f = fast_modal_lowess(output_df.lib_rt[cor_filter],
                        output_df.rt[cor_filter],
                        .01,
                        anchors=1000,
                        grid_size=1000,
                        post_smooth_frac=0.01)

        plt.subplots()
        plt.scatter(output_df.lib_rt[cor_filter],
                    output_df.rt[cor_filter], s=1,alpha=.2)
        plt.scatter(output_df.lib_rt[cor_filter],
                    f(output_df.lib_rt[cor_filter]),edgecolor="none", s=1)
        plt.scatter(f.anchor_x, f.anchor_y, color="red", s=8, edgecolor="none")
        plt.title(str(feature_percentile))
        if results_folder is not None:
            plt.savefig(results_folder + f"/first_search/Percentile_{feature_percentile}.png",
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

        k = 3.29 * sigmas[0] # middle 99.9%
        p_in = 2.0 * norm.cdf(k / sigmas) - 1.0
        num = weights * p_in
        partial_posterior = num[0] / num.sum()

        plot_rt_residuals_mixture(
            residuals=res,
            feat=feature_percentile,
            weights=weights,
            sigmas=sigmas,
            results_folder=results_folder
        )

        logger.info(
            f"Testing Percentile: {feature_percentile}, "
            f"Ratio: {outside_ratio:.4f}, #IDs: {cor_filter.sum()}, Partial Posterior: {partial_posterior:.4f}"
        )

        # new + existing stopping criteria
        if (partial_posterior >= 0.99 or
                outside_ratio < 0.05 or
                (cor_filter.sum() - bad_IDs.sum()) < 800):
            break

    return cor_filter


def _hwhm_mode_sigma(v):
    """Fit-free mode + HWHM-derived sigma (sigma = HWHM / 1.1774) of a 1-D sample.

    Reads the width off the smoothed-histogram half-max crossings, so it ignores the
    tails / contamination (no curve fit, no truncation bias)."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    hi = np.percentile(v, 99)
    h, e = np.histogram(v[v <= hi], bins=120)
    c = 0.5 * (e[:-1] + e[1:])
    hs = gaussian_filter(h.astype(float), 1.5)
    pk = int(np.argmax(hs))
    mode = float(c[pk])
    half = hs[pk] / 2.0
    li = pk
    while li > 0 and hs[li] > half:
        li -= 1
    ri = pk
    while ri < len(hs) - 1 and hs[ri] > half:
        ri += 1
    sigma = max(min(mode - c[li], c[ri] - mode) / 1.1774, 1e-3)
    return mode, sigma


def timeplex_two_stage_gate(apex_df, n_timeplex, n_sigma=2.0, n_rt_bins=4):
    """[exp1_2stage] Two-stage, RT-aware per-pair gate with triplet-toss.

    For each channel pair (a,b) the per-peptide gap |rt_b - rt_a| is gated twice
    (fit-independent): (1) a GLOBAL window mode +/- n_sigma*sigma over the whole RT range,
    then (2) an RT-LOCAL re-gate -- within each RT bin (n_rt_bins quantiles of the pair's
    mean RT, over the globally-passing points) recompute mode +/- n_sigma*sigma and keep
    again. A pair passes only if it clears both. Then TRIPLET-TOSS: a peptide's apexes are
    kept only if ALL pairs pass; if any pair is bad the whole triplet is dropped (the
    timeplex channel assignment is no longer trustworthy). Width = fit-free HWHM (min of
    the two half-widths / 1.1774), same as the single-stage gate.
    """
    K = int(n_timeplex)
    df = apex_df.reset_index(drop=True)
    if K < 2:
        return np.ones(len(df), dtype=bool)
    wide = df.pivot_table(index="stripped_seq", columns="channel", values="rt",
                          aggfunc="first").dropna(how="any")
    seqs = wide.index.to_numpy()
    rt = {k: wide[k].to_numpy(dtype=float) for k in wide.columns}
    chans = sorted(rt)

    keep_pep = np.ones(len(seqs), dtype=bool)          # triplet-toss accumulator
    for a, b in combinations(chans, 2):
        gap = np.abs(rt[b] - rt[a])
        mid = 0.5 * (rt[a] + rt[b])
        gm, gs = _hwhm_mode_sigma(gap)                 # stage 1: global
        gpass = np.abs(gap - gm) <= n_sigma * gs
        lpass = np.zeros(len(gap), dtype=bool)          # stage 2: RT-local
        if gpass.sum() >= max(4 * n_rt_bins, 40):
            edges = np.percentile(mid[gpass], np.linspace(0, 100, n_rt_bins + 1))
            for j in range(n_rt_bins):
                lo, hi = edges[j], edges[j + 1]
                m = ((mid >= lo) & (mid < hi)) if j < n_rt_bins - 1 else ((mid >= lo) & (mid <= hi))
                in_bin_pass = m & gpass
                if in_bin_pass.sum() >= 30:
                    lm, ls = _hwhm_mode_sigma(gap[in_bin_pass])
                    lpass |= m & (np.abs(gap - lm) <= n_sigma * ls)
                else:
                    lpass |= m                          # too few to re-gate; accept global
        else:
            lpass[:] = True
        pair_pass = gpass & lpass
        keep_pep &= pair_pass
        logger.info(f"two-stage gate pair {a}-{b}: global pass {gpass.mean():.3f}, "
                    f"+RT-local {pair_pass.mean():.3f}")

    keep_lookup = {s: bool(keep_pep[i]) for i, s in enumerate(seqs)}
    mask = np.array([keep_lookup.get(s, False) for s in df["stripped_seq"]], dtype=bool)
    logger.info(f"two-stage triplet-toss gate: kept {int(mask.sum())}/{len(mask)} apexes "
                f"({mask.mean():.3f})")
    return mask


def timeplex_pair_nn_gate(apex_df, n_timeplex, n_sigma=2.0):
    """Fit-independent per-channel-pair nearest-neighbor gate over the apex set.

    Uses only raw observed RT, channel, and sequence -- nothing from any RT fit. For
    each unordered channel pair (a, b) it measures the per-peptide cross-channel
    distance |rt_a - rt_b|, builds a window mode +/- n_sigma*sigma from the fit-free
    HWHM of that distance distribution, and applies the EXONERATION rule: flag both
    members of an out-of-window pair, then un-flag any member that still agrees with
    another channel. A member is therefore kept iff it is in-window with at least one
    other channel; only members that disagree with EVERY other channel are dropped.

    This replaces the old training-data filter, which de-offset each apex by the
    ridge-fit offsets (fit-dependent) and assumed equal channel spacing.

    Parameters
    ----------
    apex_df : pandas.DataFrame
        Apex rows with columns 'stripped_seq', 'rt', 'channel' (one row per
        peptide x channel; from select_timeplex_alignment_set).
    n_timeplex : int
        Number of channels K.
    n_sigma : float
        Half-width of the per-pair keep window in sigma. Default 2.0.

    Returns
    -------
    np.ndarray[bool]
        Keep mask aligned to apex_df rows.
    """
    K = int(n_timeplex)
    df = apex_df.reset_index(drop=True)
    if K < 2:
        return np.ones(len(df), dtype=bool)

    # one row per peptide, a column of rt per channel
    wide = df.pivot_table(index="stripped_seq", columns="channel", values="rt",
                          aggfunc="first")
    full = wide.dropna(how="any")              # peptides present in all K channels
    seqs = full.index.to_numpy()
    rt = {k: full[k].to_numpy(dtype=float) for k in full.columns}

    # per-channel keep (exoneration): keep_c = OR over the pairs containing c
    keep_ch = {k: np.zeros(len(seqs), dtype=bool) for k in rt}
    for a, b in combinations(sorted(rt), 2):
        d = np.abs(rt[b] - rt[a])
        mode, sigma = _hwhm_mode_sigma(d)
        lo, hi = mode - n_sigma * sigma, mode + n_sigma * sigma
        in_win = (d >= lo) & (d <= hi)
        keep_ch[a] |= in_win
        keep_ch[b] |= in_win
        logger.info(f"NN gate pair {a}-{b}: window [{lo:.2f}, {hi:.2f}] "
                    f"(mode {mode:.2f}, sigma {sigma:.2f}), "
                    f"in-window {in_win.mean():.3f}")

    # map per-(peptide, channel) keep back to apex rows
    keep_lookup = {}
    for j, s in enumerate(seqs):
        for k in rt:
            keep_lookup[(s, int(k))] = bool(keep_ch[k][j])
    mask = np.array([keep_lookup.get((s, int(c)), False)
                     for s, c in zip(df["stripped_seq"], df["channel"])], dtype=bool)
    logger.info(f"NN gate: kept {int(mask.sum())}/{len(mask)} apexes "
                f"({mask.mean():.3f})")
    return mask


def fit_timeplex_ridges(output_df, n_timeplex, results_folder=None):
    """
    Align timeplex channels as ONE shared iRT->RT shape + K equally-spaced vertical
    offsets (constant per-channel RT shift), recovered by 2D-density ridge tracking.

    Unlike the old get_multiples/split_timePlex approach, this needs no per-peptide
    completeness and no positional assignment: it finds the K parallel RT bands in
    the (lib_rt, obs_rt) density and enforces a single shared shape with a common
    channel gap d.

    Parameters
    ----------
    output_df : pandas.DataFrame
        First-search PSMs with columns: scribe_score, lib_rt, rt, mz,
        relative_error_ms1.
    n_timeplex : int
        Number of channels K (>= 1).

    Returns
    -------
    rt_spls : list[callable]
        K callables mapping library iRT -> observed RT (shared_shape + offsets[k]).
    shared_shape : callable
        The single de-offset iRT->RT curve.
    offsets : np.ndarray
        Length-K, equally spaced (base + d*arange(K)).
    d : float
        Common channel gap (np.inf when K == 1).
    filtered_output : pandas.DataFrame
        Kept PSMs with lib_rt, rt (de-offset to the shared shape), mz_diffs, mz —
        the shape alignment_plots expects.
    residuals : np.ndarray
        Signed residual of each kept PSM to its assigned ridge (for the GMM
        tolerance, the residual mixture plot, and the CDF plot).
    """
    K = int(n_timeplex)
    if K < 1:
        raise RuntimeError("fit_timeplex_ridges requires n_timeplex >= 1")

    # PSMs arrive already scribe-filtered and channel-assigned upstream
    # (select_timeplex_alignment_set), one apex per (peptide, channel). Use the
    # supplied channel labels instead of nearest-snap assignment.
    kept = output_df.reset_index(drop=True)
    x = kept["lib_rt"].to_numpy(dtype=float)
    y = kept["rt"].to_numpy(dtype=float)
    if "channel" in kept.columns:
        channel = kept["channel"].to_numpy(dtype=int)
    else:
        channel = np.zeros(len(kept), dtype=int)
    logger.info(f"Ridge tracker: {len(kept)} IDs")
    if len(x) < 10:
        raise RuntimeError(f"Ridge tracker: too few IDs ({len(x)})")

    def make_spline(off):
        def _f(r):
            return np.asarray(shared_shape(np.asarray(r, dtype=float))) + off
        return _f

    if K == 1:
        # Single channel: just the global modal trend, no offsets.
        shared_shape = fast_modal_lowess(x, y, local_frac=.01, anchors=1000,
                                         grid_size=1000, post_smooth_frac=0.01)
        offsets = np.array([0.0])
        d = float("inf")
    else:
        # --- build smoothed density image ---
        H, xedges, yedges = np.histogram2d(x, y, bins=[RIDGE_X_BINS, RIDGE_Y_BINS])
        H = gaussian_filter(H, sigma=RIDGE_SIGMA)
        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])

        def column_peaks(col, min_dist_bins):
            """Up to K strongest, well-separated peak positions (sorted by y)."""
            if col.max() <= 0:
                return np.array([])
            peaks, _ = signal.find_peaks(col, prominence=RIDGE_PEAK_PROM_FRAC * col.max(),
                                         distance=max(1, min_dist_bins))
            if len(peaks) == 0:
                return np.array([])
            top = peaks[np.argsort(col[peaks])[::-1][:K]]
            return np.sort(y_centers[top])

        min_dist_bins = int(0.02 * RIDGE_Y_BINS)
        cols = [(x_centers[i], column_peaks(H[i], min_dist_bins))
                for i in range(H.shape[0])]

        npks = np.array([len(c[1]) for c in cols])
        mid = np.median(x_centers)
        lo = x_centers < mid
        logger.info("Ridge peaks/col (low RT): "
                    + str({k: int(((npks == k) & lo).sum()) for k in range(K + 1)}))
        logger.info("Ridge peaks/col (high RT): "
                    + str({k: int(((npks == k) & ~lo).sum()) for k in range(K + 1)}))

        # --- global channel spacing d: mode of within-column adjacent peak gaps ---
        gap_list = [np.diff(c[1]) for c in cols if len(c[1]) >= 2]
        all_gaps = np.concatenate(gap_list) if gap_list else np.array([])
        if all_gaps.size == 0:
            raise RuntimeError("Ridge tracker: no column yielded >=2 peaks; "
                               "lower RIDGE_PEAK_PROM_FRAC or check --num_timeplex")
        gh, ge = np.histogram(all_gaps, bins=60)
        gmode_lo, gmode_hi = ge[np.argmax(gh)], ge[np.argmax(gh) + 1]
        d = float(np.median(all_gaps[(all_gaps >= gmode_lo) & (all_gaps <= gmode_hi)]))
        logger.info(f"Ridge tracker: global spacing d={d:.4f} "
                    f"(from {all_gaps.size} within-column gaps)")

        # --- seed: densest K-peak column most consistent with equal spacing d ---
        gap_tol = 0.35 * d
        cand = []
        for i in range(H.shape[0]):
            m = cols[i][1]
            if len(m) != K:
                continue
            if np.max(np.abs(np.diff(m) - d)) <= gap_tol:
                cand.append((H[i].sum(), i))
        if not cand:
            raise RuntimeError(f"Ridge tracker: no column had K={K} equally-spaced "
                               f"peaks (d={d:.4f}); check --num_timeplex / prominence")
        seed_i = max(cand)[1]
        seed_modes = cols[seed_i][1]
        spacing = float(np.min(np.diff(seed_modes)))
        link_thresh = 0.45 * spacing
        logger.info(f"Ridge tracker: seed @ libRT={x_centers[seed_i]:.2f}, "
                    f"modes={np.round(seed_modes,3)}, spacing={spacing:.3f} "
                    f"({len(cand)} equal-spaced candidate columns)")

        tracks = [[(x_centers[seed_i], seed_modes[k])] for k in range(K)]
        rel = seed_modes - seed_modes.mean()   # fixed relative offset prior

        def predict(h, xq):
            """Predict a ridge's y at xq from its recent history h=[(x,y),...]."""
            h = h[-5:]
            if len(h) >= 2:
                xs = np.array([p[0] for p in h]); ys = np.array([p[1] for p in h])
                if xs.max() - xs.min() > 1e-9:
                    sl, ic = np.polyfit(xs, ys, 1)
                    return sl * xq + ic
            return h[-1][1]

        def walk(order):
            hist = [[(x_centers[seed_i], seed_modes[k])] for k in range(K)]
            for i in order:
                m = cols[i][1]
                if len(m) == 0:
                    continue
                xi = x_centers[i]
                # match available peaks to ridges via each ridge's slope prediction
                pred = np.array([predict(hist[k], xi) for k in range(K)])
                cost = np.abs(pred[:, None] - m[None, :])
                ri, ci = linear_sum_assignment(cost)
                matched, used = {}, set()
                for r, c in zip(ri, ci):
                    if cost[r, c] <= link_thresh:
                        matched[r] = m[c]; used.add(c)
                # recover unmatched ridges from siblings via the fixed-offset prior
                if matched:
                    s = np.median([matched[r] - rel[r] for r in matched])
                    for k in range(K):
                        if k in matched:
                            continue
                        target = s + rel[k]
                        avail = [c for c in range(len(m)) if c not in used]
                        if avail:
                            c = min(avail, key=lambda c: abs(m[c] - target))
                            if abs(m[c] - target) <= link_thresh:
                                matched[k] = m[c]; used.add(c)
                for r, val in matched.items():
                    tracks[r].append((xi, val))
                    hist[r].append((xi, val))

        walk(range(seed_i - 1, -1, -1))       # leftward
        walk(range(seed_i + 1, H.shape[0]))   # rightward

        # --- one shared shape: pool de-offset points, single lowess ---
        pool_x, pool_s = [], []
        for k in range(K):
            for (xi, val) in tracks[k]:
                pool_x.append(xi); pool_s.append(val - rel[k])
        pool_x = np.asarray(pool_x); pool_s = np.asarray(pool_s)
        o = np.argsort(pool_x)
        shape_sm = lowess(pool_s[o], pool_x[o], frac=0.15, it=0)
        shared_shape = interp1d(shape_sm[:, 0], shape_sm[:, 1], bounds_error=False,
                                fill_value=(shape_sm[0, 1], shape_sm[-1, 1]))

        # --- equal-spacing offsets: offset[k] = base + k*d, d fixed to global ---
        order_k = np.argsort([
            np.median([v - float(shared_shape(xi)) for (xi, v) in tracks[k]])
            if tracks[k] else rel[k] for k in range(K)])
        kk, rr = [], []
        for ki, k in enumerate(order_k):
            for (xi, val) in tracks[k]:
                kk.append(ki); rr.append(val - float(shared_shape(xi)))
        kk = np.asarray(kk, float); rr = np.asarray(rr, float)
        base_off = float(np.mean(rr - d * kk))
        offsets = base_off + d * np.arange(K)
        logger.info(f"Ridge tracker: offsets={np.round(offsets,4)}, "
                    f"gaps={np.round(np.diff(offsets),4)}")

    # --- per-channel curves: joint shape+offset, relaxed into local modes (wiggle) ---
    if K == 1:
        rt_spls = [make_spline(0.0)]
    else:
        rt_spls = _relax_channels(x, y, shared_shape, offsets, d, channel)

    # --- signed residual of each PSM to its pre-assigned channel ridge ---
    shape_x = np.asarray(shared_shape(x))
    residuals = y - (shape_x + offsets[channel])

    filtered_output = pd.DataFrame({
        "lib_rt": x,
        "rt": y,                                       # original observed RT
        "channel": channel,
        "stripped_seq": kept["stripped_seq"].to_numpy(),
        "mz_diffs": kept["relative_error_ms1"].to_numpy(dtype=float),
        "mz": kept["mz"].to_numpy(dtype=float),
        "scribe_score": kept["scribe_score"].to_numpy(dtype=float),
    })

    return rt_spls, shared_shape, offsets, d, filtered_output, residuals


def MZRTfit_timeplex(dia_spectra,librarySpectra,mz_tol,ms1=False,results_folder=None,ms2=False,mass_tag=None,SILAC=None):
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
    logger.info("Initial search")

    #################################################################################

    # Run the peppy_sage MS2 library search (same as the non-timeplex MZRTfit path)
    import src.preliminary_search as preliminary_search
    from src.elution_analysis import calculate_elution_width, select_timeplex_alignment_set
    output_df = preliminary_search.fit_with_features(
        dia_spectra, librarySpectra, mass_tag, SILAC,
        ms1_ppm_error=20, ms2_ppm_error=10,
    )

    # Elution width — provides elution_sd for the RT-tolerance GMM (same formula as
    # the non-timeplex path).
    _fwhm, elution_sd, output_df = calculate_elution_width(output_df)
    logger.info(f"Mean elution width: FWHM {_fwhm:.4f}, SD {elution_sd:.4f}")

    # Build the RT-alignment input set (vectorized): sweep scribe cutoffs and pick
    # the one that maximizes peptides seen in exactly K=n_timeplex RT clusters, take
    # each cluster's apex, and assign channel 0..K-1 by RT rank. Elutions are
    # seconds-wide and timeplex windows minutes apart, so an 8*elution_sd RT gap
    # separates the windows.
    n_timeplex = config.args.num_timeplex
    rt_gap = 8 * elution_sd
    apex_pl, scribe_cut, n_pep = select_timeplex_alignment_set(output_df, n_timeplex, rt_gap)
    logger.info(f"Timeplex alignment set: scribe >= {scribe_cut:.4f}, {n_pep} peptides "
                f"x {n_timeplex} channels = {apex_pl.height} IDs (RT gap {rt_gap:.4f})")

    output_df = apex_pl.sort("rt").to_pandas().reset_index(drop=True)

    id_keys = list(zip(output_df["seq"], output_df["z"]))

    # Dump the first-search table (one apex row per elution cluster) for debugging,
    # mirroring the non-timeplex MZRTfit path.
    if results_folder is not None:
        output_df.to_csv(results_folder+"/first_search/firstSearch.tsv", index=False, sep='\t')

    updatedLibrary = copy.deepcopy(librarySpectra)

    # --- Phase 2: fit-independent per-channel-pair NN gate ---------------------
    # Clean the apex set with the per-pair nearest-neighbor gate (raw |dRT| windows,
    # exoneration rule). This uses NO RT fit, so the empirical and fine-tuned models
    # are fit and compared on the same gated data.
    gate_mask = timeplex_two_stage_gate(output_df, n_timeplex)   # [exp1_2stage]
    gated_df = output_df[gate_mask].reset_index(drop=True)

    # Empirical ridge fit on the GATED set (also the source of channel offsets, the
    # channel separation d, and the OriginalRTfit shape).
    rt_spls, emp_shape, offsets, d, filtered_output, residuals = \
        fit_timeplex_ridges(gated_df, n_timeplex, results_folder)

    channel = filtered_output["channel"].to_numpy()
    filtered_output["updated_lib_rt"] = filtered_output["lib_rt"]   # x-axis = library iRT

    # rt_spls from fit_timeplex_ridges are per-channel iRT->RT curves (empirical default).
    emp_curves = rt_spls
    plot_curves = emp_curves                      # what alignment_plots draws
    emp_data, emp_p, emp_cdf_auc = cdf_data(residuals)
    chosen_diffs = residuals
    chosen_channel = channel                      # channel labels aligned to chosen_diffs
    pred_data = pred_p = None
    mixture_feat = "_timeplex_empirical"

    if not config.args.use_emp_rt:
        logger.info("Per-channel RT prediction (timeplex)")
        # --- Phase 3: bagged, OOB, (1-PEP) soft-weighted fine-tune per channel ----
        # Each channel is fine-tuned independently on its gated apexes with a 5-member
        # bagged ensemble; held-out OOB residuals make the empirical-vs-fine-tuned
        # comparison fair. The CNN predicts the channel's observed RT directly, so the
        # per-channel {seq: rt} dict is already on the observed scale run_jmod consumes.
        seqs_all = filtered_output["stripped_seq"].to_numpy()
        obs_rt = filtered_output["rt"].to_numpy(dtype=float)
        uniq = sorted({updatedLibrary[k]["seq"] for k in librarySpectra})
        model_path = rt_model_path()

        lib_pred_by_ch, oob_by_ch, ft_hist = [], [], []
        for k in range(n_timeplex):
            sel = channel == k
            oob_resid, lib_pred, hist = bagged_finetune_channel(
                seqs_all[sel], obs_rt[sel], uniq, model_path,
                results_path=results_folder, channel=k)
            lib_pred_by_ch.append(lib_pred)
            oob_by_ch.append((k, oob_resid))
            ft_hist.append(hist)
            logger.info(f"  timeplex channel {k}: OOB |dRT| median = "
                        f"{np.median(np.abs(oob_resid)):.3f} min")

        # per-channel fine-tune training curve (held-out OOB MedAE vs epoch)
        if results_folder is not None:
            try:
                os.makedirs(results_folder + "/first_search/fine_tuning", exist_ok=True)
                figc, axc = plt.subplots(figsize=(7, 5))
                for k, h in enumerate(ft_hist):
                    axc.plot(range(len(h["oob_medae"])), h["oob_medae"],
                             marker=".", label=f"T{k}")
                    axc.axvline(h["best_epoch"], color=f"C{k}", ls=":", lw=1)
                axc.set_xlabel("epoch"); axc.set_ylabel("held-out OOB MedAE (min)")
                axc.set_title("Timeplex fine-tune training curves")
                axc.legend()
                figc.savefig(results_folder + "/first_search/fine_tuning/"
                             "RT_finetune_oob_medae.png", dpi=300, bbox_inches="tight")
                plt.close(figc)
            except Exception as e:
                logger.warning(f"Could not save fine-tune training curve: {e}")

        pooled_oob = np.concatenate([r for _, r in oob_by_ch])
        pooled_channel = np.concatenate([np.full(len(r), k, dtype=int)
                                         for k, r in oob_by_ch])
        pred_data, pred_p, pred_cdf_auc = cdf_data(pooled_oob)

        # Higher CDF-AUC = tighter residuals = better (now a FAIR, held-out comparison).
        if pred_cdf_auc > emp_cdf_auc:
            logger.info(f"Fine-tuned RT model chosen (timeplex) "
                        f"[empirical CDF-AUC={emp_cdf_auc:.4f} vs fine-tuned CDF-AUC={pred_cdf_auc:.4f}]")
            mixture_feat = "_timeplex_fine_tuned"
            rt_spls = [lib_pred_by_ch[k] for k in range(n_timeplex)]  # per-channel {seq: rt}
            # NOTE: the model SELECTION above uses OOB residuals (a fair, held-out
            # comparison so the memorizing CNN doesn't always win). But the reported
            # sigma, the RTfit plot, and the RT tolerance use the IN-SAMPLE full-ensemble
            # residuals (the in-distribution precision) -- consistent with the empirical
            # branch, whose ridge residuals are also in-sample.
            # TODO(rt-tolerance): in-sample full-ensemble residuals are overfit-optimistic
            # (the 5 members have memorized the training peptides), so this UNDER-estimates
            # the RT spread for unseen library peptides. We use it because in-sample
            # ensemble worked best empirically here, and on this data the tolerance clamps
            # to d/2 regardless. The honest deployed-model estimate is the held-out full
            # ensemble (~0.75 vs in-sample ~0.6 vs OOB ~0.9); revisit if a run's channels
            # are far enough apart that sigma -- not the d/2 overlap clamp -- drives the
            # tolerance, where too-tight a window would cost real IDs.
            ft_pred = np.array([lib_pred_by_ch[c][s]
                                for s, c in zip(seqs_all, channel)], dtype=float)
            chosen_diffs = obs_rt - ft_pred             # in-sample full-ensemble residual
            chosen_channel = channel
            # RTfit: de-offset the full-ensemble prediction to a shared axis (each channel
            # minus its own median level) so matched peptides share an x and the channels
            # show as parallel diagonals offset by their RT gap. Plot-only; returned
            # rt_spls dicts (full ensemble) are what the search uses.
            ch_level = np.array([np.median(obs_rt[channel == k])
                                 for k in range(n_timeplex)])
            ch_rel = ch_level - ch_level.mean()        # centered channel offsets
            deoff_pred = ft_pred - ch_rel[channel]      # shared axis across channels
            filtered_output["updated_lib_rt"] = deoff_pred
            plot_curves = []
            for k in range(n_timeplex):
                mk = channel == k
                if mk.sum() >= 20:
                    plot_curves.append(fast_modal_lowess(
                        deoff_pred[mk], obs_rt[mk], local_frac=.05, anchors=1000,
                        grid_size=1000, post_smooth_frac=0.02))
                else:
                    plot_curves.append(lambda v: np.asarray(v, dtype=float))
        else:
            logger.info(f"Empirical RT model chosen (timeplex) "
                        f"[empirical CDF-AUC={emp_cdf_auc:.4f} vs fine-tuned CDF-AUC={pred_cdf_auc:.4f}]")
    else:
        logger.info("RT model: empirical ridge alignment (timeplex; --use_emp_rt)")

    # --- Phase 4: RT tolerance from the per-channel narrow GMM component -------
    # Fit a 2-component zero-mean GMM per channel on the chosen residuals, take the
    # narrow sigma, and average across channels.
    narrow_sigmas = []
    for k in range(n_timeplex):
        rk = chosen_diffs[chosen_channel == k]
        if len(rk) >= 10:
            _, sg = fit_zero_mean_gmm_1d(rk, n_components=2)
            narrow_sigmas.append(float(np.sort(sg)[0]))
    if narrow_sigmas:
        sigma_narrow = float(np.mean(narrow_sigmas))
    else:
        _, sg = fit_zero_mean_gmm_1d(chosen_diffs, n_components=2)
        sigma_narrow = float(np.sort(sg)[0])
    boundary = 4 * sigma_narrow + 8 * elution_sd
    logger.info(f"Per-channel narrow sigma {[round(s, 3) for s in narrow_sigmas]} "
                f"-> mean {sigma_narrow:.3f}; RT boundary {boundary:.3f}")

    # representative pooled GMM for the residual-mixture plot
    weights, sigmas = fit_zero_mean_gmm_1d(chosen_diffs, n_components=2)
    sigmas = np.sort(sigmas)

    #####  Assume that the mz error is independent of n_timeplex
    # Use the precomputed per-PSM MS1 error (same as the non-timeplex MZRTfit path),
    # so the mz fit is consistent with filtered_output.mz_diffs used in plotting.
    diffs = output_df["relative_error_ms1"].to_numpy()
    
    
    
    
    
    ################################################
    ########### correct mz errors wrt RT    ########
    ################################################
    
    rts = np.array([updatedLibrary[(i[0],float(i[1]))]["iRT"] for i in id_keys])#np.array([i[0] for i in t_vals[0]])
    # rt_filter_bool = filter_rts_by_dense(rts,30)
    # rt_filter_bool = np.logical_and(rts>15,rts<30)
    rt_mz_filter_bool = np.array(output_df.matched_lib_pct)>90 # use as proxy for correct IDs
    f_rt_mz = fast_modal_lowess(rts[rt_mz_filter_bool],np.array(diffs)[rt_mz_filter_bool],local_frac=.02,
                               anchors=1000,
                               grid_size=1000,
                               post_smooth_frac=0.01)
    # plt.scatter(rts[rt_mz_filter_bool],np.array(diffs)[rt_mz_filter_bool],label="Original_MZ",s=1,alpha=.1)
    # plt.scatter(output_df.rt,f_rt_mz(output_df.rt),s=1,alpha=.2)
    
    # plt.scatter(id_mzs,diffs,label="Original_MZ",s=1,alpha=.1)
    # plt.scatter(id_mzs,diffs-f_rt_mz(output_df.rt),label="Original_MZ",s=1,alpha=.1)
    
    # mz_spl = twostepfit(np.array(id_mzs)[rt_mz_filter_bool],(diffs-f_rt_mz(output_df.rt))[rt_mz_filter_bool],1)
    mz_spl = fast_modal_lowess(np.array(output_df.mz)[rt_mz_filter_bool],(diffs-f_rt_mz(output_df.rt))[rt_mz_filter_bool])
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
    
    # ensure there is no overlap (d = the common channel gap)
    if new_rt_tol>np.abs(d/2):
        logger.warning("Warning; Library RTs overlapping")
        new_rt_tol = np.abs(d/2)*.99 # ensure no overlap
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
            
        # Per-channel RT plots (channel-aware alignment_plots); mz panels shared.
        rt_dist_params = fit_gaussian(chosen_diffs)
        plot_rt_residuals_mixture(chosen_diffs, feat=mixture_feat, weights=weights, sigmas=sigmas,
                                  results_folder=results_folder)
        alignment_plots(filtered_output, emp_shape, emp_shape, f_rt_mz, mz_spl, rt_dist_params,
                        results_folder=results_folder, channels=channel, rt_spls=plot_curves, offsets=offsets)
        cdf_plots(emp_data, emp_p, config.rt_percentile, boundary, pred_data, pred_p,
                  results_folder=results_folder)
        plt.close("all")
    
    
    # if ms2:
    #     return (rt_spls, mz_func, ms2_func), updatedLibrary
    # else:
    # Return the measured elution FWHM (third slot) so downstream fragment-ion
    # correlation features can be computed; returning None here makes
    # compute_fragment_correlations bail out to all-NaN for every row.
    return (rt_spls, mz_func), updatedLibrary, _fwhm



