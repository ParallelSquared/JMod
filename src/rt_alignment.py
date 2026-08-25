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


def _ls_line(dx, yy, wt=None):
    """
    (Weighted) least-squares line y = value_at_zero + slope*dx.

    dx is measured from the point the line is read off at, so the intercept IS
    the fitted value there. Returns (slope, value_at_zero); a degenerate window
    (all dx equal) falls back to a flat line at the median.
    """
    if dx.size == 0:
        return 0.0, np.nan
    if wt is None:
        wt = np.ones_like(dx)
    sw = wt.sum()
    sx = np.dot(wt, dx)
    sxx = np.dot(wt, dx * dx)
    den = sw * sxx - sx * sx
    if dx.size < 2 or sw <= 0 or not np.isfinite(den) or den <= 1e-12 * max(1.0, sxx):
        return 0.0, float(np.median(yy))
    sy = np.dot(wt, yy)
    slope = (sw * np.dot(wt, dx * yy) - sx * sy) / den
    return float(slope), float((sy - slope * sx) / sw)


def _tricube(dx):
    """
    LOWESS distance weights over a window: 1 at the anchor, 0 at the window edge.

    Without these the local line is fit with every point in the window counting
    equally, which is fine where the window is narrow in x but badly over-weights
    distant points where it is not -- i.e. exactly in the sparse ends this fit is
    meant to get right.
    """
    h = np.max(np.abs(dx))
    if not np.isfinite(h) or h <= 0:
        return np.ones_like(dx)
    u = np.abs(dx) / (h * 1.000001)
    return np.power(1.0 - np.power(u, 3), 3)


def _robust_line(dx, yy):
    """
    Outlier-resistant starting line for a local window: slope from the median of
    the upper half minus the median of the lower half (the window is already
    sorted by x), level from the median residual. Used only to give the modal
    step a trend to work against, so it does not have to be efficient — just
    insensitive to the wrong IDs scattered through the window.
    """
    m = dx.size
    mid = m // 2
    if mid >= 2:
        x_lo, x_hi = np.median(dx[:mid]), np.median(dx[mid:])
        if x_hi - x_lo > 0:
            slope = (np.median(yy[mid:]) - np.median(yy[:mid])) / (x_hi - x_lo)
            if np.isfinite(slope):
                return float(slope), float(np.median(yy - slope * dx))
    return _ls_line(dx, yy)


def _clamp_slope(slope, overall):
    """Hold an edge slope between zero and twice the overall trend."""
    if not np.isfinite(slope):
        return 0.0
    lo, hi = (0.0, 2.0 * overall) if overall >= 0 else (2.0 * overall, 0.0)
    return float(np.clip(slope, lo, hi))
# How far past the fitted knots the alignment curve may be extrapolated, and how
# far its value may travel while doing so. The library extends well past the range
# where confident first-search IDs exist (on JD0319, 3.1% of the library sits
# beyond the last knot), and inside that region the curve is held flat -- so
# late-eluting precursors are predicted ~8 min early against a ~1.3 min search
# window and cannot be found at all. Tunable so the right values can be measured.
RT_EXTRAP_FRAC = float(os.environ.get("JMOD_RT_EXTRAP_FRAC", 0.25))
RT_YPAD_FRAC = float(os.environ.get("JMOD_RT_YPAD_FRAC", 0.20))


class ModalLowessFit:
    """
    Callable x -> y returned by :func:`fast_modal_lowess`.

    Linear interpolation between the fitted knots. Past either end the curve
    continues with the slope of the outer ``edge_frac`` of the fit instead of
    being held flat, but only for ``max_extrap_frac`` of the fitted x-span, and
    the total excursion is capped at ``y_pad_frac`` of the fitted y-range. An
    edge slope read off a noisy fit is not to be trusted far, and a library value
    well outside the observed range should not map to a nonsensical RT.
    """

    def __init__(self, x, y, edge_frac=0.1, max_extrap_frac=None, y_pad_frac=None):
        max_extrap_frac = RT_EXTRAP_FRAC if max_extrap_frac is None else max_extrap_frac
        y_pad_frac = RT_YPAD_FRAC if y_pad_frac is None else y_pad_frac
        self.x_ = np.asarray(x, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        span = float(self.x_[-1] - self.x_[0]) if self.x_.size > 1 else 0.0
        self.max_extrap_ = max_extrap_frac * span
        self.y_pad_ = y_pad_frac * float(np.ptp(self.y_)) if self.y_.size > 1 else 0.0
        n_edge = max(2, int(np.ceil(edge_frac * self.x_.size)))
        # Edge slopes are held between zero and twice the overall trend. Read off
        # a noisy fit an edge slope can come out with the wrong sign, and
        # extrapolating on it sends points further from the truth than simply
        # holding the curve flat would.
        overall = _ls_line(self.x_ - self.x_[0], self.y_)[0]
        self.slope_lo_ = _clamp_slope(
            _ls_line(self.x_[:n_edge] - self.x_[0], self.y_[:n_edge])[0], overall)
        self.slope_hi_ = _clamp_slope(
            _ls_line(self.x_[-n_edge:] - self.x_[-1], self.y_[-n_edge:])[0], overall)
        # raw (pre-smoothing) modal anchors, for diagnostic plotting
        self.anchor_x = self.x_
        self.anchor_y = self.y_

    def __call__(self, v):
        vv = np.asarray(v, dtype=float)
        flat = vv.ravel()
        out = np.interp(flat, self.x_, self.y_)
        lo = flat < self.x_[0]
        if lo.any():
            d = np.clip(flat[lo] - self.x_[0], -self.max_extrap_, 0.0)
            out[lo] = self.y_[0] + np.clip(self.slope_lo_ * d,
                                           -self.y_pad_, self.y_pad_)
        hi = flat > self.x_[-1]
        if hi.any():
            d = np.clip(flat[hi] - self.x_[-1], 0.0, self.max_extrap_)
            out[hi] = self.y_[-1] + np.clip(self.slope_hi_ * d,
                                            -self.y_pad_, self.y_pad_)
        if vv.ndim == 0:
            return float(out[0])
        return out.reshape(vv.shape)


def fast_modal_lowess(x, y,
                      local_frac=0.2,
                      grid_size=100,
                      anchors=200,
                      post_smooth_frac=0.1):
    """
    Modal LOCAL-LINEAR smoother mapping x to y, robust to the wrong IDs that sit
    off the main trend.

    Each anchor gets a local straight line, not a local constant: the window is
    detrended, the mode of the residuals about that trend picks out the true
    crowd, the line is refit to the crowd only, and the reported value is that
    line read off at the anchor's own x.

    Fitting a local CONSTANT (taking the mode of raw y in the window, as this
    did previously) carries a bias of roughly slope x window-width. Because the
    window is defined by rank, it is narrow in x through the dense middle of an
    RT gradient — where the bias is invisible — but spans many minutes in the
    sparse ends, where every anchor then sees the same crowd of points and
    reports the same y. That is what produced the flat shelves at each end of
    the alignment curve, sitting well above the points they were fit to. A local
    LINEAR fit has no such boundary bias, which is exactly why LOWESS is defined
    that way.

    Parameters
    ----------
    x, y : array
        Points to fit. Non-finite pairs are dropped.
    local_frac : float, optional
        Fraction of the points in each local window. The default is 0.2.
    grid_size : int, optional
        Resolution of the residual grid the mode is searched on, clipped to
        [64, 1024]. The default is 100.
    anchors : int, optional
        Number of positions the local fit is evaluated at. The default is 200.
    post_smooth_frac : float, optional
        LOWESS fraction used to smooth the anchor curve. The default is 0.1.

    Returns
    -------
    ModalLowessFit
        Callable mapping x to y.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = x.size
    if n == 0:
        raise ValueError("fast_modal_lowess: no finite (x, y) pairs to fit")

    order = np.lexsort((y, x))
    x, y = x[order], y[order]

    if n < 10:
        # too few points to localise anything; interpolate them directly
        return ModalLowessFit(*_collapse_ties(x, y))

    # A local LINE needs more points than a local constant did: 20 is the floor,
    # below which the slope is fit to noise.
    w = int(np.clip(int(local_frac * n), min(20, n), n))
    n_anchor = int(np.clip(anchors, 2, n))
    anchor_idx = np.unique(np.linspace(0, n - 1, n_anchor).astype(int))

    # Bin-and-blur KDE: histogram the residuals onto the grid, then convolve with
    # the gaussian kernel. Same mode as the dense (points x grid) kernel matrix
    # the old code built per anchor, at a small fraction of the cost.
    grid_size = int(np.clip(grid_size, 64, 1024))

    modal_vals = np.empty(anchor_idx.size)
    half = w // 2
    for k, i in enumerate(anchor_idx):
        start, end = i - half, i - half + w
        if start < 0:
            start, end = 0, w
        elif end > n:
            start, end = n - w, n
        xw, yw = x[start:end], y[start:end]
        dx = xw - x[i]
        wt = _tricube(dx)

        slope, val0 = _robust_line(dx, yw)
        resid = yw - (val0 + slope * dx)

        # local, robust residual scale -> local bandwidth. The old fixed
        # bandwidth came from the global spread of y, i.e. the whole RT
        # gradient, which is orders of magnitude wider than the scatter about
        # the local trend and washed the mode out to the window centre.
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.std(resid))
        if not np.isfinite(scale) or scale <= 0:
            modal_vals[k] = val0
            continue

        bw = 1.06 * scale * yw.size ** (-0.2)
        lim = 4.0 * scale
        counts, edges = np.histogram(resid, bins=grid_size, range=(-lim, lim),
                                     weights=wt)
        density = gaussian_filter(counts.astype(float),
                                  bw / (2.0 * lim / grid_size),
                                  mode="constant")
        mode = 0.5 * (edges[np.argmax(density)] + edges[np.argmax(density) + 1])

        # refit to the modal crowd, twice: once about the robust line, then once
        # about the refit line, so the level is set by the inliers alone
        line = (slope, val0 + mode)
        crowd = None
        for band in (2.0 * bw, 2.5 * bw):
            resid = yw - (line[1] + line[0] * dx)
            inliers = np.abs(resid) <= band
            if inliers.sum() < max(5, int(0.1 * yw.size)):
                break
            if np.ptp(dx[inliers]) <= 0:
                break
            crowd = yw[inliers]
            line = _ls_line(dx[inliers], yw[inliers], wt[inliers])

        # A local line is read off at the anchor's own x, which at the first and
        # last anchors means reading it at the very edge of a one-sided window.
        # Where the window is wide in x and the points are noisy, an overshooting
        # slope can put that value outside the range of the points it was fit to
        # -- so hold it to what the crowd in the window actually spans.
        crowd = yw if crowd is None else crowd
        modal_vals[k] = min(max(line[1], crowd.min()), crowd.max())

    anchor_x = x[anchor_idx]
    ux, uy = _collapse_ties(anchor_x, modal_vals)

    if ux.size >= 4:
        frac = float(np.clip(post_smooth_frac, 3.0 / ux.size, 1.0))
        smooth = lowess(uy, ux, frac=frac, it=0)[:, 1]
    else:
        smooth = uy

    fit = ModalLowessFit(ux, smooth)
    # Expose raw anchors for diagnostic plotting (pre-smoothing modal estimates)
    fit.anchor_x = anchor_x
    fit.anchor_y = modal_vals
    return fit


def _collapse_ties(x, y):
    """Average y over repeated x so the knots are strictly increasing."""
    ux, inv = np.unique(x, return_inverse=True)
    if ux.size == x.size:
        return x, y
    return ux, np.bincount(inv, weights=y) / np.bincount(inv)

    
    
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



# empirical_fit stepping thresholds. Loosened from (0.99, 0.05): the sweep stops
# as soon as ONE of these is met, so higher outside-ratio / lower posterior means
# it stops at a more LENIENT percentile and keeps more peptides for RT fine-tuning.
# Both are overridable per-run via env vars (JMOD_GAP_TOL_FRAC,
# JMOD_EF_TRAIN_SIGMA) so a sweep does not need a separate codebase copy.
GAP_TOL_FRAC = float(os.environ.get("JMOD_GAP_TOL_FRAC", 0.80))
                             # each inter-channel gap must be within this
                             # fraction of the run's median spacing
EF_TRAIN_SIGMA = float(os.environ.get("JMOD_EF_TRAIN_SIGMA", 8.0))
                       # DISABLED by default: rank order fixes the channel label and
                       # gap-consistency verifies the triplet is genuine, so
                       # distance from a BAD library RT is not evidence of a
                       # bad peptide -- it is exactly what fine-tuning exists
                       # to correct. An absolute residual bound here re-imposes
                       # the circularity the gap guard was built to avoid.         # loose residual backstop (wrong IDs, not
                             # mis-channelled ones)
# RT search window per channel = RT_SIGMA_MULT * sigma_k + RT_ELUTION_MULT * elution_sd.
# Sweepable so the sigma multiplier can be optimised without a codebase copy.
RT_SIGMA_MULT = float(os.environ.get("JMOD_RT_SIGMA_MULT", 4.0))
RT_ELUTION_MULT = float(os.environ.get("JMOD_RT_ELUTION_MULT", 0.0))
# When true, each time channel is searched with its OWN boundary instead of every
# channel being held to min(per_ch_boundary).
RT_PER_CHANNEL_TOL = os.environ.get("JMOD_RT_PER_CHANNEL_TOL", "1") not in ("0", "", "false")
EF_POSTERIOR_STOP = 0.95     # was 0.99
EF_OUTSIDE_STOP = 0.15       # was 0.05


def empirical_fit(output_df, results_folder=None, channel=None, min_ids=800):
    """
    Filter data by confidence then fit LOWESS to empirical RT

    Parameters
    ----------
    output_df : pd.DataFrame
        Dataframe of IDs from the preliminary search.
    results_folder : str, optional
        Where to write the per-percentile diagnostic plots.
    min_ids : int, optional
        Good-ID floor for the scribe-stepping stop criterion: the sweep stops once
        the number of within-tolerance IDs would drop below this, to avoid
        over-filtering. Default 800 (the non-timeplex whole-set value); the
        timeplex per-channel caller scales it down since each channel holds ~1/K of
        the IDs.
    channel : int, optional
        Timeplex channel index. When set, it is appended to the diagnostic plot
        filenames so the per-channel calls (one per timeplex channel) do not
        overwrite each other. ``None`` keeps the original non-timeplex filenames.

    Returns
    -------
    cor_filter : np.ndarray(bool)
        DESCRIPTION.
    emp_rt_spl : TYPE
        DESCRIPTION.

    """

    logger.info("")
    logger.info("Filtering IDs from initial search")

    sfx = "" if channel is None else f"_ch{channel}"

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
            feat=f"{feature_percentile}{sfx}",
            weights=weights,
            sigmas=sigmas,
            results_folder=results_folder
        )
        

        logger.info(
            f"Testing Percentile: {feature_percentile}, "
            f"Ratio: {outside_ratio:.4f}, #IDs: {cor_filter.sum()}, Partial Posterior: {partial_posterior:.4f}"
        )

        # Stopping criteria. The sweep starts LENIENT (p20) and tightens, so
        # stopping EARLIER keeps MORE data. These thresholds are now module-level
        # constants because on the timeplex path this stepping became the binding
        # constraint on fine-tune training size once the upstream scribe cut was
        # loosened: with no scribe pre-filter it kept only 48%/77%/38% per channel
        # (5,107 peptides) against a uniform ~77% (7,247) when the upstream filter
        # did the cleaning. Two filters doing the same job in series -- weakening
        # the first simply strengthened the second.
        if (partial_posterior >= EF_POSTERIOR_STOP or
                outside_ratio < EF_OUTSIDE_STOP or
                (cor_filter.sum() - bad_IDs.sum()) < min_ids):
            break

    logger.debug(f"{feature_percentile} {np.round(outside_ratio, 4)} {cor_filter.sum()}")

    # Keep the scribe-only selection separately. The LOWESS below still uses the
    # tight RT cut (a local fit needs outlier protection), but the CNN training
    # set does not want it: distance from a not-yet-corrected library RT is what
    # fine-tuning exists to fix, so cutting on it discards exactly the informative
    # peptides. The timeplex path applies a loose EF_TRAIN_SIGMA MAD guard to this
    # pool instead, and the non-timeplex path now does the same.
    scribe_filter = cor_filter.copy()
    cor_filter = np.logical_and(
        cor_filter,
        np.abs(first_rt_diffs) < first_rt_tolerance
    )

    # NOTE: weighted robust LOESS fit on the score-selected set was measured and
    # REVERTED. It fits its own input visibly better and removes modal lowess's
    # oscillation and terminal dive, but at 1% FDR it LOSES identifications
    # wherever the empirical model is the one chosen: JD0434 -239 own / -39 best,
    # JD0581_re -863 own / -1,319 best. The visual defect that prompted it was a
    # PLOTTING artifact (RTfit drew the fine-tuning training set against curves
    # fitted on empirical_fit's survivors), not a fitting one.
    emp_rt_spl = fast_modal_lowess(
        np.array(output_df.lib_rt)[cor_filter],
        np.array(output_df.rt)[cor_filter],
        .01,
        anchors=1000,
        grid_size=1000,
        post_smooth_frac=0.01
    )

    return cor_filter, emp_rt_spl, scribe_filter


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
                    offsets=None,
                    mz_source=None):
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
        

        # The RT panels above may be drawn on the fine-tuning TRAINING set, which
        # carries only lib_rt/rt/updated_lib_rt. The m/z panels need mz_diffs and
        # mz, so they draw from mz_source (the real filtered_output) instead.
        if mz_source is None:
            mz_source = filtered_output
        # `columns` via getattr: callers may pass any object exposing the fields
        # as attributes (the tests do), not only a DataFrame.
        _cols = getattr(mz_source, "columns", None)
        _needed = ("mz_diffs", "mz", "updated_lib_rt")
        _have = (set(_needed).issubset(_cols) if _cols is not None
                 else all(hasattr(mz_source, c) for c in _needed))
        if not _have:
            plt.close("all")
            return

        ##plot mz rt alignment
        plt.subplots()
        plt.scatter(mz_source.updated_lib_rt,np.array(mz_source.mz_diffs),label="Original_MZ",s=1,alpha=min(1,5/((len(np.array(mz_source.mz))//1000)+1)))
        plt.scatter(mz_source.updated_lib_rt,f_rt_mz(mz_source.updated_lib_rt),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("Updated RT")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/first_search/MZrtfit.png",dpi=600,bbox_inches="tight")
            plt.close()
        

        ##plot mz alignment
        plt.subplots()
        plt.scatter(np.array(mz_source.mz),(mz_source.mz_diffs-f_rt_mz(mz_source.updated_lib_rt)),label="Original_MZ",s=1,alpha=min(1,5/((len(np.array(mz_source.updated_lib_rt))//1000)+1)))
        plt.scatter(mz_source.mz,mz_spl(mz_source.mz),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("m/z")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        if results_folder is not None:
            plt.savefig(results_folder+"/first_search/MZfit.png",dpi=600,bbox_inches="tight")
            plt.close()
        
        
        
        ## plot mz diff
        plt.subplots()
        plt.hist(np.array(mz_source.mz_diffs),100,label="Theoretical m/z")
        # plt.hist(((np.array(id_mzs)+np.array(mz_source.mz_diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs,100,alpha=.5)
        # plt.hist(((np.array(id_mzs)+np.array(mz_source.mz_diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
        vals,bins,_ = plt.hist((mz_source.mz_diffs-mz_spl(mz_source.mz)-f_rt_mz(mz_source.updated_lib_rt)),100,alpha=.5,label="Updated m/z")
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
    # n_channels=1 also yields the MEDIAN INDIVIDUAL PEAK FWHM, the same
    # estimator the timeplex path uses per channel. The pooled `fwhm` beside it
    # comes from one gaussian fitted to every peptide's elution overlaid, which
    # measures ~2x wider than any real peak, so the two are not interchangeable
    # and only the median is used in the RT window.
    fwhm, elution_sd, output_df, _ew1 = elution_analysis.calculate_elution_width(
        output_df, n_channels=1)
    peak_fwhm = float(_ew1["fwhm"][0])
    logger.info("Mean elution width: FWHM {fwhm:.4f}, SD {elution_sd:.4f}; "
                "median peak FWHM {pf:.4f} (from {n} peaks)".format(
                    fwhm=fwhm, elution_sd=elution_sd, pf=peak_fwhm,
                    n=_ew1["n_peaks"][0]))

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


    cor_filter, emp_rt_spl, scribe_filter = empirical_fit(output_df, results_folder=results_folder)
        
    
    
    
    
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
        # Training pool = the SCRIBE-only selection, not cor_filter. cor_filter
        # additionally carries empirical_fit's tight 4-sigma RT cut, and training
        # the CNN on peptides pre-selected for already agreeing with the
        # uncorrected library RT is circular -- it removes the hard cases that
        # fine-tuning exists to learn. Mirrors the timeplex path.
        _train_filter = scribe_filter
        seq_rt = {}
        seq_lib = {}
        for s,rt,lrt in zip(np.array(id_keys)[_train_filter],
                            np.array(output_df.rt)[_train_filter],
                            np.array(output_df.lib_rt)[_train_filter]):
            key=librarySpectra[(s[0],float(s[1]))]["seq"]
            seq_rt.setdefault(key,[])
            seq_rt[key].append(rt)
            seq_lib.setdefault(key,[])
            seq_lib[key].append(lrt)
        # exclude those with ambiguity (differences between channels/charge states)
        filtered_seq_rt = {s:np.median(seq_rt[s]) for s in seq_rt if np.ptp(seq_rt[s])<1}

        # Loose EF_TRAIN_SIGMA MAD guard in place of the tight cut: keeps the
        # informative outliers but still removes the wrong IDs (residuals run to
        # tens of minutes), which measured better than either extreme on timeplex.
        if filtered_seq_rt:
            _seqs = list(filtered_seq_rt)
            _obs = np.array([filtered_seq_rt[s] for s in _seqs], dtype=float)
            _lib = np.array([np.median(seq_lib[s]) for s in _seqs], dtype=float)
            _res = _obs - emp_rt_spl(_lib)
            _sig = 1.4826 * np.median(np.abs(_res - np.median(_res)))
            if np.isfinite(_sig) and _sig > 0:
                _keep = np.abs(_res) <= EF_TRAIN_SIGMA * _sig
                logger.info(f"RT fine-tune training guard "
                            f"({EF_TRAIN_SIGMA:.0f} sigma = {EF_TRAIN_SIGMA*_sig:.2f} min): "
                            f"kept {int(_keep.sum()):,}/{len(_seqs):,}")
                filtered_seq_rt = {s: filtered_seq_rt[s]
                                   for s, k in zip(_seqs, _keep) if k}

        ## use observed rt for fine_tuning
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

        # Log the actual margin, not just the verdict: on the timeplex path the
        # equivalent numbers showed the fine-tune winning by only 1.4% (i.e. within
        # noise of the empirical fit), which was invisible until it was printed.
        logger.info(f"RT model comparison (no timeplex): empirical CDF-AUC="
                    f"{emp_cdf_auc:.4f} vs fine-tuned CDF-AUC={pred_cdf_auc:.4f} "
                    f"({100*(pred_cdf_auc-emp_cdf_auc)/max(emp_cdf_auc,1e-9):+.2f}%); "
                    f"held-out |dRT| median {np.median(np.abs(all_pred_diffs)):.4f} min "
                    f"vs empirical {np.median(np.abs(all_emp_diffs)):.4f} min "
                    f"({len(all_pred_diffs):,} held-out / {len(all_emp_diffs):,} empirical)")
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
            boundary = RT_SIGMA_MULT * sigmas[0] + RT_ELUTION_MULT * peak_fwhm
            logger.info(f"  narrow sigma {sigmas[0]:.4f}, wide {sigmas[1]:.4f}; "
                        f"RT boundary {boundary:.4f} (elution_sd {elution_sd:.4f})")
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
            boundary = RT_SIGMA_MULT * sigmas[0] + RT_ELUTION_MULT * peak_fwhm
            logger.info(f"  narrow sigma {sigmas[0]:.4f}, wide {sigmas[1]:.4f}; "
                        f"RT boundary {boundary:.4f} (elution_sd {elution_sd:.4f})")
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
        boundary = RT_SIGMA_MULT * sigmas[0] + RT_ELUTION_MULT * peak_fwhm
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


def _hwhm_mode_sigma(v, mode_smooth=3.0):
    """Fit-free mode + HWHM-derived sigma (sigma = HWHM / 1.1774) of a 1-D sample.

    Reads the width off the smoothed-histogram half-max crossings, so it ignores the
    tails / contamination (no curve fit, no truncation bias).

    The mode (peak location) is taken from a histogram smoothed at `mode_smooth` (default
    3.0, heavy) so a narrow contaminant spike cannot out-top the broad genuine hump
    (argmax rewards peak height, not mass); the WIDTH is then measured on a LIGHTLY
    smoothed histogram (sigma=1.5) around that location, so the mode smoothing does not
    inflate sigma and widen the gate. The two are deliberately decoupled. Pass a smaller
    `mode_smooth` (e.g. 1.5) when the sample is large and well-populated -- with many
    points the broad hump already dominates and no spurious spike can win, so heavy
    smoothing only blurs a genuine peak."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    hi = np.percentile(v, 99)
    h, e = np.histogram(v[v <= hi], bins=120)
    c = 0.5 * (e[:-1] + e[1:])
    hs_mode = gaussian_filter(h.astype(float), mode_smooth)  # locate peak (robust to spikes)
    hs_width = gaussian_filter(h.astype(float), 1.5)   # measure HWHM (tight, faithful)
    pk = int(np.argmax(hs_mode))
    mode = float(c[pk])
    half = hs_width[pk] / 2.0
    li = pk
    while li > 0 and hs_width[li] > half:
        li -= 1
    ri = pk
    while ri < len(hs_width) - 1 and hs_width[ri] > half:
        ri += 1
    sigma = max(min(mode - c[li], c[ri] - mode) / 1.1774, 1e-3)
    return mode, sigma


def timeplex_two_stage_gate(apex_df, n_timeplex, n_sigma_global=2.0,
                            min_trend_pts=200):
    """De-trended per-pair gate with triplet-toss.

    For each channel pair (a,b) the per-peptide gap |rt_b - rt_a| is gated
    (fit-independent) against the gap's own RT TREND rather than against a single
    global constant: a modal LOWESS gives the expected gap at each peptide's RT, and
    the window is mode +/- n_sigma_global*sigma of the RESIDUAL about that trend
    (width = fit-free HWHM). Then TRIPLET-TOSS: a peptide's apexes are kept only if
    ALL pairs pass; if any pair is bad the whole triplet is dropped (the timeplex
    channel assignment is no longer trustworthy).

    Why de-trended and not global: the channel gap is not constant across a gradient.
    On JD0412 the 1-2 gap drifts 3.92 -> 4.70 min early-to-late while the residual
    spread is only ~0.11 min, so a global mode +/- 2*sigma window ([3.64, 4.09]) is
    NARROWER than the physical drift. It then rejects every peptide eluting after
    ~45 min -- not as bad IDs but as "wrong gap" -- and the surviving alignment set
    truncates at lib_rt ~40. The per-channel LOWESS is then fit only over that stub,
    ModalLowessFit freezes past its last knot, and every library precursor above that
    iRT is mapped onto one RT: no scan after ~54 min ever gets a candidate. De-trending
    moves the window's CENTRE with the gradient and leaves its WIDTH set by the true
    residual spread, which is what the 2-sigma gate always assumed.

    This is the same assumption removal already applied to the per-channel fit itself
    (fit_timeplex_channels dropped the shared-shape + equal-spacing ridge); it simply
    had never been carried into the gate.

    Note this is NOT a return of the old RT-LOCAL second stage, which re-estimated
    SIGMA per RT bin and so collapsed the window where points were sparse. Here sigma
    stays global; only the centre is allowed to move.

    Parameters
    ----------
    apex_df : pandas.DataFrame
        Apex rows with 'stripped_seq', 'rt', 'channel' (from
        select_timeplex_alignment_set).
    n_timeplex : int
        Number of channels K.
    n_sigma_global : float
        Half-width of the keep window, in sigma of the de-trended residual.
    min_trend_pts : int
        Below this many paired peptides the gap trend is not fit and the pair falls
        back to the flat global window (a trend read off few points is noise).

    Returns
    -------
    np.ndarray[bool]
        Keep mask aligned to apex_df rows.
    """
    K = int(n_timeplex)
    df = apex_df.reset_index(drop=True)
    if K < 2:
        return np.ones(len(df), dtype=bool)
    # NOTE: do NOT dropna here. Dropping peptides absent from any channel made this
    # gate re-impose the exactly-K requirement that select_timeplex_alignment_set
    # already applies, so peptides recovered in 2-of-K were thrown straight back
    # out (measured: alignment set 1,947 -> 3,163 peptides changed the kept apex
    # count not at all, 4,539 both times). A peptide seen in channels 0 and 2 is
    # still testable against the 0-2 gap trend, and its apexes are still valid
    # per-channel training data for the channels it WAS seen in.
    wide = df.pivot_table(index="stripped_seq", columns="channel", values="rt",
                          aggfunc="first")
    seqs = wide.index.to_numpy()
    rt = {k: wide[k].to_numpy(dtype=float) for k in wide.columns}
    chans = sorted(rt)

    # A peptide is tossed only if a pair it ACTUALLY HAS fails. Pairs it does not
    # have are not evidence against it.
    keep_pep = np.ones(len(seqs), dtype=bool)
    n_testable = np.zeros(len(seqs), dtype=int)
    for a, b in combinations(chans, 2):
        both = np.isfinite(rt[a]) & np.isfinite(rt[b])
        if both.sum() < 2:
            continue
        gap_all = np.full(len(seqs), np.nan)
        gap_all[both] = np.abs(rt[b][both] - rt[a][both])
        gap = gap_all[both]

        # Flat global window, kept as the fallback and as the log comparison.
        # Fitted on the peptides that HAVE this pair (complete or not).
        gm, gs = _hwhm_mode_sigma(gap, mode_smooth=1.5)
        flat_pass = np.abs(gap - gm) <= n_sigma_global * gs
        rta_sub = rt[a][both]

        # Expected gap as a function of elution time. Modal LOWESS, so the
        # cross-channel contaminants this gate exists to remove sit off the trend
        # and do not drag it.
        if len(gap) >= min_trend_pts:
            trend = fast_modal_lowess(rta_sub, gap, local_frac=0.2,
                                      anchors=200, grid_size=256,
                                      post_smooth_frac=0.1)
            resid = gap - np.asarray(trend(rta_sub))
            rm, rs = _hwhm_mode_sigma(resid, mode_smooth=1.5)
            pair_pass = np.abs(resid - rm) <= n_sigma_global * rs
            drift = float(np.ptp(np.asarray(trend(np.percentile(rt[a], [1, 99])))))
            logger.info(f"gate pair {a}-{b}: gap trend drift {drift:.3f} min over the "
                        f"gradient, residual sigma {rs:.3f} (flat sigma {gs:.3f}); "
                        f"pass {pair_pass.mean():.3f} (flat would be {flat_pass.mean():.3f})")
            # A de-trended window should never keep LESS than the flat one it
            # replaces; if it does the trend fit misbehaved, so fall back.
            if pair_pass.sum() < flat_pass.sum():
                logger.warning(f"gate pair {a}-{b}: de-trended window kept fewer than "
                               f"flat; falling back to the flat global window")
                pair_pass = flat_pass
        else:
            pair_pass = flat_pass
            logger.info(f"gate pair {a}-{b}: only {len(gap)} paired peptides "
                        f"(< {min_trend_pts}), using flat global window; "
                        f"pass {pair_pass.mean():.3f}")

        # scatter this pair's verdict back to full-length, defaulting to PASS for
        # peptides that do not have the pair at all
        full_pass = np.ones(len(seqs), dtype=bool)
        full_pass[both] = pair_pass
        keep_pep &= full_pass
        n_testable[both] += 1

    # A peptide with no testable pair (seen in a single channel) cannot be checked
    # by this gate; keep it -- it is still training data for its own channel, and
    # the per-channel scribe/RT filters downstream still apply to it.
    n_untestable = int((n_testable == 0).sum())
    keep_lookup = {s: bool(keep_pep[i]) for i, s in enumerate(seqs)}
    mask = np.array([keep_lookup.get(s, False) for s in df["stripped_seq"]], dtype=bool)
    logger.info(f"  gate: {int((n_testable > 0).sum()):,} peptides had a testable "
                f"pair, {n_untestable:,} single-channel kept untested")
    logger.info(f"triplet-toss gate: kept {int(mask.sum())}/{len(mask)} apexes "
                f"({mask.mean():.3f})")

    # The gate must not become an RT filter: if it strips one end of the gradient the
    # per-channel fit is left with a stub, ModalLowessFit freezes past its last knot,
    # and the search silently loses every precursor beyond it. Cheap to check, and it
    # is the symptom that is otherwise invisible until the ID count comes back.
    if mask.any() and "lib_rt" in df.columns:
        lo_all, hi_all = np.percentile(df["lib_rt"], [1, 99])
        lo_k, hi_k = np.percentile(df["lib_rt"][mask], [1, 99])
        span = hi_all - lo_all
        if span > 0 and (hi_all - hi_k) > 0.1 * span:
            logger.warning(
                f"Timeplex gate kept lib_rt up to {hi_k:.1f} but the apex set reaches "
                f"{hi_all:.1f} ({(hi_all - hi_k) / span:.0%} of the range dropped off the "
                f"top): the per-channel RT curve will be extrapolated there and late "
                f"precursors may not be searched at all.")
        if span > 0 and (lo_k - lo_all) > 0.1 * span:
            logger.warning(
                f"Timeplex gate kept lib_rt down to {lo_k:.1f} but the apex set reaches "
                f"{lo_all:.1f} ({(lo_k - lo_all) / span:.0%} of the range dropped off the "
                f"bottom): early precursors may not be searched at all.")
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


def fit_timeplex_channels(output_df, n_timeplex, results_folder=None):
    """
    Align timeplex channels by fitting each channel's iRT->RT curve INDEPENDENTLY,
    one channel at a time, mirroring the per-channel structure of the fine-tune path.

    The apexes arrive already channel-assigned (by RT rank, upstream in
    select_timeplex_alignment_set) and cross-channel gated (timeplex_two_stage_gate).
    For each channel we then run the SAME iterative scribe-score stepping the
    non-timeplex path uses (empirical_fit): step the scribe cutoff up until the
    per-channel RT residuals are clean, dropping the low-quality tail, and fit a
    modal LOWESS to the survivors. This drops the old ridge-tracker's two strong
    assumptions (one shared shape + equal channel spacing); each channel gets its
    own empirical shape and its own quality cut.

    Parameters
    ----------
    output_df : pandas.DataFrame
        Gated, channel-assigned first-search apexes with columns: scribe_score,
        lib_rt, rt, mz, relative_error_ms1, stripped_seq, channel.
    n_timeplex : int
        Number of channels K (>= 2).

    Returns
    -------
    rt_spls : list[callable]
        K callables mapping library iRT -> observed RT (one independent fit each).
    shared_shape : callable
        A representative de-offset iRT->RT curve (mean of the K curves after
        removing each channel's level). Used only for the alignment plot x-shape.
    offsets : np.ndarray
        Length-K per-channel RT level (median of each channel's curve).
    d : float
        Common channel gap (median adjacent difference of the sorted offsets);
        used downstream only for the d/2 RT-tolerance overlap clamp.
    filtered_output : pandas.DataFrame
        The per-channel survivors (post scribe-stepping) with lib_rt, rt, channel,
        stripped_seq, mz_diffs, mz, scribe_score.
    residuals : np.ndarray
        Signed residual rt - f_k(lib_rt) of each surviving PSM, aligned to
        filtered_output rows.
    """
    K = int(n_timeplex)
    if K < 2:
        raise RuntimeError("fit_timeplex_channels requires n_timeplex >= 2")

    kept = output_df.reset_index(drop=True)
    channel = kept["channel"].to_numpy(dtype=int)
    logger.info(f"Per-channel timeplex fit: {len(kept)} gated IDs across {K} channels")

    # --- fit each channel independently, one at a time, with its own scribe stepping ---
    rt_spls = [None] * K
    keep_frames = []                 # surviving rows per channel (for filtered_output)
    resid_parts = []                 # per-channel residuals, aligned to keep_frames
    for k in range(K):
        in_channel = channel == k    # boolean mask: apex rows assigned to channel k
        ch_df = kept.loc[in_channel, ["lib_rt", "rt", "scribe_score",
                                      "stripped_seq", "mz",
                                      "relative_error_ms1"]].reset_index(drop=True)
        if len(ch_df) < 10:
            raise RuntimeError(f"Per-channel timeplex fit: channel {k} has too few "
                               f"IDs ({len(ch_df)})")
        # iterative scribe-percentile stepping + final residual cut (per channel).
        # Scale the good-ID stop floor down: each channel holds ~1/K of the IDs, so
        # inheriting the whole-set 800 floor would break the sweep at the most
        # lenient cutoff (or immediately) for normal channel sizes.
        cor_filter, f_k, _scribe_k = empirical_fit(ch_df, results_folder=results_folder,
                                        channel=k, min_ids=max(100, 800 // K))
        rt_spls[k] = f_k

        surv = ch_df.loc[cor_filter].copy()
        surv["channel"] = k
        resid = surv["rt"].to_numpy(dtype=float) - np.asarray(f_k(surv["lib_rt"].to_numpy(dtype=float)))
        keep_frames.append(surv)
        resid_parts.append(resid)
        logger.info(f"  channel {k}: kept {int(cor_filter.sum())}/{len(ch_df)} apexes, "
                    f"median |dRT| = {np.median(np.abs(resid)):.3f}")

    # --- assemble survivors into the shape downstream expects ---
    filtered_output = pd.concat(keep_frames, ignore_index=True).rename(
        columns={"relative_error_ms1": "mz_diffs"})
    residuals = np.concatenate(resid_parts)

    # --- summary values derived from the fitted curves (plot + d/2 clamp only) ---
    x = kept["lib_rt"].to_numpy(dtype=float)
    grid = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 400)
    curves = np.vstack([np.asarray(rt_spls[k](grid)) for k in range(K)])  # (K, grid)
    offsets = np.median(curves, axis=1)                  # per-channel RT level
    d = float(np.median(np.diff(np.sort(offsets))))      # common channel gap
    shape = (curves - offsets[:, None]).mean(axis=0)     # de-offset mean shape (plot only)
    shared_shape = interp1d(grid, shape, bounds_error=False,
                            fill_value=(shape[0], shape[-1]))
    logger.info(f"Per-channel timeplex fit: offsets={np.round(offsets, 4)}, "
                f"channel gap d={d:.4f}")

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
    # Per-time-channel elution width as well as the pooled value. The pooled one
    # keeps only each peptide's LARGEST cluster, so with timePlex it discards two
    # of every three elutions and is biased toward whichever channel happened to
    # give the fattest peak; the per-channel fit uses all K, ranked by RT.
    _n_tp = config.args.num_timeplex
    _ew = calculate_elution_width(output_df, n_channels=_n_tp)
    _fwhm, elution_sd, output_df, _per_ch_ew = _ew
    elution_fwhm_by_ch = _per_ch_ew["fwhm"]
    logger.info(f"Mean elution width: FWHM {_fwhm:.4f}, SD {elution_sd:.4f}; "
                f"per-channel median peak FWHM "
                f"{[round(float(s), 4) for s in elution_fwhm_by_ch]} "
                f"(from {_per_ch_ew['n_peaks']} peaks); per-channel SD "
                f"{[round(float(s), 4) for s in _per_ch_ew['sigma']]}")


    # Build the RT-alignment input set (vectorized): sweep scribe cutoffs and pick
    # the one that maximizes peptides seen in exactly K=n_timeplex RT clusters, take
    # each cluster's apex, and assign channel 0..K-1 by RT rank. Elutions are
    # seconds-wide and timeplex windows minutes apart, so an 8*elution_sd RT gap
    # separates the windows.
    n_timeplex = config.args.num_timeplex
    rt_gap = 8 * elution_sd
    # Diagnostic: how big is the RAW first search before any timeplex selection?
    # Needed to tell whether the small alignment set is a SEARCH shortfall or a
    # CLUSTERING/completeness shortfall -- the old feature-seeded search yielded
    # 32,419 unique peptides where the alignment set here is ~4,141.
    try:
        _raw_pep = output_df['stripped_seq'].n_unique() if hasattr(output_df, 'n_unique') \
            else output_df['stripped_seq'].nunique()
        logger.info(f'RAW first search before timeplex selection: {len(output_df):,} rows, '
                    f'{_raw_pep:,} unique peptides')
    except Exception as _e:
        logger.warning(f'could not size raw first search: {_e}')
    try:
        _o = output_df.to_pandas() if hasattr(output_df,'to_pandas') else output_df
        if 'rank' in _o.columns:
            _cum=set()
            for _r in sorted(_o['rank'].unique()):
                _sub=_o[_o['rank']==_r]
                _cum |= set(_sub['stripped_seq'])
                logger.info(f'  RAW rank {_r}: {len(_sub):,} rows, '
                            f"{_sub['stripped_seq'].nunique():,} unique; cumulative {len(_cum):,}")
    except Exception as _e:
        logger.warning(f'rank diag failed: {_e}')
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

    # --- Phase 2: fit-independent cross-channel gate ---------------------------
    # Clean the apex set with the per-pair gate (raw |dRT| HWHM global window,
    # triplet-toss). This uses NO RT fit, so the empirical and fine-tuned models
    # are fit and compared on the same gated data.
    gate_mask = timeplex_two_stage_gate(output_df, n_timeplex)
    gated_df = output_df[gate_mask].reset_index(drop=True)


    # Per-channel empirical fit on the GATED set: each channel is fit independently
    # with its own iterative scribe-score stepping (empirical_fit). Also the source
    # of the per-channel offsets, the channel gap d, and the alignment-plot shape.
    rt_spls, emp_shape, offsets, d, filtered_output, residuals = \
        fit_timeplex_channels(gated_df, n_timeplex, results_folder)

    channel = filtered_output["channel"].to_numpy()
    filtered_output["updated_lib_rt"] = filtered_output["lib_rt"]   # x-axis = library iRT

    # rt_spls from fit_timeplex_channels are per-channel iRT->RT curves (empirical default).
    emp_curves = rt_spls
    plot_curves = emp_curves                      # what alignment_plots draws
    emp_data, emp_p, emp_cdf_auc = cdf_data(residuals)   # in-sample; replaced
    # below with held-out once the per-channel refit exists, so RTelbows.png
    # compares like with like.
    chosen_diffs = residuals
    chosen_channel = channel                      # channel labels aligned to chosen_diffs
    pred_data = pred_p = None
    mixture_feat = "_timeplex_empirical"

    # Needed by the collision clamp below, which runs whether or not fine-tuning
    # does -- so it cannot live inside the fine-tune branch.
    seqs_all = filtered_output["stripped_seq"].to_numpy()
    obs_rt = filtered_output["rt"].to_numpy(dtype=float)

    if not config.args.use_emp_rt:
        logger.info("Per-channel RT prediction (timeplex)")
        # --- Phase 3: bagged, OOB, (1-PEP) soft-weighted fine-tune per channel ----
        # Each channel is fine-tuned independently on its gated apexes with a 5-member
        # bagged ensemble; held-out OOB residuals make the empirical-vs-fine-tuned
        # comparison fair. The CNN predicts the channel's observed RT directly, so the
        # per-channel {seq: rt} dict is already on the observed scale run_jmod consumes.

        # --- fine-tune TRAINING set: observation, not agreement ----------------
        # The set that reaches the CNN has passed three successive RT-residual
        # filters (triplet-toss window, scribe stepping, and empirical_fit's
        # |first_rt_diffs| < first_rt_tolerance). Every one of them keeps peptides
        # that ALREADY sit close to the current RT model -- so the CNN is trained
        # only on peptides it would have predicted well anyway and can never learn
        # to correct the errors that actually cost IDs. It also puts hard edges on
        # the residual distribution, visible as sharp cut-offs in the diagnostics.
        #
        # Require only that a peptide was OBSERVED in all K channels: that is real
        # evidence it exists, and it is the one criterion that does not presuppose
        # the answer. The empirical curve is still fitted on the filtered subset
        # (a LOWESS fit does need outlier protection), so only the CNN's training
        # set is widened.
        ft_seqs, ft_rt, ft_lib = {}, {}, {}
        try:
            _al = output_df  # the alignment set (apex per peptide/channel)
            _cnt = _al.groupby("stripped_seq")["channel"].nunique()
            _complete = set(_cnt[_cnt >= n_timeplex].index)
            _use = _al[_al["stripped_seq"].isin(_complete)]
            for k in range(n_timeplex):
                _sub = _use[_use["channel"] == k]
                ft_seqs[k] = _sub["stripped_seq"].to_numpy()
                ft_rt[k] = _sub["rt"].to_numpy(dtype=float)
                ft_lib[k] = _sub["lib_rt"].to_numpy(dtype=float)
            # GAP-CONSISTENCY GUARD (replaces the residual-based one).
            #
            # "Exactly K clusters" does NOT mean "one cluster per channel": a
            # peptide with a split peak can contribute two clusters to one
            # channel and none to another, still total K, and then rank-order
            # channel assignment mislabels all K. Measured on JD0319: 7.7% of
            # complete peptides have a consecutive gap under 40% of the median
            # channel spacing (the split-peak signature) and 22.8% deviate more
            # than 50%.
            #
            # This tests the timeplex STRUCTURE rather than agreement with the RT
            # model: a peptide seen once per channel must show inter-cluster gaps
            # matching the run's channel spacing. It therefore keeps peptides that
            # are far from the library RT but correctly assigned -- which the
            # residual cut removed and which are exactly the informative ones.
            try:
                _piv = _use.pivot_table(index="stripped_seq", columns="channel",
                                        values="rt", aggfunc="first").dropna(how="any")
                if len(_piv) >= 200 and all(c in _piv.columns for c in range(n_timeplex)):
                    _gaps = np.column_stack([_piv[c + 1].to_numpy() - _piv[c].to_numpy()
                                             for c in range(n_timeplex - 1)])
                    _med = np.median(_gaps, axis=0)
                    # scale-free: each gap within GAP_TOL_FRAC of its median
                    _rel = np.abs(_gaps - _med) / np.maximum(_med, 1e-6)
                    _good = np.all(_rel <= GAP_TOL_FRAC, axis=1)
                    _keep_seq = set(np.asarray(_piv.index)[_good])
                    logger.info(f"  gap-consistency guard (each gap within "
                                f"{100*GAP_TOL_FRAC:.0f}% of median spacing "
                                f"{np.round(_med,2).tolist()}): kept "
                                f"{int(_good.sum()):,}/{len(_piv):,} peptides")
                    for k in range(n_timeplex):
                        _m = np.array([q in _keep_seq for q in ft_seqs[k]])
                        ft_seqs[k] = ft_seqs[k][_m]
                        ft_rt[k] = ft_rt[k][_m]
                        ft_lib[k] = ft_lib[k][_m]
            except Exception as _e:
                logger.warning(f"gap-consistency guard failed ({_e}); "
                               f"training set left unfiltered")

            # Residual-based guard, retained as a backstop for peptides whose
            # gaps look fine but whose RT is physically impossible (wrong ID).
            # Deliberately loose. Requiring only "observed in all K
            # channels" cost 4.3% of IDs (57,436 vs 60,035): the unfiltered
            # alignment set has residuals out to 56 min, and a peptide teaching
            # the CNN an RT that is 20+ min wrong does more damage than a
            # thousand correct ones repair. But the shipped 4-sigma cut
            # (empirical_fit line 720) is too tight -- it keeps only peptides
            # that ALREADY fit the model. 8 sigma admits the genuinely hard
            # cases while still excluding the physically impossible ones.
            for k in range(n_timeplex):
                _l, _r = ft_lib[k], ft_rt[k]
                _ok = np.isfinite(_l) & np.isfinite(_r)
                if _ok.sum() >= 200:
                    _f0 = fast_modal_lowess(_l[_ok], _r[_ok], .05, anchors=500,
                                            grid_size=500, post_smooth_frac=0.05)
                    _res = _r - np.asarray(_f0(_l))
                    _sig = 1.4826 * np.nanmedian(np.abs(_res - np.nanmedian(_res)))
                    _keep = np.isfinite(_res) & (np.abs(_res) <= EF_TRAIN_SIGMA * _sig)
                    logger.info(f"  T{k} training outlier guard "
                                f"({EF_TRAIN_SIGMA:.0f} sigma = "
                                f"{EF_TRAIN_SIGMA*_sig:.2f} min): kept "
                                f"{int(_keep.sum()):,}/{len(_keep):,}")
                    ft_seqs[k] = ft_seqs[k][_keep]
                    ft_rt[k] = _r[_keep]
                    ft_lib[k] = _l[_keep]
            logger.info("Fine-tune training set (observed in all "
                        f"{n_timeplex} channels, NO RT-residual filtering): "
                        + ", ".join(f"T{k}={len(ft_rt[k]):,}"
                                    for k in range(n_timeplex))
                        + "  [filtered path would give "
                        + ", ".join(f"T{k}={int((channel==k).sum()):,}"
                                    for k in range(n_timeplex)) + "]")
        except Exception as _e:
            logger.warning(f"observation-only training set failed ({_e}); "
                           f"falling back to the filtered set")
            for k in range(n_timeplex):
                ft_seqs[k] = seqs_all[channel == k]
                ft_rt[k] = obs_rt[channel == k]
                ft_lib[k] = filtered_output.loc[channel == k, "lib_rt"].to_numpy(dtype=float)
        uniq = sorted({updatedLibrary[k]["seq"] for k in librarySpectra})
        model_path = rt_model_path()

        # --- per-channel fine-tune, ORIGINAL fine_tune_rt (not the bagged path) ---
        # Each time channel is fitted SEPARATELY with the same fine_tune_rt used on
        # the no-timeplex path, and each channel independently decides empirical vs
        # fine-tuned on its own held-out residuals. The old code fitted only channel
        # 0 and mapped the rest on; the bagged replacement fitted all three but with
        # a different estimator. This does the plain per-channel fit.
        lib_pred_by_ch, val_diffs_by_ch, ft_hist = [], [], []
        emp_paired = {}
        for k in range(n_timeplex):
            # one RT per sequence for this channel (median over duplicate charges)
            srt, slib = {}, {}
            for s_, r_, l_ in zip(ft_seqs[k], ft_rt[k], ft_lib[k]):
                srt.setdefault(s_, []).append(r_)
                slib.setdefault(s_, []).append(l_)
            _seqs = list(srt)
            gdf = pd.DataFrame({'Stripped.Sequence': _seqs,
                                'RT': [float(np.median(srt[q])) for q in _seqs]})
            _libmed = np.array([float(np.median(slib[q])) for q in _seqs])
            # Recover EXACTLY which peptides the CNN holds out. create_model_data
            # builds X in grouped_df order and splits with
            # train_test_split(test_size=0.1, random_state=123), so replicating
            # that split on the index gives the same partition. Fitting the
            # empirical curve on the same 90% and scoring it on the same 10%
            # makes the two models comparable on IDENTICAL peptides -- previously
            # each was scored on its own separate draw from a different
            # population, so the comparison was unpaired.
            try:
                from sklearn.model_selection import train_test_split as _tts2
                _itr, _ite = _tts2(np.arange(len(gdf)), test_size=0.1,
                                   random_state=123)
                _ltr, _lte = _libmed[_itr], _libmed[_ite]
                _rtr = gdf['RT'].to_numpy()[_itr]; _rte = gdf['RT'].to_numpy()[_ite]
                _ok = np.isfinite(_ltr) & np.isfinite(_rtr)
                if _ok.sum() >= 50:
                    _f = fast_modal_lowess(_ltr[_ok], _rtr[_ok], .01, anchors=1000,
                                           grid_size=1000, post_smooth_frac=0.01)
                    _oke = np.isfinite(_lte) & np.isfinite(_rte)
                    emp_paired[k] = _rte[_oke] - np.asarray(_f(_lte[_oke]))
            except Exception as _e:
                logger.warning(f"  channel {k}: paired empirical hold-out failed ({_e})")
            data_split, models, convertor = fine_tune_rt(
                gdf, qc_plots=True, results_path=results_folder, tag=config.tag)
            # held-out validation residuals for THIS channel -> fair vs empirical
            val_pred = convertor(np.mean([m.predict(np.array(data_split[1]))
                                          for m in models], axis=0).flatten())
            val_diffs_by_ch.append(np.asarray(data_split[3]) - val_pred)
            # library predictions for this channel, on the observed scale
            lib_oh = [one_hot_encode_sequence(q) for q in uniq]
            lp = convertor(np.mean([m.predict(np.array(lib_oh))
                                    for m in models], axis=0).flatten())
            lib_pred_by_ch.append(dict(zip(uniq, lp)))
            ft_hist.append({"oob_medae": [], "best_epoch": 0})
            logger.info(f"  timeplex channel {k}: fine-tune held-out |dRT| median = "
                        f"{np.median(np.abs(val_diffs_by_ch[k])):.3f} min "
                        f"({len(gdf):,} peptides)")

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

        # --- PER-CHANNEL empirical vs fine-tuned decision -----------------------
        # Each time channel picks its own model on its OWN held-out residuals, so a
        # channel whose fine-tune failed keeps the empirical fit instead of being
        # dragged along by a pooled vote.
        rt_spls = list(rt_spls)                 # empirical callables by default
        # `chosen_diffs` stay IN-SAMPLE (plots/mixture, consistent with the
        # empirical branch). `sigma_diffs` are HELD-OUT for fine-tuned channels,
        # because the sigma feeds the RT tolerance: an in-sample sigma is
        # overfit-optimistic (the CNN memorised its training peptides) and would
        # set a window too tight for library peptides it has never seen.
        chosen_diffs_l, chosen_channel_l, per_ch_choice = [], [], []
        sigma_diffs_l, sigma_channel_l = [], []
        # HELD-OUT empirical residuals, so the comparison is like-for-like.
        # The fine-tune's residuals are validation-only (peptides the CNN never
        # trained on) while `residuals` are the empirical fit's own IN-SAMPLE
        # residuals. Comparing those penalised the fine-tune by exactly its
        # generalisation gap -- in the CDF plot AND in this selection. Refit the
        # empirical curve per channel on the same 90/10 split the CNN uses
        # (test_size=0.1, random_state=123 in finetune_funs) and score it on the
        # held-out tenth.
        emp_holdout = {}
        for k in range(n_timeplex):
            selk = channel == k
            lk = filtered_output.loc[selk, "lib_rt"].to_numpy(dtype=float)
            rk = obs_rt[selk]
            good = np.isfinite(lk) & np.isfinite(rk)
            lk, rk = lk[good], rk[good]
            if len(lk) < 50:
                emp_holdout[k] = residuals[selk]        # too few to split
                continue
            try:
                from sklearn.model_selection import train_test_split as _tts
                li, lo, ri, ro = _tts(lk, rk, test_size=0.1, random_state=123)
                f_tr = fast_modal_lowess(li, ri, .01, anchors=1000,
                                         grid_size=1000, post_smooth_frac=0.01)
                emp_holdout[k] = ro - np.asarray(f_tr(lo))
            except Exception as _e:
                logger.warning(f"  channel {k}: held-out empirical refit failed "
                               f"({_e}); falling back to in-sample residuals")
                emp_holdout[k] = residuals[selk]

        for k in range(n_timeplex):
            emp_k = emp_paired.get(k, emp_holdout[k])
            ft_k = val_diffs_by_ch[k]
            _, _, emp_auc_k = cdf_data(emp_k)
            _, _, ft_auc_k = cdf_data(ft_k)
            if ft_auc_k > emp_auc_k:
                per_ch_choice.append("fine-tuned")
                rt_spls[k] = lib_pred_by_ch[k]          # {seq: RT_k}
                ftp = np.array([lib_pred_by_ch[k][q] for q in seqs_all[channel == k]],
                               dtype=float)
                dk = obs_rt[channel == k] - ftp         # in-sample, as before
            else:
                per_ch_choice.append("empirical")
                dk = emp_k
            chosen_diffs_l.append(dk)
            chosen_channel_l.append(np.full(len(dk), k, dtype=int))
            sk_diffs = ft_k if per_ch_choice[k] == "fine-tuned" else emp_k
            sigma_diffs_l.append(sk_diffs)
            sigma_channel_l.append(np.full(len(sk_diffs), k, dtype=int))
            logger.info(f"  channel {k}: {per_ch_choice[k]} chosen "
                        f"[empirical CDF-AUC={emp_auc_k:.4f} vs "
                        f"fine-tuned CDF-AUC={ft_auc_k:.4f}]")
        chosen_diffs = np.concatenate(chosen_diffs_l)
        chosen_channel = np.concatenate(chosen_channel_l)
        sigma_diffs = np.concatenate(sigma_diffs_l)
        sigma_channel = np.concatenate(sigma_channel_l)
        mixture_feat = ("_timeplex_fine_tuned"
                        if any(c == "fine-tuned" for c in per_ch_choice)
                        else "_timeplex_empirical")
        pred_data, pred_p, _ = cdf_data(np.concatenate(val_diffs_by_ch))
        # redraw the empirical CDF on held-out residuals too (RTelbows.png)
        emp_data, emp_p, emp_cdf_auc = cdf_data(
            np.concatenate([emp_holdout[k] for k in range(n_timeplex)]))
        logger.info("Per-channel RT model: " + ", ".join(
            f"T{k}={per_ch_choice[k]}" for k in range(n_timeplex)))

    else:
        logger.info("RT model: empirical ridge alignment (timeplex; --use_emp_rt)")

    # --- Phase 4: RT tolerance from the per-channel narrow GMM component -------
    # Fit a 2-component zero-mean GMM per channel on the chosen residuals, take the
    # narrow sigma, and average across channels.
    # Each channel gets its OWN tolerated residual from its OWN narrow sigma, on the
    # same 4-sigma criterion as the no-timeplex path (4*sigma + 8*elution_sd). A
    # channel that chose empirical and one that chose fine-tuned no longer share an
    # averaged sigma.
    # --use_emp_rt skips the fine-tune branch entirely, so fall back to the
    # empirical residuals (already held-out-equivalent: a ridge fit, not a CNN
    # that memorised its training peptides).
    if 'sigma_diffs' not in locals():
        sigma_diffs, sigma_channel = chosen_diffs, chosen_channel

    narrow_sigmas, per_ch_boundary = [], []
    for k in range(n_timeplex):
        rk = sigma_diffs[sigma_channel == k]
        if len(rk) < 10:
            rk = sigma_diffs
        _, sg = fit_zero_mean_gmm_1d(rk, n_components=2)
        sk = float(np.sort(sg)[0])
        narrow_sigmas.append(sk)
        # this channel's own median peak FWHM, not the pooled largest-cluster SD
        _ew_k = (elution_fwhm_by_ch[k] if (elution_fwhm_by_ch is not None
                                           and k < len(elution_fwhm_by_ch))
                 else 2.355 * elution_sd)
        per_ch_boundary.append(RT_SIGMA_MULT * sk + RT_ELUTION_MULT * _ew_k)

    # COLLISION CLAMP. Adjacent time channels sit a gap `d` apart. If two channels'
    # +/- windows overlap, a peptide eluting in one can be admitted as the other and
    # the channels stop being separable. Where a pair collides, BOTH are pulled down
    # to the largest common half-window that keeps them apart (d/2) -- the smaller
    # sigma governs, and neither window is ever widened.
    ch_med = [float(np.median(obs_rt[channel == k])) if (channel == k).any() else np.nan
              for k in range(n_timeplex)]
    gaps = []
    for i in range(n_timeplex - 1):
        g = abs(ch_med[i + 1] - ch_med[i])
        if not np.isfinite(g) or g <= 0:
            try:
                g = abs(float(np.atleast_1d(d)[0]))
            except Exception:
                g = np.inf
        gaps.append(g)
    for i, g in enumerate(gaps):
        if per_ch_boundary[i] + per_ch_boundary[i + 1] > g:
            cap = 0.5 * g
            o0, o1 = per_ch_boundary[i], per_ch_boundary[i + 1]
            per_ch_boundary[i] = min(o0, cap)
            per_ch_boundary[i + 1] = min(o1, cap)
            logger.info(f"  channels {i}/{i+1} windows collide over gap {g:.3f}: "
                        f"{o0:.3f}/{o1:.3f} -> {per_ch_boundary[i]:.3f}/"
                        f"{per_ch_boundary[i+1]:.3f}")
    sigma_narrow = float(np.mean(narrow_sigmas))
    # One scalar is consumed downstream, so the TIGHTEST collision-safe channel
    # governs; no channel is ever searched wider than its own safe window.
    boundary = float(min(per_ch_boundary))
    # Publish both for downstream use: the per-channel windows (so the search can
    # use each channel's own instead of the tightest) and the per-channel narrow
    # sigmas (so RT error can be z-scored against the channel it came from rather
    # than a pooled spread the channels do not actually share).
    config.rt_per_ch_boundary = [float(b) for b in per_ch_boundary]
    config.rt_narrow_sigmas = [float(s) for s in narrow_sigmas]
    # Also persist: scoring may run in a spawned process that does not inherit
    # module-level state, in which case the config attributes above are absent and
    # the z-score feature would silently vanish rather than fail loudly.
    if results_folder is not None:
        try:
            import json
            with open(os.path.join(results_folder, "first_search",
                                   "rt_channel_sigmas.json"), "w") as _fh:
                json.dump({"narrow_sigmas": config.rt_narrow_sigmas,
                           "per_ch_boundary": config.rt_per_ch_boundary}, _fh)
        except Exception as _e:
            logger.warning(f"could not persist per-channel RT sigmas: {_e}")
    logger.info(f"Per-channel narrow sigma {[round(s, 3) for s in narrow_sigmas]}; "
                f"per-channel RT boundary {[round(b, 3) for b in per_ch_boundary]}; "
                f"applied {boundary:.3f} "
                f"(sigma mult {RT_SIGMA_MULT}, elution mult {RT_ELUTION_MULT}, "
                f"per-channel tol {'ON' if RT_PER_CHANNEL_TOL else 'OFF'})")

    # --- per-channel RT-error diagnostic ------------------------------------
    # One histogram per channel overlaying the two candidate RT models on a FAIR
    # footing -- both held out, so neither is flattered by residuals on peptides
    # it was fitted to:
    #   * fine-tuned CNN, out-of-bag (its validation split)
    #   * original-library empirical fit, held out (refit on 90%, scored on 10%)
    # Medians are annotated for both, plus the recommended RT tolerance for that
    # channel AFTER the collision clamp -- i.e. the window the search would use.
    if results_folder is not None:
        try:
            _oob = locals().get("val_diffs_by_ch", None)
            _empp = locals().get("emp_paired", None) or {}
            _emph = locals().get("emp_holdout", None)
            # prefer the PAIRED hold-out (same peptides the CNN was scored on)
            _emp = ({k: _empp.get(k, (_emph or {}).get(k))
                     for k in range(n_timeplex)} if (_empp or _emph) else None)
            os.makedirs(results_folder + "/first_search", exist_ok=True)
            for k in range(n_timeplex):
                ft = np.asarray(_oob[k], dtype=float) if _oob is not None else None
                em = (np.asarray(_emp[k], dtype=float) if _emp is not None
                      else np.asarray(residuals[chosen_channel == k], dtype=float))
                em = em[np.isfinite(em)]
                lim = float(np.nanpercentile(np.abs(em), 99)) if em.size else 1.0
                if ft is not None and ft.size:
                    ft = ft[np.isfinite(ft)]
                    lim = max(lim, float(np.nanpercentile(np.abs(ft), 99)))
                lim = max(lim, per_ch_boundary[k] * 1.1, 0.5)
                bins = np.linspace(-lim, lim, 80)

                figh, axh = plt.subplots(figsize=(7, 4.5))
                if em.size:
                    axh.hist(em, bins=bins, alpha=0.55, label=
                             f"original library (held-out)  median |dRT|="
                             f"{np.median(np.abs(em)):.3f}", color="tab:orange")
                if ft is not None and ft.size:
                    axh.hist(ft, bins=bins, alpha=0.55, label=
                             f"fine-tuned (out-of-bag)  median |dRT|="
                             f"{np.median(np.abs(ft)):.3f}", color="tab:blue")
                axh.axvline(per_ch_boundary[k], color="r", ls="--", lw=1.2,
                            label=f"recommended RT tol = {per_ch_boundary[k]:.3f} min")
                axh.axvline(-per_ch_boundary[k], color="r", ls="--", lw=1.2)
                axh.set_xlabel("RT error (observed - predicted, min)")
                axh.set_ylabel("count")
                axh.set_title(f"Channel {k}: RT error, both models held out "
                              f"(narrow sigma {narrow_sigmas[k]:.3f})")
                axh.legend(fontsize=8)
                figh.savefig(results_folder +
                             f"/first_search/rt_error_hist_ch{k}.png",
                             dpi=200, bbox_inches="tight")
                plt.close(figh)
                _fm = (f"{np.median(np.abs(ft)):.3f}" if ft is not None and ft.size
                       else "n/a")
                logger.info(f"  channel {k} RT error: original-library(held-out) "
                            f"median {np.median(np.abs(em)):.3f}, fine-tuned(OOB) "
                            f"median {_fm}, recommended tol "
                            f"{per_ch_boundary[k]:.3f}")
        except Exception as e:
            logger.warning(f"Could not write per-channel RT error histograms: {e}")

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
    # f_rt_mz was fit against `rts` (updated LIBRARY RT), so the RT term must be
    # removed at `rts` too. Evaluating it at output_df.rt (OBSERVED RT, a
    # different scale entirely) reads the wiggly modal fit at the wrong
    # abscissa; because m/z and RT are correlated through hydrophobicity, that
    # spurious RT term re-emerges as a fake m/z-dependent mass error and leaves
    # the corrected residual off-zero. The non-timeplex path already uses one
    # RT variable throughout.
    mz_spl = fast_modal_lowess(np.array(output_df.mz)[rt_mz_filter_bool],(diffs-f_rt_mz(rts))[rt_mz_filter_bool])
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
        # Plot the set the CNN was actually TRAINED ON, unfiltered.
        #   OriginalRTfit.png : observed RT vs ORIGINAL library RT (before training)
        #   RTfit.png         : the SAME peptides vs the FINE-TUNED library RT (after)
        # Previously both were drawn on `filtered_output`, which has passed
        # empirical_fit's 4-sigma residual cut, so the figures had sharp edges BY
        # CONSTRUCTION and could not show what the filters removed -- a diagnostic
        # that cannot reveal whether its own filter is right.
        # Both figures are drawn against LIBRARY iRT on the x-axis -- the same
        # convention `filtered_output["updated_lib_rt"] = filtered_output["lib_rt"]`
        # sets above, and the domain the curves are functions OF. They show the
        # same points and differ only in the curve overlaid: the shared shape plus
        # a per-channel offset (before) vs the per-channel model actually chosen
        # (after). Putting the fine-tuned prediction (observed MINUTES, ~5-71) in
        # updated_lib_rt instead made the plot evaluate an iRT->RT curve at
        # minute-scale x, which sampled a narrow slice of the iRT domain and drew
        # curves at ~0.4 the slope of the points they were meant to describe.
        _plot_df, _plot_ch = filtered_output, channel
        # RTfit.png must draw the set each CURVE WAS FIT ON, otherwise points and
        # curve describe different populations and a correct fit looks broken.
        # Drawing the fine-tuning training set here (a different selection from
        # empirical_fit's survivors) is what made the T0/T1 curves appear to leave
        # their data at high library RT.
        _plot_training_set = os.environ.get("JMOD_RTFIT_PLOT_TRAINING", "1") == "1"
        try:
            if not _plot_training_set:
                raise RuntimeError("plotting the fitted set")
            _pl, _po, _pc = [], [], []
            for k in range(n_timeplex):
                for _s, _r, _l in zip(ft_seqs[k], ft_rt[k], ft_lib[k]):
                    _pl.append(_l); _po.append(_r); _pc.append(k)
            if _pl:
                _pl = np.asarray(_pl, float); _po = np.asarray(_po, float)
                _plot_ch = np.asarray(_pc, dtype=int)
                _plot_df = pd.DataFrame({"lib_rt": _pl, "rt": _po,
                                         "updated_lib_rt": _pl})
                logger.info(f"RTfit plots drawn on the TRAINING set "
                            f"({len(_plot_df):,} rows, unfiltered) rather than the "
                            f"4-sigma-filtered set ({len(filtered_output):,} rows)")
        except Exception as _e:
            logger.warning(f"could not plot the training set ({_e}); "
                           f"falling back to filtered_output")

        # `plot_curves` aliases `rt_spls`, which is MUTATED above: a channel that
        # chose fine-tuning holds a {seq: RT} dict, not a callable. Rebuild it as an
        # iRT->RT interpolator over the training peptides so RTfit.png shows the
        # model that was actually used, instead of raising TypeError on a dict or
        # silently drawing the empirical curve the channel rejected.
        _curves = list(plot_curves)
        for k in range(len(_curves)):
            if callable(_curves[k]):
                continue
            try:
                _d = _curves[k]
                _xy = [(float(_l), float(_d[_s]))
                       for _s, _l in zip(ft_seqs[k], ft_lib[k]) if _s in _d]
                _xy.sort()
                _cx = np.array([p[0] for p in _xy]); _cy = np.array([p[1] for p in _xy])
                _cx, _cy = _collapse_ties(_cx, _cy)
                _curves[k] = (lambda gx, cx=_cx, cy=_cy: np.interp(gx, cx, cy))
            except Exception as _e:
                logger.warning(f"could not render channel {k}'s fine-tuned curve "
                               f"for RTfit.png ({_e}); using the empirical curve")
                _curves[k] = emp_shape

        alignment_plots(_plot_df, emp_shape, emp_shape, f_rt_mz, mz_spl, rt_dist_params,
                        results_folder=results_folder, channels=_plot_ch,
                        rt_spls=_curves, offsets=offsets,
                        mz_source=filtered_output)
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



