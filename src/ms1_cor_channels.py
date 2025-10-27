import numpy as np
import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config 

import tqdm
# from src.trace_fns import fit_mTRAQ_isotopes
from src.utils.misc_functions import  closest_ms1spec,np_pearson_cor
from scipy.interpolate import interp1d
from src.utils import misc_functions as mf
# from src import iso_functions as iso

import warnings
from scipy.signal import find_peaks
import pickle
import math
from functools import reduce
from brainpy import isotopic_variants
import logging

import json
from pyteomics import mass
from scipy import optimize
from scipy.sparse import coo_matrix
import ptinnls as sparse_nnls
from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression
from scipy.sparse.linalg import lsmr
import numba as nb



min_int = 1e-3
from src.logger import logger


# @profile
def ms1_cor_channels(all_spectra, filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol,tag=None,timeplex=False):

    # config.tag=tag
    # print(config.tag)

    logger.info("Fitting tagged channels together")
    decoy_coeffs["untag_seq"] = [re.sub(rf"(\({tag.name}-\d+\))?","",peptide) for peptide in decoy_coeffs["seq"]]
    decoy_coeffs["untag_prec"] = ["_".join([i[0],str(int(i[1]))]) for i in zip(decoy_coeffs["untag_seq"],decoy_coeffs["z"])]
    
    if "med_frag_error" not in decoy_coeffs.columns:
        frag_errors = [mf.unstring_floats(mz) if mz==mz else [] for mz in decoy_coeffs.frag_errors]
        median  = np.median(np.concatenate([i for i in frag_errors]))
        decoy_coeffs["med_frag_error"] = [np.median(np.abs(median-i)) for i in frag_errors]
    
    if "abs_rt_error" not in decoy_coeffs.columns:
        decoy_coeffs["abs_rt_error"] = np.abs(decoy_coeffs.rt_error)
    
    if "abs_mz_error" not in decoy_coeffs.columns:
        decoy_coeffs["abs_mz_error"] = np.abs(decoy_coeffs.mz_error)
        
    
    ## features where bigger is better 
    greater_features =["hyperscore","frag_cosines_p","frag_cosines_p","manhattan_distances","coeff","frac_lib_int"]
    greater_features_present = [i for i in greater_features if i in decoy_coeffs.columns and np.ptp(decoy_coeffs[i][~np.isnan(decoy_coeffs[i])])>0]
    
    lesser_features =["scribe_scores","gof_stats","max_matched_residuals","med_frag_error","abs_mz_error","abs_rt_error"]
    lesser_features_present = [i for i in lesser_features if i in decoy_coeffs.columns and np.ptp(decoy_coeffs[i][~np.isnan(decoy_coeffs[i])])>0]
    
    
    ms1_spectra = all_spectra.ms1scans
    ms2_spectra = all_spectra.ms2scans
    
    ## array of ms1 and ms2 retention time
    ms2_rt = np.array([i.RT for i in ms2_spectra])
    ms1_rt = np.array([i.RT for i in ms1_spectra])
    # ms2_rt = np.array([i.RT for i in ms2_spectra])
    # ms1_rt = np.array([i.RT for i in ms1_spectra])
    
    ## array of scan numbers for ms1 and ms2 spectra
    ms1_spec_idxs = np.array([i.scan_num for i in ms1_spectra])
    ms2_spec_idxs = np.array([i.scan_num for i in ms2_spectra])
    # ms1_spec_idxs = np.array([i.scan_num for i in ms1_spectra])
    # ms2_spec_idxs = np.array([i.scan_num for i in ms2_spectra])
    
    ## get ms2 info for filtering
    bottom_of_window, top_of_window = np.array([i.ms1window for i in ms2_spectra]).T
    ms2_rt = np.array([i.RT for i in ms2_spectra])
    # bottom_of_window, top_of_window = np.array([i.ms1window for i in all_spectra.ms2scans]).T
    # ms2_rt = np.array([i.RT for i in all_spectra.ms2scans])

    ## list of scan nums of the closest ms1 scan for each ms2 scan
    resp_ms1scans = [ms1_spec_idxs[closest_ms1spec(ms2_rt[i], ms1_rt)] for i in range(len(ms2_rt))]

    ## mapping of ms2 scan nums to ms1 scan nums
    ms2_ms1_scan_map = {spec.scan_num:resp_ms1scans[i] for i,spec in enumerate(ms2_spectra)}

    
    if timeplex:
        grouped_decoy_coeffs = decoy_coeffs.groupby(["seq","z","time_channel"])
        fdc_group = filtered_decoy_coeffs.groupby(["untag_seq","z","time_channel"])
    else:
        grouped_decoy_coeffs = decoy_coeffs.groupby(["seq","z"])
        fdc_group = filtered_decoy_coeffs.groupby(["untag_seq","z"])

    all_ms1, all_coeff, all_iso, all_group_pearson, all_trace, all_fitted, all_group_keys, all_scans_len = ([] for _ in range(8))

    num_iso = config.num_iso_ms1
    num_iso_r = config.num_iso_r
    window_half_width = 10

    ms1_spec_dict = {k: {"fdc_idx": [], "peak_mz": [], "rel_iso_int": [], "monoiso_groups": [], "MS1_spectra": None} for k in ms1_spec_idxs}
    dummy_idx_list = list(filtered_decoy_coeffs.index)
    fake_fdc_dict = {}

    
    print(f"number of fdc: {len(filtered_decoy_coeffs)}")
    print(f"number of fdc groups: {len(fdc_group)}")
    fdc_group_idx = -1
    for key in tqdm.tqdm(list(fdc_group.groups)):
        fdc_group_idx += 1
        prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel = get_seqs_and_mzs(fdc_group, timeplex, tag, key)
        all_scans, spectra_subset = minmax_spec_window(largest_coeff_scans, ms1_spec_idxs, ms1_spectra, all_spectra, window_half_width)

        ms1_traces, coeff_traces, is_traces, all_pearson, iso_ratios = ([] for _ in range(5))
        obs_ratios, group_iso, group_keys, all_channel_scans, interp_funcs, best_coeff = ([] for _ in range(6))

        for prec_mz,prec_seq in zip(prec_mzs,prec_seqs):
            
            ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
            group_keys.append(channel_key)  
            
            interp_func = build_ms2_interpolator(ms2_vals)
            interp_funcs.append(interp_func)

            all_ms1_vals, all_ms2_vals, all_iso_vals, isotopes, interp_func = get_isotopes_and_vals(prec_seq, prec_z, num_iso, tag, all_scans, prec_mz, mz_ppm, spectra_subset, interp_func)   
            group_iso.append(isotopes)

            ## use monoiso ms1 prec mz to find the elution ms1 peak
            ms1_index_of_max = get_ms1_index_of_max(ms2_vals, top_ms1_spec_idx, highest_ranked_spec)
                
            ms1_peak_idx,ms1_peak_edge_idxs = get_ms1_peak(list(all_ms1_vals.keys()), moving_average(list(all_ms1_vals.values())), ms1_index_of_max)
            
            ## redefine all_scans to keep only thoe from the above peak
            channel_scans, all_ms1_vals, all_iso_vals, all_ms2_vals = filter_all_scans(all_scans, ms1_peak_edge_idxs, all_ms1_vals, all_iso_vals, all_ms2_vals)
            all_channel_scans.append(channel_scans)
            ms1_traces.append([all_ms1_vals,*all_iso_vals])
            coeff_traces.append(all_ms2_vals)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                all_pearson_to_append, iso_ratios_to_append = compute_ms1_ms2_cors(all_ms2_vals, all_ms1_vals, all_iso_vals, num_iso_r, channel_scans, isotopes)
                all_pearson.append(all_pearson_to_append)
                iso_ratios.append(iso_ratios_to_append)
            
        ### need to reduce the number of spectra we fit to
        ### fit to those from each channel
        scans_to_search = reduce_search_space(top_ms1_spec_idx, all_scans, all_channel_scans, window_half_width)

        group_pred, group_obs_peaks, group_matrices, group_fit_cor = ([] for _ in range(4))

        # ms1_spec_dict = add_to_ms1_spec_dict(ms1_spec_dict,
        #                                      ms1_spectra,
        #                                      ms1_spec_idxs,
        #                                      group_iso,
        #                                      key,
        #                                      fdc_group,
        #                                      filtered_decoy_coeffs,
        #                                      tag,
        #                                      scans_to_search,
        #                                      fdc_group_idx,
        #                                      dummy_idx_list,
        #                                      fake_fdc_dict
        #                                      )

        for ms1_spec_idx in scans_to_search:
            pred_coeff, obs_peaks, fit_matrix, fit_cor = fit_isotopes_and_score(ms1_spectra, ms1_spec_idxs, ms1_spec_idx, group_iso, mz_ppm)
            group_pred.append(pred_coeff)
            group_obs_peaks.append(obs_peaks)
            group_matrices.append(fit_matrix)
            group_fit_cor.append(fit_cor)
            
        # all_fitted.append(vals)
        all_fitted.append([np.array(group_pred),group_obs_peaks,group_matrices,group_fit_cor,scans_to_search])
        all_ms1.append(ms1_traces)
        all_coeff.append(coeff_traces)
        all_iso.append(iso_ratios)
        all_group_pearson.append(all_pearson)
        all_group_keys.append(group_keys)
        
        # break
        # all_pearson, ms1_traces, coeff_traces, iso_ratios

    new_output_dict = {}
    # for ms1_spec_idx, dicts in tqdm.tqdm(ms1_spec_dict.items()):
    #     if dicts["MS1_spectra"] == None:
    #         continue
    #     mapped_pred_coeffs, spectra_obs_peaks, spectra_sparse_matrix = fit_whole_MS1_spectrum(dicts, mz_ppm, ms1_spec_idx)
    #     new_output_dict[ms1_spec_idx] = {
    #                                     "mapped_pred_coeffs": mapped_pred_coeffs,
    #                                     "spectra_obs_peaks": spectra_obs_peaks,
    #                                     "spectra_sparse_matrix": spectra_sparse_matrix
    #                                     }
        
    # global nanmeans, var_ints

    # log10var = np.log10(var_ints)
    # log10mean = np.log10(nanmeans)
    # model = LinearRegression()
    # model.fit(log10mean.reshape(-1, 1), log10var)
    # intercept = model.intercept_
    # coefficient = model.coef_[0]

    # print(f"Intercept: {intercept}")
    # print(f"Slope: {coefficient}")
    # r2 = model.score(log10mean.reshape(-1, 1), log10var)
    # print(f"R²: {r2}")

    # hb = plt.hexbin(
    #     log10mean,
    #     log10var,
    #     gridsize=80,
    #     cmap="viridis",
    #     norm=mpl.colors.LogNorm()
    # )

    # plt.xlabel("Log10 Intensity")
    # plt.ylabel("Log10 Modelled Variance")
    # plt.title(f"Modelled Variance vs. Mean")
    # plt.colorbar(hb, label="Counts")
    # plt.xlim(3.5, 13)
    # plt.ylim(3.5, 13)
    # plt.show()
    # plt.close()

    # fitted = coefficient*log10mean + intercept
    # residuals = log10var - fitted

    # hb = plt.hexbin(
    #     fitted,
    #     residuals,
    #     gridsize=80,
    #     cmap="viridis",
    #     norm=mpl.colors.LogNorm()
    # )

    # plt.xlabel("Fitted values")
    # plt.ylabel("Residuals (Observed - Fitted)")
    # plt.title(f"Residuals vs Fitted Values For LogLog Line of Best fit")
    # plt.axhline(0, color='black', linestyle='--', linewidth=1)
    # plt.colorbar(hb, label="Counts")
    # plt.show()
    # plt.close()




    
    return all_group_pearson, all_ms1, all_coeff, all_iso, all_group_keys, all_fitted, new_output_dict, fake_fdc_dict

# @profile
def get_seqs_and_mzs(fdc_group, timeplex, tag, key):
    """
    Retrieve sequences, m/z values, charge, retention time, and scan information 
    for a given precursor group, optionally including timeplex channels.

    Parameters
    ----------
    fdc_group : pandas.core.groupby.generic.DataFrameGroupBy
        Grouped FDC DataFrame, either by ('seq', 'z') or ('seq', 'z', 'time_channel') if timeplex.
    timeplex : bool
        Whether the data includes timeplex channels.
    tag : massTag
        A massTag instance
    key : tuple
        The key identifying the group in fdc_group:
        - (seq, z) if not timeplex
        - (seq, z, time_channel) if timeplex

    Returns
    -------
    prec_seqs : tuple of str
        Unstripped sequences for all channels in the group.
    prec_mzs : tuple of float
        m/z values corresponding to each sequence in prec_seqs.
    prec_z : int or float
        Charge of the precursor.
    prec_rt : float
        Retention time of the channel with the highest coefficient.
    top_ms1_spec_idx : int
        MS1 spectrum index of the channel with the highest coefficient.
    largest_coeff_scans : list of int
        List of MS1 spectrum indices for all scans in the group.
    time_channel : int or None
        Time channel index if timeplex is True; otherwise None.
    """

    tag_group = fdc_group.get_group(key)
    prec_mzs = tag_group["mz"]
    prec_seqs = tag_group["seq"]
    prec_z = key[1]
    if timeplex:
        time_channel = key[2]
    else:
        time_channel = None
    largest_id = np.argmax(tag_group["coeff"])
    top_ms1_spec_idx = list(tag_group["Ms1_spec_id"])[largest_id]
    prec_rt = list(tag_group["rt"])[largest_id]
    
    ### search for all channels always:
    channel_dict = get_other_channels((prec_seqs.iloc[largest_id],prec_z), prec_mzs.iloc[largest_id], tag)
    prec_seqs,prec_mzs = tuple(zip(*channel_dict.values()))

    largest_coeff_scans = list(tag_group["Ms1_spec_id"])

    return prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel

def minmax_spec_window(largest_coeff_scans, ms1_spec_idxs, ms1_spectra, all_spectra, window_half_width): 
    """
    Retrieve the list of MS1 scan and scan idxs within a window of the highest coeff scans

    Parameters
    ----------
    largest_coeff_scans : list of int
        List of MS1 spectrum indices for all scans in the group.
    ms1_spec_idxs : 1d array of int
        Array of all MS1 spec indexes in MS run
    ms1_spectra : list of Spectrum
        Spectrum Object
    all_spectra : SpectrumFile
        SpectrumFile Object containing all Spectrum Objects
    window_half_width: int
        The width around the largest coeff scans that are returned

    Returns
    -------
    all_scans : list of int
        A list of scan idxs 
    spectra_subset : list of Spectrum
        A list of corresponding Spectrum Objects

    Notes
    -----
    The function identifies the min and max scan numbers from `largest_coeff_scans` 
    and selects a contiguous window of MS1 scans surrounding them, 
    bounded by `window_half_width`

    """
    ## max and min of this list
    max_scan, min_scan = max(largest_coeff_scans), min(largest_coeff_scans)
    ms1_list_idx_min = list(ms1_spec_idxs).index(min_scan)
    ms1_list_idx_max = list(ms1_spec_idxs).index(max_scan)
    scans_each_side = np.array(ms1_spec_idxs)[np.arange(max(0,ms1_list_idx_min-window_half_width),min(len(ms1_spectra),ms1_list_idx_max+window_half_width+1))]
    all_scans = list(scans_each_side)

    spectra_subset = [all_spectra.get_by_idx(idx) for idx in all_scans]
    return all_scans, spectra_subset


def get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs):
    # print(prec_seq)
    # print(prec_z)
    # print(prec_rt)
    # print(time_channel)
    # print(timeplex)
    # print(grouped_decoy_coeffs)
    # print(ms2_rt)
    # print(rt_tol)
    # print(prec_mz)
    # print(bottom_of_window)
    # print(top_of_window)
    # print(ms2_spec_idxs)
    # print("\n")

    
    ## keep decoys mathching to the correct MS1
    # offset = config.decoy_mz_offset if "Decoy" in prec_seq else 0
    offset = 0
    
    if timeplex:
        channel_key = (prec_seq,prec_z,time_channel)
    else:
        channel_key = (prec_seq,prec_z)
    
    ## create dummy 
    ms2_vals = {0:0}
    
    if channel_key in grouped_decoy_coeffs.groups:

        # use coeff as rank score directly
        group = grouped_decoy_coeffs.get_group(channel_key)
        highest_ranked_spec = group.loc[group["coeff"].idxmax(), "Ms1_spec_id"]


        ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
        prec_rt = group.rt.iloc[np.argmax(group.coeff)]
        ms2_window_bool = np.logical_and(prec_mz+offset>bottom_of_window,prec_mz+offset<top_of_window)
        
        min_rt = np.minimum(prec_rt-rt_tol,np.min(group.rt)*.99)
        max_rt = np.maximum(prec_rt+rt_tol,np.max(group.rt)*1.01)
        ms2_rt_bool = np.logical_and(ms2_rt>=min_rt,ms2_rt<=max_rt)
        
        ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
        possible_ms2_scans = ms2_spec_idxs[ms2_bool]
        ms2_vals = {i:min_int for i in possible_ms2_scans}
    
        for scan,c in zip(group["spec_id"],group["coeff"]):
            ms2_vals[scan]=c
    else:
        highest_ranked_spec = None

    # print(ms2_vals)
    # print(highest_ranked_spec)
    # print(channel_key)

    return ms2_vals, highest_ranked_spec, channel_key

# def get_isotopes_and_vals(prec_seq, prec_z, num_iso, tag, all_scans, prec_mz, mz_ppm, spectra_subset, interp_func):
#     ms1_vals = get_precursor_trace(prec_mz, mz_ppm, spectra_subset)
#     isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
#     prec_isotope_traces = get_isotope_traces(isotopes, mz_ppm, spectra_subset)
#     all_ms1_vals, all_ms2_vals, all_iso_vals = unnamed_function(all_scans, prec_isotope_traces, interp_func, ms1_vals)

#     return all_ms1_vals, all_ms2_vals, all_iso_vals, isotopes, interp_func

def get_isotopes_and_vals(prec_seq, prec_z, num_iso, tag, all_scans, prec_mz, mz_ppm, spectra_subset, interp_func):
    isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
    ms1_vals, prec_isotope_traces = get_isotope_traces_vectorized(isotopes, mz_ppm, spectra_subset)
    all_ms1_vals, all_ms2_vals, all_iso_vals = fill_scan_values(all_scans, prec_isotope_traces, interp_func, ms1_vals)

    return all_ms1_vals, all_ms2_vals, all_iso_vals, isotopes, interp_func

def build_ms2_interpolator(ms2_vals):
    return interp1d(list(ms2_vals.keys()), np.array(list(ms2_vals.values())), bounds_error=False)   

# def get_precursor_trace(prec_mz, mz_ppm, spectra_subset):
#     return {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec in spectra_subset}

# def get_isotope_traces(isotopes, mz_ppm, spectra_subset):
#     prec_isotope_traces=[]
#     ## note: we have collected similar values for previous channel if the isotopic envelopes are overlapping. 
#     ### However, in cases like diethlyation, isoptopes can differ by > 10 ppm #!!!Maybe investigate wider ppm tol for these cases?
#     for isotope in isotopes[1:]:# we already have the monoisotopic trace
#         iso_trace = {spec.scan_num:get_trace_int(spec, isotope.mz,rtol=mz_ppm) for spec in spectra_subset}
#         prec_isotope_traces.append(iso_trace)
    
#     return prec_isotope_traces

def get_isotope_traces_vectorized(isotopes, mz_ppm, spectra_subset):
    ## note: we have collected similar values for previous channel if the isotopic envelopes are overlapping. 
    ### However, in cases like diethlyation, isoptopes can differ by > 10 ppm #!!!Maybe investigate wider ppm tol for these cases?
    prec_isotope_traces=[{} for _ in isotopes]

    for i, spec in enumerate(spectra_subset):
        iso_ints = get_trace_int_numba(spec.mz, spec.intens, np.array([isotope.mz for isotope in isotopes]), 0, mz_ppm, min_int)
        for i in range(len(isotopes)):
            prec_isotope_traces[i][spec.scan_num] = iso_ints[i]

    
    return prec_isotope_traces[0], prec_isotope_traces[1:]

@nb.njit(cache=True)
def get_trace_int_numba(spec_mz, spec_intens, mz_array, atol, rtol, base):

    order_idx = np.searchsorted(spec_mz, mz_array)

    left_idx = np.clip(order_idx - 1, 0, len(spec_mz) - 1)
    right_idx = np.clip(order_idx, 0, len(spec_mz) - 1)

    # compute diffs
    left_diff = np.abs(spec_mz[left_idx] - mz_array)
    right_diff = np.abs(spec_mz[right_idx] - mz_array)

    # pick closer side
    choose_right = right_diff < left_diff
    closest_idx = np.where(choose_right, right_idx, left_idx)
    mz_diff = np.where(choose_right, right_diff, left_diff)

    # apply tolerance
    within_tol = mz_diff <= mz_array * rtol

    # build output
    vals_to_return = np.where(within_tol, spec_intens[closest_idx], base)

    return vals_to_return


def compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag):
    # isotopes = iso.precursor_isotopes(prec_seq,prec_z,num_iso)
    isotopes = precursor_isotopes(prec_seq,prec_z,num_iso)

    delta_mz = 0
    if tag.name in prec_seq:
        delta_mz = prec_mz-isotopes[0].mz
    for i in isotopes:
        i.mz+=delta_mz

    return isotopes

# def unnamed_function(all_scans, prec_isotope_traces, interp_func, ms1_vals):
#     all_ms1_vals = {i:min_int for i in all_scans}
#     all_ms2_vals = {i:min_int for i in all_scans}
#     all_iso_vals = [{i:min_int for i in all_scans} for _ in range(len(prec_isotope_traces))]
    
#     for scan,c in zip(all_scans,interp_func(all_scans)):
#         if scan in ms1_vals:
#             all_ms1_vals[scan] = ms1_vals[scan]
#             all_ms2_vals[scan] = c#f(scan)
#         for iso_idx in range(len(prec_isotope_traces)):
#             if scan in prec_isotope_traces[iso_idx]:
#                 all_iso_vals[iso_idx][scan] = prec_isotope_traces[iso_idx][scan]

#     return all_ms1_vals, all_ms2_vals, all_iso_vals

def fill_scan_values(all_scans, prec_isotope_traces, interp_func, ms1_vals):
    all_ms1_vals = {i: min_int for i in all_scans}
    all_ms2_vals = {i: min_int for i in all_scans}
    all_iso_vals = [{i: min_int for i in all_scans} for _ in range(len(prec_isotope_traces))]

    interp_vals = interp_func(all_scans)

    for scan, c in zip(all_scans, interp_vals):
        if scan in ms1_vals:
            all_ms1_vals[scan] = ms1_vals[scan]
            all_ms2_vals[scan] = c

    for iso_idx, iso_dict in enumerate(prec_isotope_traces):
        for scan, val in iso_dict.items():
            all_iso_vals[iso_idx][scan] = val

    return all_ms1_vals, all_ms2_vals, all_iso_vals


def get_ms1_index_of_max(ms2_vals, top_ms1_spec_idx, highest_ranked_spec):
    ## use monoiso ms1 prec mz to find the elution ms1 peak
    if ms2_vals=={0:0}:
        ms1_index_of_max = top_ms1_spec_idx ## should I just use the max of MS1???? Need to look into again
    else:
        ms1_index_of_max = highest_ranked_spec
    return ms1_index_of_max

def filter_all_scans(all_scans, ms1_peak_edge_idxs, all_ms1_vals, all_iso_vals, all_ms2_vals):
    channel_scans = all_scans[all_scans.index(ms1_peak_edge_idxs[0]):all_scans.index(ms1_peak_edge_idxs[1])+1]
    all_ms1_vals = {i:all_ms1_vals[i] for i in channel_scans}
    all_iso_vals = [{i:iso_vals[i] for i in channel_scans} for iso_vals in all_iso_vals]
    all_ms2_vals = {i:all_ms2_vals[i] for i in channel_scans}

    return channel_scans, all_ms1_vals, all_iso_vals, all_ms2_vals

def compute_ms1_ms2_cors(all_ms2_vals, all_ms1_vals, all_iso_vals, num_iso_r, channel_scans, isotopes):
    spec_pearsons = [np_pearson_cor(list(all_ms2_vals.values()),list(i.values())).statistic for i in [all_ms1_vals,*all_iso_vals[:num_iso_r]]]

    ms1_spec_idx = channel_scans[np.argmax(list(all_ms2_vals.values()))]

    theoretical_pattern = [i.intensity for i in isotopes]
    obs_pattern = [all_ms1_vals[ms1_spec_idx],*[iso_trace[ms1_spec_idx] for iso_trace in all_iso_vals]]
    iso_ratio = [np_pearson_cor(theoretical_pattern,obs_pattern),theoretical_pattern,obs_pattern]

    return spec_pearsons, iso_ratio

def reduce_search_space(top_ms1_spec_idx, all_scans, all_channel_scans, window_half_width):
    idx_of_max =all_scans.index(top_ms1_spec_idx)
    scans_to_search = np.array(all_scans)[np.arange(max(0,idx_of_max-window_half_width),min(len(all_scans),idx_of_max+window_half_width+1))]
    scans_to_search = np.sort(np.unique(np.concatenate(all_channel_scans)))
    return scans_to_search

def fit_isotopes_and_score(ms1_spectra, ms1_spec_idxs, ms1_spec_idx, group_iso, mz_ppm):
    spec = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]
               
    pred_coeff, obs_peaks, fit_matrix = fit_channel_isotopes_numba(spec,group_iso,mz_ppm)

    if len(obs_peaks)==0:
        fit_cor = np.nan
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_cor = np_pearson_cor(np.sum(fit_matrix*pred_coeff,1),obs_peaks)
      
    return pred_coeff, obs_peaks, fit_matrix, fit_cor

#@profile
def add_to_ms1_spec_dict(ms1_spec_dict, ms1_spectra, ms1_spec_idxs, group_iso, key, fdc_group, filtered_decoy_coeffs, tag, scans_to_search, fdc_group_idx, dummy_idx_list, fake_fdc_dict):
    # for g in group_iso:
    #     print(g) 
    idxs = fdc_group.groups[key]
    rows = filtered_decoy_coeffs.loc[idxs]
    # print(rows[["seq", "channel"]])
    # This all assumes tag.channel_names is ordered, as well as group_iso is ordered
    row_channels = rows["channel"].to_numpy()
    observed_channels = []
    for channel in tag.channel_names:
        if int(channel) in row_channels:
            observed_channels.append(channel)
    # print(f"obsered_channels: {observed_channels}")
    
    
    for i, channel in enumerate(tag.channel_names):
        if not channel in observed_channels:
            # This assumes dummy_idx_list is sorted
            if dummy_idx_list[-1]+1 in dummy_idx_list:
                raise ValueError("Dummy Index List +1 is in Dummy Index List Already")
            dummy_idx_list.append(dummy_idx_list[-1]+1)
            fdc_index = dummy_idx_list[-1]
            fake_fdc_dict[fdc_index] = (fdc_group_idx, channel)
        else:
            fdc_index = rows.index[rows["channel"] == int(channel)]
            if len(fdc_index) == 0:
                raise ValueError("This is a valueError")
            elif len(fdc_index) > 1:
                raise ValueError(f"Multiple matches for channel {channel}")
            else:
                fdc_index = fdc_index[0]

        iso_group = group_iso[i]


        for ms1_spec_idx in scans_to_search:
            for peak in iso_group:
                ms1_spec_dict[ms1_spec_idx]["peak_mz"].append(peak.mz)
                ms1_spec_dict[ms1_spec_idx]["rel_iso_int"].append(peak.intensity)
                ms1_spec_dict[ms1_spec_idx]["fdc_idx"].append(fdc_index)

        if ms1_spec_dict[ms1_spec_idx]["MS1_spectra"] is None:
            ms1_spec_dict[ms1_spec_idx]["MS1_spectra"] = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]
    
    for ms1_spec_idx in scans_to_search:
        ms1_spec_dict[ms1_spec_idx]["monoiso_groups"].append([iso_group[0].mz for iso_group in group_iso])

    return ms1_spec_dict

from scipy.sparse import diags
import matplotlib as mpl

#@profile
def fit_whole_MS1_spectrum(dicts, mz_ppm, ms1_spec_idx):

    spec = dicts["MS1_spectra"]
    mz_peaks = spec.mz
    dia_spec_int = spec.intens

    offsets = mz_ppm * mz_peaks
    centroid_breaks = np.empty(mz_peaks.size * 2, dtype=mz_peaks.dtype)
    centroid_breaks[:mz_peaks.size] = mz_peaks - offsets
    centroid_breaks[mz_peaks.size:] = mz_peaks + offsets
    centroid_breaks.sort()

    # intercept, slope = calculate_mean_variance(centroid_breaks, mz_peaks, dia_spec_int, dicts["monoiso_groups"])
    # calculate_mean_variance(centroid_breaks, mz_peaks, dia_spec_int, dicts["monoiso_groups"])


    # print(f"centroid_breaks: {centroid_breaks}")

    all_mz = dicts["peak_mz"]
    ref_coords = np.searchsorted(centroid_breaks, all_mz)
    # print(f"ref_coords: {ref_coords}")

    lib_peaks_matched = np.array((ref_coords % 2 == 1).tolist())
    # print(f"lib_peaks_matched: {lib_peaks_matched}")

    num_lib_peaks_unmatched = np.count_nonzero(~lib_peaks_matched)
    # print(num_lib_peaks_unmatched)

    #matrix = np.zeros((len(mz_peaks)+num_lib_peaks_unmatched, len(set(dicts["fdc_idx"]))), dtype=float)


    unique_fdcs = (set(dicts["fdc_idx"]))
    fdc_to_col = {fdc: i for i, fdc in enumerate(unique_fdcs)}

    last_unnasigned_peak_idx = len(mz_peaks)
    if not len(ref_coords) == len(lib_peaks_matched) == len(dicts["rel_iso_int"]) == len(dicts["peak_mz"]) == len(dicts["fdc_idx"]):
        raise ValueError("Length of lists for fitting is not the same")
    
    sparse_rows = []
    sparse_cols = []
    sparse_vals = []
    fdc_idx_dummy_peak_dict = {k: None for k in unique_fdcs}
    for fdc_idx, mz, intensity, ref_coord, matched in zip(dicts["fdc_idx"],
                                                            dicts["peak_mz"],
                                                            dicts["rel_iso_int"],
                                                            ref_coords,
                                                            lib_peaks_matched
                                                            ):

        # #Type C
        # if matched:
        #     ms1_peak_coord = (ref_coord-1)//2
        # else:
        #     last_unnasigned_peak_idx += 1
        #     ms1_peak_coord = last_unnasigned_peak_idx

        # #Type B
        # if matched:
        #     ms1_peak_coord = (ref_coord-1)//2
        # else:
        #     if fdc_idx_dummy_peak_dict[fdc_idx] == None:
        #         last_unnasigned_peak_idx += 1
        #         fdc_idx_dummy_peak_dict[fdc_idx] = last_unnasigned_peak_idx
                
        #     ms1_peak_coord = fdc_idx_dummy_peak_dict[fdc_idx]

        # sparse_rows.append(ms1_peak_coord)
        # sparse_cols.append(fdc_to_col[fdc_idx])
        # sparse_vals.append(intensity)

        # matrix_sparse = coo_matrix((sparse_vals, (sparse_rows, sparse_cols)))
        # dia_spec_int = np.append(dia_spec_int,[0]*(matrix_sparse.shape[0]-dia_spec_int.shape[0])) 


        # Type A
        if matched: 
            ms1_peak_coord = (ref_coord-1)//2
            sparse_rows.append(ms1_peak_coord)
            sparse_cols.append(fdc_to_col[fdc_idx])
            sparse_vals.append(intensity)

    matrix_sparse = coo_matrix((sparse_vals, (sparse_rows, sparse_cols)), shape=(len(mz_peaks), len(unique_fdcs)))

    # dia_spec_int_vars = dia_spec_int**slope * 10**intercept
    # median_var = np.median(dia_spec_int_vars)

    


    matrix_csr = matrix_sparse.tocsr()
    mask = np.diff(matrix_csr.indptr) != 0
    matrix_sparse = matrix_csr[mask].tocoo()
    dia_spec_int = dia_spec_int[mask]

    # count = np.sum(matrix_sparse.data > 0.999)
    # print(f"Number of entries > 0.999: {count}")


    # plt.hist(np.log10(dia_spec_int[dia_spec_int != 0]))
    # plt.title(f"Spectra {ms1_spec_idx}")
    # plt.ylabel("Counts")
    # plt.xlabel("log10 Intensity")
    # plt.show()

    # dia_spec_int_vars = np.append(dia_spec_int_vars, [median_var]*(matrix_sparse.shape[0]-dia_spec_int_vars.shape[0]))
    # weights = 1/dia_spec_int_vars

    # sqrt_w = np.sqrt(weights)
    # W_sqrt = diags(sqrt_w)
    # X_w = W_sqrt @ matrix_sparse
    # y_w = sqrt_w * dia_spec_int 

    res = lsq_linear(matrix_sparse, dia_spec_int, bounds=(0, np.inf), method="trf", lsmr_tol="auto")
    # res = lsq_linear(X_w, y_w, bounds=(0, np.inf), method="trf", lsmr_tol="auto")
    lib_coefficients = res.x
    lib_coefficients = lib_coefficients.flatten()
    mapped_lib_coefficients = {k: v for k, v in list(zip(unique_fdcs, lib_coefficients))}

    # if ms1_spec_idx in [11949, 12104, 12019, 11960]:
    #     matrix_dense = matrix_sparse.toarray()   
    #     fitted = matrix_sparse @ res.x
    #     residuals = dia_spec_int - fitted

    #     n, p = matrix_dense.shape
    #     mse = np.sum(residuals**2) / (n - p)

    #     XtX_inv = np.linalg.pinv(matrix_dense.T @ matrix_dense)
    #     H_diag = np.einsum("ij,jk,ik->i", matrix_dense, XtX_inv, matrix_dense)
    #     H_diag = np.clip(H_diag, None, 0.999999)


    #     # Cook's distance
    #     cooks_d = (residuals**2 / (p * mse)) * (H_diag / (1 - H_diag)**2)

    #     plt.figure(figsize=(8,4))
    #     plt.stem(cooks_d, markerfmt=",", basefmt=" ")
    #     plt.xlabel("Observation index")
    #     plt.ylabel("Cook’s Distance")
    #     plt.title(f"Cook’s Distance for MS1 Spectrum Fit for Spec {ms1_spec_idx}")
    #     # Common rule of thumb:
    #     threshold = 4 / (n - p)
    #     plt.axhline(threshold, color="red", linestyle="--", label="4/(n-p) threshold")
    #     plt.legend()
    #     plt.show()

    #     cooks_d_clipped = np.clip(cooks_d, 1e-10, None)
    #     plt.figure(figsize=(8,4))
    #     plt.stem(cooks_d_clipped, markerfmt=",", basefmt=" ")
    #     plt.xlabel("Observation index")
    #     plt.ylabel("Cook’s Distance")
    #     plt.title(f"Cook’s Distance for MS1 Spectrum Fit for Spec {ms1_spec_idx}")
    #     # Common rule of thumb:
    #     threshold = 4 / (n - p)
    #     plt.axhline(threshold, color="red", linestyle="--", label="4/(n-p) threshold")
    #     plt.yscale("log")
    #     plt.legend()
    #     plt.show()
        



    # fitted = matrix_sparse @ res.x
    # residuals = dia_spec_int - fitted

    # hb = plt.hexbin(
    #     fitted,
    #     residuals,
    #     gridsize=80,
    #     cmap="viridis",
    #     norm=mpl.colors.LogNorm()
    # )

    # plt.xlabel("Fitted values")
    # plt.ylabel("Residuals (Observed - Fitted)")
    # plt.title(f"Residuals vs Fitted Values Spec {ms1_spec_idx}")
    # plt.axhline(0, color='black', linestyle='--', linewidth=1)
    # plt.colorbar(hb, label="Counts")
    # plt.show()
    # plt.close()

    # fitted_clipped = np.clip(fitted, 10**-2.5, None)
    # log_x = np.log10(fitted_clipped)
    # hb = plt.hexbin(
    #     log_x,
    #     residuals,
    #     gridsize=80,
    #     cmap="viridis",
    #     norm=mpl.colors.LogNorm()
    # )

    # plt.xlabel("Log10 Fitted values")
    # plt.ylabel("Residuals (Observed - Fitted)")
    # plt.title(f"Residuals vs Fitted Values Spec {ms1_spec_idx}")
    # plt.axhline(0, color='black', linestyle='--', linewidth=1)
    # plt.colorbar(hb, label="Counts")
    # plt.show()
    # plt.close()





    # except:
    #     logger.info(f"LSQ Linear Failed for spec: {ms1_spec_idx}")
    #     logger.info(f"Matrix.shape: {matrix_sparse.shape}")
    #     logger.info(f"Obs_Spectra.shape: {dia_spec_int.shape}")
    #     mapped_lib_coefficients = {k: np.nan for k in unique_fdcs}



    #return lib_coefficients, dia_spec_int, new_matrix, matrix
    return mapped_lib_coefficients, dia_spec_int, matrix_sparse


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
nanmeans = []
var_ints = []
def calculate_mean_variance(centroid_breaks, mz_peaks, dia_spec_int, monoiso_groups):
    global nanmeans, var_ints
    # nanmeans = []
    # var_ints = []
    monoisos_matched = []
    for plex_group in monoiso_groups:
        monoiso_mzs = [p for p in plex_group]
        ref_coords = np.searchsorted(centroid_breaks, monoiso_mzs)
        lib_peaks_matched = (ref_coords % 2 == 1)

        num_monoiso_matched = np.sum(lib_peaks_matched)
        if num_monoiso_matched != 5:
            continue

        group_intensities = np.where(
                                lib_peaks_matched,
                                dia_spec_int[(ref_coords-1)//2],
                                np.nan
        )
        
        nanmean = np.nanmean(group_intensities)
        var_int = np.nanvar(group_intensities)
        monoisos_matched.append(num_monoiso_matched)
        nanmeans.append(nanmean)
        var_ints.append(var_int)

    # log10var = np.log10(var_ints)
    # log10mean = np.log10(nanmeans)
    # model = LinearRegression()
    # model.fit(log10mean.reshape(-1, 1), log10var)
    # intercept = model.intercept_
    # coefficient = model.coef_[0]

    # print(f"Intercept: {intercept}")
    # print(f"Slope: {coefficient}")

    # plt.scatter(nanmeans, var_ints,  c=monoisos_matched, cmap=ListedColormap(plt.cm.tab10.colors[:5]), alpha=0.4)
    # plt.title("Mean Variance Relationship")
    # plt.xlabel("Log10 Mean Intensity")
    # plt.ylabel("Log10 Intensity Variance")
    # plt.xscale("log")
    # plt.yscale("log")
    # cbar = plt.colorbar()
    # cbar.set_label("Monoisos Matched")
    # cbar.locator = ticker.MaxNLocator(integer=True)
    # cbar.update_ticks()
    # lims = [np.nanmin([nanmeans, var_ints]), np.nanmax([nanmeans, var_ints])]
    # lims[0] = lims[0] - lims[0]*0.1
    # lims[1] = lims[1] + lims[1]*0.5
    # plt.xlim(lims)
    # plt.ylim(lims)
    # plt.axis("square")
    # plt.show()

    # return intercept, coefficient




    

def get_other_channels(prec,mz,tag):
    """
    Return the m/z and sequences for all channels of a given precursor.

    Parameters
    ----------
    prec : tuple
        A tuple (unstripped_seq, z) containing the peptide sequence (with tags) 
        and its charge state.
    mz : float
        The m/z value of the precursor in the current channel.
    tag : massTag
        A massTag instance.

    Returns
    -------
    channel_dict : dict
        A dictionary mapping each channel name (e.g. 'PSMtag_5plex-0') to a list:
        [unstripped_seq, float(mz)] for that channel.
    """
    
    ## identify what channel the current prec is in
    channels = re.findall(rf"({tag.name}-\d+)",prec[0])
    num_tags = len(channels)
    assert len(set(channels))==1, f"{channels}"
    channel = channels[0]
    assert channel in tag.mass_dict
    channel_dict = {i:[] for i in tag.mass_dict}
    
    for c in channel_dict:
        if c==channel:
            channel_dict[channel] = [prec[0],mz]
        else:
            c_seq = re.sub(channel,c,prec[0])
            c_mz = mz + (num_tags*(tag.mass_dict[c]-tag.mass_dict[channel])/prec[1])
            channel_dict[c] = [c_seq,c_mz]
    return channel_dict

# # @profile
# def get_trace_int(spec,mz,atol=0,rtol=0,base=min_int):
#     ## speed up of above
#     order_idx = np.searchsorted(spec.mz, mz)
    
#     # Handle edge cases for indices at the bounds
#     if order_idx == 0:
#         closest_idx = 0
#         mz_diff = spec.mz[0]-mz
#     elif order_idx == len(spec.mz):
#         closest_idx = len(spec.mz) - 1
#         mz_diff = mz-spec.mz[-1]
#     else:
#         # Compare the closest values on both sides of the searchsorted index
#         left_idx = order_idx - 1
#         right_idx = order_idx
        
#         # Find the closest value between the two neighboring indices
#         left_diff = abs(spec.mz[left_idx] - mz)
#         right_diff = abs(spec.mz[right_idx] - mz)
#         if left_diff < right_diff:
#             closest_idx = left_idx
#             mz_diff = left_diff
#         else:
#             closest_idx = right_idx
#             mz_diff = right_diff
    
# #    mz_diff = abs(spec.mz[closest_idx] - mz)
#     if mz_diff <= mz * rtol:  # Use the relative tolerance condition
#         return spec.intens[closest_idx]

#     return base

def get_ms1_peak(x,y,idx):
    x = np.array(x)
    y = np.array(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        peaks,peak_attr= find_peaks(y,width=(None,None))
    
    ## if no peak, return the index of the max
    if len(peaks)==0:
        return x[np.argmax(y)],[x[0],x[-1]]
    
    peak_idxs = x[peaks]
    
    closest_idx = np.argmin(np.abs(idx-peak_idxs))

    peak_idx = peaks[closest_idx]
    peak_edge_idxs = [max(0,peak_attr["left_bases"][closest_idx]-config.additional_scans),min(len(x)-1,peak_attr["right_bases"][closest_idx]+config.additional_scans)]
    # peak_edge_idxs = [peak_attr["left_bases"][closest_idx],peak_attr["right_bases"][closest_idx]]
    
    return x[peak_idx],x[peak_edge_idxs]


def moving_average(x, w=4):
    return np.convolve(x, np.ones(w), 'same') / w


def fit_channel_isotopes_numba(spec,all_iso,mz_ppm, typefit="b"):

    flat = np.array([(p.mz, p.intensity) for iso in all_iso for p in iso], dtype=np.float64)
    ms1_iso_patterns = flat.reshape(len(all_iso), len(all_iso[0]), 2)

    dia_spectrum = np.array(spec.peak_list(), dtype=np.float64).T

    group_lengths = np.array([len(g) for g in all_iso])

    dense_matrix, dia_spec_int = get_matrix_to_fit_numba(ms1_iso_patterns, group_lengths, dia_spectrum, len(all_iso), mz_ppm)

    nonzero_mask = np.any(dense_matrix != 0, axis=1)
    nonzero_mask[-len(all_iso):] = True
    dense_matrix = dense_matrix[nonzero_mask, :]
    dia_spec_int = dia_spec_int[nonzero_mask]

    lib_coefficients, residuals = optimize.nnls(dense_matrix, dia_spec_int)

    return lib_coefficients, dia_spec_int, dense_matrix



@nb.njit(cache=True)
def get_matrix_to_fit_numba(ms1_iso_patterns, group_lengths, dia_spectrum, all_iso_len, mz_ppm):    
    ### we only need to conseider the part of the spectrum that falls within the isotopic envelopes of the channels
    min_isotope = ms1_iso_patterns[:, :, 0].min() - 1
    max_isotope = ms1_iso_patterns[:, :, 0].max() + 1

    mz = dia_spectrum[:, 0]
    lo = np.searchsorted(mz, min_isotope, side="right")
    hi = np.searchsorted(mz, max_isotope, side="left")
    dia_spectrum = dia_spectrum[lo:hi]

    mz_peaks = dia_spectrum[:, 0]
    offsets = mz_ppm * mz_peaks
    centroid_breaks = np.empty(mz_peaks.size * 2, dtype=mz_peaks.dtype)
    centroid_breaks[:mz_peaks.size] = mz_peaks - offsets
    centroid_breaks[mz_peaks.size:] = mz_peaks + offsets
    centroid_breaks.sort()

    fdc_idxs = np.repeat(np.arange(all_iso_len), group_lengths)
    all_mz = ms1_iso_patterns[:,:,0].ravel()
    rel_iso_int = ms1_iso_patterns[:, :, 1].ravel()
    ref_coords = np.searchsorted(centroid_breaks, all_mz)

    lib_peaks_matched = (ref_coords % 2 == 1)

    # last_unnasigned_peak_idx = (max(ref_coords)-1)//2
    last_unnasigned_peak_idx = len(mz_peaks)
        
    unique_fdcs = (set(fdc_idxs))

    n_rows = len(mz_peaks) + len(unique_fdcs)
    n_cols = len(unique_fdcs)

    dense_matrix = np.zeros((n_rows, n_cols), dtype=np.float64)

    ms1_peak_coord = np.where(
        lib_peaks_matched,
        (ref_coords - 1) // 2,
        last_unnasigned_peak_idx + fdc_idxs
    )

    # np.add.at(dense_matrix, (ms1_peak_coord, fdc_idxs), rel_iso_int)
    for i in range(rel_iso_int.size):
        dense_matrix[ms1_peak_coord[i], fdc_idxs[i]] += rel_iso_int[i]

    dia_spec_int = dia_spectrum[:, 1]

    n_rows = dense_matrix.shape[0]
    if len(dia_spec_int) > n_rows:
        dia_spec_int = dia_spec_int[:n_rows]      
    elif len(dia_spec_int) < n_rows:
        dia_spec_int = np.append(dia_spec_int, [0]*(n_rows - dia_spec_int.shape[0]))

    # dia_spec_int = np.append(dia_spec_int,[0]*(dense_matrix.shape[0]-dia_spec_int.shape[0])) 

    return dense_matrix, dia_spec_int

def fit_mTRAQ_isotopes(spec,all_iso,mz_ppm):
    """
    
    ### spec is an ms1 spectrum
    #### all_iso is a list of the mTRAQ isotopes 
    e.g.
    [[Peak(mz=661.011960, intensity=0.352935, charge=3),
      Peak(mz=661.346233, intensity=0.335236, charge=3),
      Peak(mz=661.680188, intensity=0.192931, charge=3)]
     ...]
    
    mz_ppm is the relative mz tolerance e.g. 5.6e-6
    
    """
    ### spec is an ms1 spectrum
    #### all_iso is a list of the mTRAQ isotopes 
    
    

    flat = np.array([(p.mz, p.intensity) for iso in all_iso for p in iso], dtype=np.float64)
    ms1_iso_patterns = flat.reshape(len(all_iso), len(all_iso[0]), 2)

    
    dia_spectrum = np.array(spec.peak_list(), dtype=np.float64).T
    # dia_spectrum = np.array(np.array([spec.mz,spec.intens]), dtype=np.float64).T
    
    
    
    ### we only need to conseider the part of the spectrum that falls within the isotopic envelopes of the channels

    min_isotope = ms1_iso_patterns[:, :, 0].min() - 1
    max_isotope = ms1_iso_patterns[:, :, 0].max() + 1


    # dia_spectrum2 = dia_spectrum[np.logical_and(dia_spectrum[:,0]>min_isotope,dia_spectrum[:,0]<max_isotope)]

    mz = dia_spectrum[:, 0]
    lo = np.searchsorted(mz, min_isotope, side="right")
    hi = np.searchsorted(mz, max_isotope, side="left")
    dia_spectrum = dia_spectrum[lo:hi]

    
    #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)

    mz = dia_spectrum[:, 0]
    offsets = mz_ppm * mz
    centroid_breaks = np.empty(mz.size * 2, dtype=mz.dtype)
    centroid_breaks[:mz.size] = mz - offsets
    centroid_breaks[mz.size:] = mz + offsets
    centroid_breaks.sort()

 

    all_mz = ms1_iso_patterns[:,:,0].ravel()   # flatten all m/z values
    ref_coords_flat = np.searchsorted(centroid_breaks, all_mz)
    ref_coords = ref_coords_flat.reshape(ms1_iso_patterns.shape[0], -1)

    lib_peaks_matched = (ref_coords % 2 == 1).tolist()

    ref_spec_row_indices_split = [(((rc + 1) // 2) - 1).astype(np.int32)[mask]for rc, mask in zip(ref_coords, lib_peaks_matched)]

    num_lib_peaks_matched = np.fromiter((sum(i) for i in lib_peaks_matched), dtype=np.int32)

    ref_spec_col_indices_split = [np.array([idx]*i,dtype=np.int32) for idx,i in zip(range(len(ref_coords)),num_lib_peaks_matched)] 

    ref_spec_values_split = [ms1_iso_patterns[idx, :, 1][mask] for idx, mask in enumerate(lib_peaks_matched)]

    
    lib_coefficients = np.zeros(len(ref_coords))
    dia_spec_int = []
    matrix = []
    if any([i.size>0 for i in ref_spec_row_indices_split]):
        
        ref_spec_row_indices = np.concatenate(ref_spec_row_indices_split)
        ref_spec_col_indices = np.concatenate(ref_spec_col_indices_split)
        ref_spec_values = np.concatenate(ref_spec_values_split)
        # what peaks from the spectrum are matched by library peps
        unique_row_idxs = [int(i) for i in set(ref_spec_row_indices)]
        unique_row_idxs.sort()
        
        dia_spec_int = dia_spectrum[unique_row_idxs,1]
        
        lower_limit=1e-10
        last_row = max(unique_row_idxs)
        
        #### Type B
        not_dia_col_indices = np.arange(len(ref_coords))
        not_dia_row_indices = [last_row+1]*len(not_dia_col_indices)+not_dia_col_indices
        # not_dia_values2 = np.array([np.sum([ms1_iso_patterns[:,:,1][idx][peak_idx] for peak_idx in range(len(ms1_iso_patterns[:,:,1][idx])) if ref_coords[idx][peak_idx]%2==0])
        #                           for idx in range(len(ref_coords))])
                                  
        
        ref_coords_arr = np.array(ref_coords)
        ms1_intensities = ms1_iso_patterns[:, :, 1]  # shape: (num_peptides, num_peaks)

        mask = (ref_coords_arr % 2 == 0)
        not_dia_values = (ms1_intensities * mask).sum(axis=1)
         
        # sparse_row_indices2 = np.append(ref_spec_row_indices,not_dia_row_indices)
        # sparse_col_indices2 = np.append(ref_spec_col_indices,not_dia_col_indices)
        # sparse_values2 = np.append(ref_spec_values,not_dia_values)

        sparse_row_indices = np.concatenate([ref_spec_row_indices, not_dia_row_indices])
        sparse_col_indices = np.concatenate([ref_spec_col_indices, not_dia_col_indices])
        sparse_values = np.concatenate([ref_spec_values, not_dia_values])

        # #### Type C — each unmatched (non-DIA) peak gets its own matrix row
        # ref_coords_arr = np.array(ref_coords)
        # ms1_intensities = ms1_iso_patterns[:, :, 1]   # shape: (num_peptides, num_peaks)
        # mask = (ref_coords_arr % 2 == 0)              # True for unmatched peaks

        # # find (precursor_idx, peak_idx) pairs of unmatched peaks
        # not_dia_pairs = np.argwhere(mask)
        # if not_dia_pairs.size > 0:
        #     not_dia_values = ms1_intensities[mask]

        #     # give each unmatched peak its own row
        #     not_dia_row_indices = last_row + 1 + np.arange(len(not_dia_values))
        #     not_dia_col_indices = not_dia_pairs[:, 0]   # column = precursor index

        #     # combine with matched (DIA) peaks
        #     sparse_row_indices = np.concatenate([ref_spec_row_indices, not_dia_row_indices])
        #     sparse_col_indices = np.concatenate([ref_spec_col_indices, not_dia_col_indices])
        #     sparse_values = np.concatenate([ref_spec_values, not_dia_values])
        # else:
        #     # fallback if no unmatched peaks
        #     sparse_row_indices = ref_spec_row_indices
        #     sparse_col_indices = ref_spec_col_indices
        #     sparse_values = ref_spec_values
        # #end type C

        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows

        unique_vals, new_indices = np.unique(sparse_row_indices, return_inverse=True)
        sparse_row_indices = new_indices.astype(np.int32)
        
        max_row = np.max(sparse_row_indices)+1 # plus 1 for indexing
        max_col = np.max(sparse_col_indices)+1
        matrix = np.zeros((max_row,max_col))
        matrix[sparse_row_indices,sparse_col_indices] = sparse_values

        
        dia_spec_int = np.append(dia_spec_int,[0]*(matrix.shape[0]-dia_spec_int.shape[0])) 

        lib_coefficients, residuals = optimize.nnls(matrix, dia_spec_int)
    
    return lib_coefficients, dia_spec_int,  matrix








##iso functions
import re
mod_pattern = re.compile(r"\([A-z]+\:(\d+)\)")
from brainpy import isotopic_variants
from functools import reduce
from pyteomics import mass
unimods = mass.Unimod()

# @profile
def precursor_isotopes(sequence,charge,n_isotopes=2,decoys=True):    ##added decoys=True
    if decoys:  #added this conditional
        sequence = re.sub("Decoy_","",sequence)
    #split_seq = split_peptide(sequence)
    split_seq = parse_peptide(sequence)
    
    seq_comp = get_seq_comp(split_seq, "M")
    
    # if config.tag:
    #     tags = [t for aa in split_seq for t in re.findall(f"\(({config.tag.name}.*?)\)",aa)]
    #     if config.tag.channel_comp is not None and len(tags)>0:
    #             tag_comp = reduce(lambda x, y: x + y, [config.tag.channel_comp[re.findall(f"{config.tag.name}-(\d+)",t)[0]] for t in tags])
    #             seq_comp = tag_comp + seq_comp

    if config.tag:
        seq_comp = apply_tag_to_comp(split_seq, seq_comp)
        
            
    
    isotopes = isotopic_variants(seq_comp,
                                 npeaks=n_isotopes,
                                 charge = int(charge))
    
    return isotopes

def apply_tag_to_comp(split_seq, seq_comp):
    tag_name = config.tag.name
    prefix = f"({tag_name}-"
    prefix_len = len(prefix)
    channel_comp = config.tag.channel_comp

    comps = []
    for aa in split_seq:
        start = 0
        while True:
            i = aa.find(prefix, start)
            if i == -1:
                break
            j = aa.find(")", i)
            if j == -1:
                break
            num = aa[i+prefix_len:j]
            comp = channel_comp.get(num)
            if comp is not None:
                comps.append(comp)
            start = j + 1

    if comps:
        tag_comp = mass.Composition()
        for c in comps:
            tag_comp += c
        seq_comp += tag_comp

    return seq_comp

# @profile
def parse_peptide(seq):
    close_d = {"[": "]", "(": ")"}
    open_set = set(close_d.keys())
    close_set = set(close_d.values())
    
    new_seq = []
    current = ""
    s_idx = 0

    while s_idx < len(seq):
        s = seq[s_idx]

        if s in open_set:
            # Begin collecting the bracketed modification
            opener = s
            closer = close_d[opener]
            mod = s
            stack = [closer]
            s_idx += 1

            while s_idx < len(seq) and stack:
                c = seq[s_idx]
                mod += c

                if c in open_set:
                    stack.append(close_d[c])
                elif c in close_set:
                    if stack and c == stack[-1]:
                        stack.pop()
                s_idx += 1

            current += mod  # Append full modification to current letter

        elif s.isalpha():
            if current:
                new_seq.append(current)
            current = s
            s_idx += 1

        else:
            # If somehow an unexpected char, just add it
            current += s
            s_idx += 1

    if current:
        new_seq.append(current)

    return new_seq

# @profile
def get_seq_comp(split_seq,ion_type):
    
    stripped_seq = "".join([i[0] for i in split_seq]) ## assumes AA comes first before mods
    
    mods = [int(j) for i in split_seq for j in mod_pattern.findall(i) if len(i)>1]  #this line changed


    # tags = [t for aa in split_seq for t in re.findall("(\(.*?\))",aa)]
    seq_comp = mass.Composition(sequence=stripped_seq,ion_type=ion_type)
    for unimod_idx in mods:
        seq_comp += unimods.by_id(unimod_idx)["composition"]
    return seq_comp


if __name__ == "__main__":
    # import cProfile
    # cProfile.run("main()", r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\Line_Profiler_MS1_Cor_Channels\Cleaned_Up_profiling\output_after_iso.prof")
    main()