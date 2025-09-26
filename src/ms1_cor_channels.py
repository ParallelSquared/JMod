import numpy as np
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
import tqdm
from src.trace_fns import fit_mTRAQ_isotopes
from src.utils.misc_functions import  closest_ms1spec,np_pearson_cor
from scipy.interpolate import interp1d
from src.utils import misc_functions as mf
from src import iso_functions as iso

import warnings
from scipy.signal import find_peaks
import pickle
import math
from functools import reduce
from brainpy import isotopic_variants
import logging

import json
from pyteomics import mass

log_path = r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\Line_Profiler_MS1_Cor_Channels\log.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger = logging.getLogger("ms1_cor_logger")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_path, mode='w') 
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


min_int = 1e-3
config.opt_ms1_tol = 2.272908760635218e-06
# config.opt_rt_tol = 0.05
config.n_most_intense_features = 100000
# config.rt_tol = 0.05
# config.user_rt_tol = True
# config.plexDIA = True



##TODO re-enable logger messages before returning to Jmod

    


def main():
    print("config.tag:")
    #print(config.tag)
    print(config.opt_ms1_tol)
    print("Reading Pickle")
    pickle_path = r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\Line_Profiler_MS1_Cor_Channels\MS1_cor_inouts_final.pkl"
    # Load the pickle file
    with open(pickle_path, "rb") as f:
        pickle_dic = pickle.load(f)

    print(pickle_dic.keys())

    # Now you can access the data
    DIAspectra = pickle_dic.get("DIAspectra")
    fdc = pickle_dic.get("fdc")
    dc = pickle_dic.get("dc")
    mz_ppm = pickle_dic.get("mz_ppm")
    rt_tol = pickle_dic.get("rt_tol")
    mass_tag = pickle_dic.get("mass_tag")
    timeplex = pickle_dic.get("timeplex")
    group_p_corrs = pickle_dic.get("group_p_corrs")
    group_ms1_traces = pickle_dic.get("group_ms1_traces")
    group_ms2_traces = pickle_dic.get("group_ms2_traces")
    group_iso_ratios = pickle_dic.get("group_iso_ratios")
    group_keys = pickle_dic.get("group_keys")
    group_fitted = pickle_dic.get("group_fitted")

    print("pickle read")

    print(len(fdc))
    print(len(dc))

    group_p_corrs_out, group_ms1_traces_out, group_ms2_traces_out, group_iso_ratios_out, group_keys_out, group_fitted_out = ms1_cor_channels(DIAspectra, fdc, dc, mz_ppm, rt_tol, mass_tag, timeplex)

    global highest_ranked_specs_not_equal
    print(f"highest_ranked_specs_not_equal: {highest_ranked_specs_not_equal}")

    pickle_path = r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\Line_Profiler_MS1_Cor_Channels\new_MS1_cor_outs.pkl"
    new_pickle_dic = {
        "group_p_corrs_out": group_p_corrs_out,
        "group_ms1_traces_out": group_ms1_traces_out,
        "group_ms2_traces_out": group_ms2_traces_out,
        "group_iso_ratios_out": group_iso_ratios_out,
        "group_keys_out": group_keys_out,
        "group_fitted_out": group_fitted_out
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(new_pickle_dic, f)


    
def ms1_cor_channels(all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol,tag=None,timeplex=False):

    # print("config:", config.__dict__)
    config.tag=tag
    print("min_int:", min_int)
    print(config.tag)
    print(config.opt_ms1_tol)

    logger.info("Fitting tagged channels together")
    decoy_coeffs["untag_seq"] = [re.sub(f"(\({tag.name}-\d+\))?","",peptide) for peptide in decoy_coeffs["seq"]]
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
    
    ## array of scan numbers for ms1 and ms2 spectra
    ms1_spec_idxs = np.array([i.scan_num for i in ms1_spectra])
    ms2_spec_idxs = np.array([i.scan_num for i in ms2_spectra])
    
    ## get ms2 info for filtering
    bottom_of_window, top_of_window = np.array([i.ms1window for i in all_spectra.ms2scans]).T
    ms2_rt = np.array([i.RT for i in all_spectra.ms2scans])

    ## list of scan nums of the closest ms1 scan for each ms2 scan
    resp_ms1scans = [ms1_spec_idxs[closest_ms1spec(ms2_rt[i], ms1_rt)] for i in range(len(ms2_rt))]

    ## mapping of ms2 scan nums to ms1 scan nums
    ms2_ms1_scan_map = {spec.scan_num:resp_ms1scans[i] for i,spec in enumerate(all_spectra.ms2scans)}

    
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
    
    print(f"number of fdc: {len(filtered_decoy_coeffs)}")
    print(f"number of fdc groups: {len(fdc_group)}")
    number_of_times_ms2_vals_ran = 0 
    for key in tqdm.tqdm(list(fdc_group.groups)):
        prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel = get_seqs_and_mzs(fdc_group, timeplex, tag, key)

        all_scans, spectra_subset = minmax_spec_window(largest_coeff_scans, ms1_spec_idxs, ms1_spectra, all_spectra, window_half_width)
        
        ms1_traces, coeff_traces, is_traces, all_pearson, iso_ratios = ([] for _ in range(5))
        obs_ratios, group_iso, group_keys, all_channel_scans, interp_funcs, best_coeff = ([] for _ in range(6))

        for prec_mz,prec_seq in zip(prec_mzs,prec_seqs):
            number_of_times_ms2_vals_ran += 1
            
            ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, lesser_features_present, greater_features_present, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
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
    print(f"number_of_times_ms2_vals_ran: {number_of_times_ms2_vals_ran}")
    return all_group_pearson, all_ms1, all_coeff, all_iso, all_group_keys, all_fitted


def get_seqs_and_mzs(fdc_group, timeplex, tag, key):
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
    ## max and min of this list
    max_scan, min_scan = max(largest_coeff_scans), min(largest_coeff_scans)
    ms1_list_idx_min = list(ms1_spec_idxs).index(min_scan)
    ms1_list_idx_max = list(ms1_spec_idxs).index(max_scan)
    scans_each_side = np.array(ms1_spec_idxs)[np.arange(max(0,ms1_list_idx_min-window_half_width),min(len(ms1_spectra),ms1_list_idx_max+window_half_width+1))]
    all_scans = list(scans_each_side)

    spectra_subset = [all_spectra.get_by_idx(idx) for idx in all_scans]
    return all_scans, spectra_subset

global highest_ranked_specs_not_equal
highest_ranked_specs_not_equal = 0
def get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, lesser_features_present, greater_features_present, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs):
    ## keep decoys mathching to the correct MS1
    offset = config.decoy_mz_offset if "Decoy" in prec_seq else 0
    
    if timeplex:
        channel_key = (prec_seq,prec_z,time_channel)
    else:
        channel_key = (prec_seq,prec_z)
    
    ## create dummy 
    ms2_vals = {0:0}
    
    if channel_key in grouped_decoy_coeffs.groups:


        # new_data= grouped_decoy_coeffs.get_group(channel_key).copy()
        # ## rank order the coeffs in terms of goodness of fit
        # new_data.loc[:,"rank_score"] = np.sum([np.argsort(-new_data.loc[:,i]).argsort() for i in lesser_features_present],0)               
        # new_data.loc[:,"rank_score"] += np.sum([np.argsort(new_data[i]).argsort() for i in greater_features_present],0)
        # highest_ranked_spec = new_data.Ms1_spec_id.iloc[np.argmax(new_data.rank_score)]

        # new_data2= grouped_decoy_coeffs.get_group(channel_key).copy()
        # new_data2.loc[:,"rank_score"] = new_data2.loc[:, "coeff"]
        # highest_ranked_spec_coeff = new_data2.Ms1_spec_id.iloc[np.argmax(new_data2.rank_score)]

        # use coeff as rank score directly
        group = grouped_decoy_coeffs.get_group(channel_key)
        highest_ranked_spec = group.loc[group["coeff"].idxmax(), "Ms1_spec_id"]

        # global highest_ranked_specs_not_equal
        # if highest_ranked_spec != highest_ranked_spec_coeff2:
        #     highest_ranked_specs_not_equal += 1
            



        ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
        # prec_rt = new_data.rt.iloc[np.argmax(new_data.coeff)]
        prec_rt = group.rt.iloc[np.argmax(group.coeff)]
        ms2_window_bool = np.logical_and(prec_mz+offset>bottom_of_window,prec_mz+offset<top_of_window)
        
        # min_rt = np.minimum(prec_rt-rt_tol,np.min(new_data.rt)*.99)
        # max_rt = np.maximum(prec_rt+rt_tol,np.max(new_data.rt)*1.01)
        min_rt = np.minimum(prec_rt-rt_tol,np.min(group.rt)*.99)
        max_rt = np.maximum(prec_rt+rt_tol,np.max(group.rt)*1.01)
        ms2_rt_bool = np.logical_and(ms2_rt>=min_rt,ms2_rt<=max_rt)
        
        ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
        possible_ms2_scans = ms2_spec_idxs[ms2_bool]
        ms2_vals = {i:min_int for i in possible_ms2_scans}
    
        # for scan,c in zip(new_data["spec_id"],new_data["coeff"]):
        #     ms2_vals[scan]=c
        for scan,c in zip(group["spec_id"],group["coeff"]):
            ms2_vals[scan]=c
    else:
        highest_ranked_spec = None

    return ms2_vals, highest_ranked_spec, channel_key

@profile
def get_isotopes_and_vals(prec_seq, prec_z, num_iso, tag, all_scans, prec_mz, mz_ppm, spectra_subset, interp_func):
    ms1_vals = get_precursor_trace(prec_mz, mz_ppm, spectra_subset)
    isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
    prec_isotope_traces = get_isotope_traces(isotopes, mz_ppm, spectra_subset)
    all_ms1_vals, all_ms2_vals, all_iso_vals = unnamed_function(all_scans, prec_isotope_traces, interp_func, ms1_vals)

    return all_ms1_vals, all_ms2_vals, all_iso_vals, isotopes, interp_func

def build_ms2_interpolator(ms2_vals):
    return interp1d(list(ms2_vals.keys()), np.array(list(ms2_vals.values())), bounds_error=False)   

@profile
def get_precursor_trace(prec_mz, mz_ppm, spectra_subset):
    return {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec in spectra_subset}

@profile
def get_isotope_traces(isotopes, mz_ppm, spectra_subset):
    prec_isotope_traces=[]
    ## note: we have collected similar values for previous channel if the isotopic envelopes are overlapping. 
    ### However, in cases like diethlyation, isoptopes can differ by > 10 ppm #!!!Maybe investigate wider ppm tol for these cases?
    for isotope in isotopes[1:]:# we already have the monoisotopic trace
        iso_trace = {spec.scan_num:get_trace_int(spec, isotope.mz,rtol=mz_ppm) for spec in spectra_subset}
        prec_isotope_traces.append(iso_trace)
    
    return prec_isotope_traces

@profile
def compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag):
    isotopes = iso.precursor_isotopes(prec_seq,prec_z,num_iso)

    delta_mz = 0
    if tag.name in prec_seq:
        delta_mz = prec_mz-isotopes[0].mz
    for i in isotopes:
        i.mz+=delta_mz

    return isotopes

@profile
def unnamed_function(all_scans, prec_isotope_traces, interp_func, ms1_vals):
    all_ms1_vals = {i:min_int for i in all_scans}
    all_ms2_vals = {i:min_int for i in all_scans}
    all_iso_vals = [{i:min_int for i in all_scans} for _ in range(len(prec_isotope_traces))]
    
    for scan,c in zip(all_scans,interp_func(all_scans)):
        if scan in ms1_vals:
            all_ms1_vals[scan] = ms1_vals[scan]
            all_ms2_vals[scan] = c#f(scan)
        for iso_idx in range(len(prec_isotope_traces)):
            if scan in prec_isotope_traces[iso_idx]:
                all_iso_vals[iso_idx][scan] = prec_isotope_traces[iso_idx][scan]

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
                
    pred_coeff, obs_peaks, fit_matrix = fit_mTRAQ_isotopes(spec,group_iso,mz_ppm)
    if len(obs_peaks)==0:
        fit_cor = np.nan
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_cor = np_pearson_cor(np.sum(fit_matrix*pred_coeff,1),obs_peaks)
    
    return pred_coeff, obs_peaks, fit_matrix, fit_cor

def get_other_channels(prec,mz,tag):
    ### want to return m/z and seqs for all channels including this one
    
    ## identify what channel the current prec is in
    channels = re.findall(f"({tag.name}-\d+)",prec[0])
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

@profile
def get_trace_int(spec,mz,atol=0,rtol=0,base=min_int):
    ## speed up of above
    order_idx = np.searchsorted(spec.mz, mz)
    
    # Handle edge cases for indices at the bounds
    if order_idx == 0:
        closest_idx = 0
        mz_diff = spec.mz[0]-mz
    elif order_idx == len(spec.mz):
        closest_idx = len(spec.mz) - 1
        mz_diff = mz-spec.mz[-1]
    else:
        # Compare the closest values on both sides of the searchsorted index
        left_idx = order_idx - 1
        right_idx = order_idx
        
        # Find the closest value between the two neighboring indices
        left_diff = abs(spec.mz[left_idx] - mz)
        right_diff = abs(spec.mz[right_idx] - mz)
        if left_diff < right_diff:
            closest_idx = left_idx
            mz_diff = left_diff
        else:
            closest_idx = right_idx
            mz_diff = right_diff
    
#    mz_diff = abs(spec.mz[closest_idx] - mz)
    if mz_diff <= mz * rtol:  # Use the relative tolerance condition
        return spec.intens[closest_idx]

    return base

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





if __name__ == "__main__":
    main()