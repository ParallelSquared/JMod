#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import time
import re
from matplotlib_venn import venn2, venn3
import os
from scipy import stats

from miscFunctions import  closest_ms1spec, curve_fit, gaussian, createTolWindows,np_pearson_cor
from pyteomics import mass
from read_output import get_large_prec

from load_files import Spectrum, SpectrumFile
    
import config 

import dill
import load_files
import tqdm

import iso_functions as iso
import warnings

from scipy.interpolate import interp1d
from scipy import sparse, optimize
from scipy.signal import find_peaks
import sparse_nnls
import Jplot as jp

import multiprocessing

from functools import partial

import line_profiler

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


colours = ["tab:blue","tab:orange","tab:green","tab:red",
'tab:purple',
'tab:brown',
'tab:pink',
'tab:gray',
'tab:olive',
'tab:cyan']

linestyles = ["solid", "dotted", "dashed", "dashdot"]


min_int = 1e-3


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def most_dense_idx(x):
    arr = np.array(list(x))
    w=3
    kernel = np.ones(w)*.99
    kernel[int(np.floor(w/2))]=1
    return np.argmax(np.convolve(arr, kernel, 'same') / w)

def get_trace_int_old(spec,mz,atol=0,rtol=0,base=min_int):
    closest_idx = np.argmin(np.abs(spec.mz-mz))
    # order_idx = np.searchsorted(spec.mz,mz)
    # closest_idx = order_idx-1 if order_idx >= len(spec.mz) or abs(spec.mz[order_idx]-mz)>abs(spec.mz[order_idx-1]-mz) else order_idx-1
    # if np.isclose(spec.mz[closest_idx],mz,atol=atol,rtol=rtol):

    if (abs(spec.mz[closest_idx] - mz)/mz)<rtol:
        return spec.intens[closest_idx]
    else:
        return base
    
# def find_peaks(x):
#     arr = np.array(list(x))
#     max_idx = most_dense_idx(arr)
    
#     ### get front slope
#     idx = max_idx
#     val = arr[max_idx]
#     while idx>0:
#         new_val = arr[idx-1]
#         if new_val>=val:
#             break
#         else:
#             val=new_val
#             idx-=1
#     start_idx = idx
    
#     ### get back slope
#     idx = max_idx
#     val = arr[max_idx]
#     while idx<len(arr)-1:
#         new_val = arr[idx+1]
#         if new_val>=val:
#             break
#         else:
#             val=new_val
#             idx+=1
#     end_idx = idx
    
#     return start_idx, end_idx



    
    
    

# #@profile
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
    
    peaks,peak_attr= find_peaks(y,width=(None,None))
    
    ## if no peak, return the index of the max
    if len(peaks)==0:
        return x[np.argmax(y)],[x[0],x[-1]]
    
    peak_idxs = x[peaks]
    
    closest_idx = np.argmin(np.abs(idx-peak_idxs))

    peak_idx = peaks[closest_idx]
    peak_edge_idxs = [peak_attr["left_bases"][closest_idx],peak_attr["right_bases"][closest_idx]]
    
    return x[peak_idx],x[peak_edge_idxs]
    
# all_spectra = 1#DIAspectra

# filtered_decoy_coeffs = 3#fdc
# rt_tol = 0
# mz_ppm = 10

def ms1_cor(all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol,timeplex=False):
    print("Fitting precursors individually")
    num_iso = config.num_iso_ms1
    window_half_width = 10
    
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
    else:
        grouped_decoy_coeffs = decoy_coeffs.groupby(["seq","z"])
    
    ms1_traces = []
    coeff_traces = []
    is_traces = []
    all_pearson = []
    iso_ratios = []
    obs_ratios = []
    
    for fdc_idx in tqdm.tqdm(range(0,len(filtered_decoy_coeffs))):#range(26010,26030):# 
        prec_seq=filtered_decoy_coeffs.iloc[fdc_idx]["seq"]
        prec_z = filtered_decoy_coeffs.iloc[fdc_idx]["z"]
        prec_mz = filtered_decoy_coeffs["mz"][fdc_idx]
        prec_rt = filtered_decoy_coeffs.iloc[fdc_idx]["rt"]
        
        offset = config.decoy_mz_offset if "Decoy" in prec_seq else 0
        ms2_window_bool = np.logical_and(prec_mz+offset>bottom_of_window,prec_mz+offset<top_of_window)
        ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
        ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
        possible_ms2_scans = ms2_spec_idxs[ms2_bool]
        
        if timeplex:
            time_channel = filtered_decoy_coeffs["time_channel"][fdc_idx]
            new_data= grouped_decoy_coeffs.get_group((prec_seq,prec_z,time_channel))
        else:
            new_data= grouped_decoy_coeffs.get_group((prec_seq,prec_z))
        
        rt_bool = np.abs(ms1_rt-prec_rt)<rt_tol
        
        # ms1_scans = ms1_spec_idxs[rt_bool]
        # ms1_scans_from_coeff = new_data["Ms1_spec_id"]
        # all_scans = sorted(set(ms1_scans).union(set(ms1_scans_from_coeff)))
        # all_scans = [ms2_ms1_scan_map[i] for i in possible_ms2_scans]
        # all_scans = sorted(set(ms1_scans_from_coeff))
        
        
        ### get the ms1 index of the largest coefficient and n scans nearest to it
        ms1_index_of_max = filtered_decoy_coeffs.iloc[fdc_idx]["Ms1_spec_id"]
        ms1_list_idx = list(ms1_spec_idxs).index(ms1_index_of_max)
        scans_each_side = np.array(ms1_spec_idxs)[np.arange(max(0,ms1_list_idx-window_half_width),min(len(ms1_spectra),ms1_list_idx+window_half_width+1))]
        all_scans_rt = np.array(ms1_rt)[np.arange(max(0,ms1_list_idx-window_half_width),min(len(ms1_spectra),ms1_list_idx+window_half_width+1))]
        all_scans = list(scans_each_side)
        spectra_subset = [all_spectra.get_by_idx(idx) for idx in all_scans]
        
        
        # ms2_vals = {i:j for i,j in zip(new_data["Ms1_spec_id"],new_data["coeff"])}
        ms2_vals = {i:min_int for i in possible_ms2_scans}
        for scan,c in zip(new_data["spec_id"],new_data["coeff"]):
            ms2_vals[scan]=c
        
        
        f = interp1d(list(ms2_vals.keys()), list(ms2_vals.values()), bounds_error=False)
                
            
        # ms1_vals = {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
        ms1_vals = {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec in spectra_subset}
        
        
        prec_isotope_traces = []
        isotopes = iso.precursor_isotopes(prec_seq,prec_z,num_iso)
        delta_mz = 0
        if "mTRAQ" in prec_seq:
            delta_mz = prec_mz-isotopes[0].mz
        # iso_ratios.append([i.intensity for i in isotopes])
        for isotope in isotopes[1:]:# we already have the monoisotopic trace
            iso_trace = {spec.scan_num:get_trace_int(spec, isotope.mz+delta_mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
            prec_isotope_traces.append(iso_trace)
            
        all_ms1_vals = {i:min_int for i in all_scans}
        all_ms2_vals = {i:min_int for i in all_scans}
        all_iso_vals = [{i:min_int for i in all_scans} for _ in range(len(prec_isotope_traces))]
        
        for scan in all_scans:
            if scan in ms1_vals:
                all_ms1_vals[scan] = ms1_vals[scan]
                all_ms2_vals[scan] = f(scan)
            # if scan in ms2_vals:
            #     all_ms2_vals[scan] = ms2_vals[scan]
            for iso_idx in range(len(prec_isotope_traces)):
                if scan in prec_isotope_traces[iso_idx]:
                    all_iso_vals[iso_idx][scan] = prec_isotope_traces[iso_idx][scan]
                    
        ## use monoiso ms1 prec mz to find the elution ms1 peak
        ms1_peak_idx,ms1_peak_edge_idxs = get_ms1_peak(list(all_ms1_vals.keys()), list(all_ms1_vals.values()), ms1_index_of_max)
        
        ## redefine all_scans to keep only thoe from the above peak
        all_scans = all_scans[all_scans.index(ms1_peak_edge_idxs[0]):all_scans.index(ms1_peak_edge_idxs[1])+1]
        all_ms1_vals = {i:all_ms1_vals[i] for i in all_scans}
        all_iso_vals = [{i:iso_vals[i] for i in all_scans} for iso_vals in all_iso_vals]
        all_ms2_vals = {i:all_ms2_vals[i] for i in all_scans}
        
        ms1_traces.append([all_ms1_vals,*all_iso_vals])
        coeff_traces.append(all_ms2_vals)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec_pearsons = [0]*num_iso
            if len(all_scans) > 2:
                try:
                    # Get the arrays that will be used for correlation
                    ms2_values = np.array(list(all_ms2_vals.values()), dtype=float)
                    
                    # For debugging, identify any problematic values before calculation
                    if not np.all(np.isfinite(ms2_values)):
                        bad_indices = np.where(~np.isfinite(ms2_values))[0]
                        bad_scans = list(all_ms2_vals.keys())[bad_indices[0]]
                        print(f"Warning: Non-finite MS2 value detected at scan {bad_scans}")
                    
                    # Calculate correlations with validity check
                    spec_pearsons = []
                    for i in [all_ms1_vals, *all_iso_vals]:
                        ms1_values = np.array(list(i.values()), dtype=float)
                        
                        # Only use finite values for correlation
                        valid_mask = np.isfinite(ms2_values) & np.isfinite(ms1_values)
                        if np.sum(valid_mask) >= 2:
                            result = stats.pearsonr(ms2_values[valid_mask], ms1_values[valid_mask]).statistic
                            spec_pearsons.append(0.0 if not np.isfinite(result) else result)
                        else:
                            spec_pearsons.append(0.0)
                except Exception as e:
                    # Fallback with debug info
                    print(f"Error in Pearson calculation: {str(e)}")
                    spec_pearsons = [0]*num_iso
            # all_pearson.append(stats.pearsonr(list(all_ms2_vals.values()),list(all_ms1_vals.values())).statistic)
            all_pearson.append(spec_pearsons)
            
            ms1_spec_idx = filtered_decoy_coeffs.iloc[fdc_idx]["Ms1_spec_id"]
            ms1_spec_idx = all_scans[np.argmax(list(all_ms2_vals.values()))]
            
            theoretical_pattern = [i.intensity for i in isotopes]
            obs_pattern = [all_ms1_vals[ms1_spec_idx],*[iso_trace[ms1_spec_idx] for iso_trace in all_iso_vals]]
            # obs_ratios.append(obs_pattern)
            iso_ratios.append([stats.pearsonr(theoretical_pattern,obs_pattern),theoretical_pattern,obs_pattern])
        
          
    return all_pearson, ms1_traces, coeff_traces, iso_ratios

"""

def ms1_cor_channels(all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol):
    ### like above but search all the channels together
    
    

    ### remove tags so we can group together
    decoy_coeffs["untag_seq"] = [re.sub("(\(.*?\))?","",peptide) for peptide in decoy_coeffs["seq"]]
    filtered_decoy_coeffs["untag_seq"] = [re.sub("(\(mTRAQ-\d\))?","",peptide) for peptide in filtered_decoy_coeffs["seq"]]
    untag_grouped_decoy_coeffs = decoy_coeffs.groupby(["untag_seq","z"])
    grouped_decoy_coeffs = decoy_coeffs.groupby(["seq","z"])
    
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

    
    ms1_traces = []
    coeff_traces = []
    is_traces = []
    all_pearson = []
    iso_ratios = []
    obs_ratios = []
    
    for fdc_idx in [46190,48890,34574]:#tqdm.tqdm(range(0,len(filtered_decoy_coeffs))):#range(26010,26030):# 
    
    
    
        prec_seq=filtered_decoy_coeffs.iloc[fdc_idx]["seq"]
        untag_seq = filtered_decoy_coeffs.iloc[fdc_idx]["untag_seq"]
        prec_z = filtered_decoy_coeffs.iloc[fdc_idx]["z"]
        prec_mz = filtered_decoy_coeffs["mz"][fdc_idx]
        print(prec_mz,prec_z)
        prec_rt = filtered_decoy_coeffs.iloc[fdc_idx]["rt"]
    
        ms2_window_bool = np.logical_and(prec_mz>bottom_of_window,prec_mz<top_of_window)
        ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
        ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
        possible_ms2_scans = ms2_spec_idxs[ms2_bool]
        
        
        new_data= grouped_decoy_coeffs.get_group((prec_seq,prec_z))
        
        tag_group = untag_grouped_decoy_coeffs.get_group((untag_seq,prec_z))
        
        rt_bool = np.abs(ms1_rt-prec_rt)<rt_tol
        
        ms1_scans = ms1_spec_idxs[rt_bool]
        ms1_scans_from_coeff = new_data["Ms1_spec_id"]
        all_scans = sorted(set(ms1_scans).union(set(ms1_scans_from_coeff)))
        all_scans = [ms2_ms1_scan_map[i] for i in possible_ms2_scans]
        ms1_vals = {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
        ms2_vals = {i:j for i,j in zip(new_data["Ms1_spec_id"],new_data["coeff"])}
        
        prec_isotope_traces = []
        isotopes = iso.precursor_isotopes(prec_seq,prec_z,3)
        delta_mz = 0
        if "mTRAQ" in prec_seq:
            delta_mz = prec_mz-isotopes[0].mz
        # iso_ratios.append([i.intensity for i in isotopes])
        for isotope in isotopes[1:]:# we already have the monoisotopic trace
            iso_trace = {spec.scan_num:get_trace_int(spec, isotope.mz+delta_mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
            prec_isotope_traces.append(iso_trace)
            
        all_ms1_vals = {i:min_int for i in all_scans}
        all_ms2_vals = {i:min_int for i in all_scans}
        all_iso_vals = [{i:min_int for i in all_scans} for _ in range(len(prec_isotope_traces))]
        
        for scan in all_scans:
            if scan in ms1_vals:
                all_ms1_vals[scan] = ms1_vals[scan]
            if scan in ms2_vals:
                all_ms2_vals[scan] = ms2_vals[scan]
            for iso_idx in range(len(prec_isotope_traces)):
                if scan in prec_isotope_traces[iso_idx]:
                    all_iso_vals[iso_idx][scan] = prec_isotope_traces[iso_idx][scan]
                    
        def func(x): return(np.array(list(x)))
        plt.plot(all_ms2_vals.keys(),func(all_ms2_vals.values()),label="Coeffs")
        plt.plot(all_ms1_vals.keys(),func(all_ms1_vals.values()),label="Monoiso")
        plt.plot(all_iso_vals[0].keys(),func(all_iso_vals[0].values()),label="1st Iso")
        plt.plot(all_iso_vals[1].keys(),func(all_iso_vals[1].values()),label="2nd Iso")
        # plt.scatter(all_ms2_vals.keys(),func(all_ms2_vals.values()))
        # plt.scatter(all_ms1_vals.keys(),func(all_ms1_vals.values()))
        plt.legend()
        plt.xlabel("Scan Number")
        plt.ylabel("Intensity")
        plt.xlim(24250,24750)         
        
        ms1_traces.append([all_ms1_vals,*all_iso_vals])
        coeff_traces.append(all_ms2_vals)
        
        spec_pearsons = [stats.pearsonr(list(all_ms2_vals.values()),list(i.values())).statistic for i in [all_ms1_vals,*all_iso_vals]]
        # all_pearson.append(stats.pearsonr(list(all_ms2_vals.values()),list(all_ms1_vals.values())).statistic)
        all_pearson.append(spec_pearsons)
        
        ms1_spec_idx = filtered_decoy_coeffs.iloc[fdc_idx]["Ms1_spec_id"]
        ms1_spec_idx = all_scans[np.argmax(list(all_ms2_vals.values()))]
        
        theoretical_pattern = [i.intensity for i in isotopes]
        obs_pattern = [all_ms1_vals[ms1_spec_idx],*[iso_trace[ms1_spec_idx] for iso_trace in all_iso_vals]]
        # obs_ratios.append(obs_pattern)
        iso_ratios.append([stats.pearsonr(theoretical_pattern,obs_pattern),theoretical_pattern,obs_pattern])
        
    return all_pearson, ms1_traces, coeff_traces, iso_ratios


"""

# @profile
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
    
    
    ms1_iso_patterns = np.array([[[i.mz,i.intensity] for i in isotope] for isotope in all_iso])
    
    dia_spectrum = np.stack(spec.peak_list(),1)
    
    
    ### we only need to conseider the part of the spectrum that falls within the isotopic envelopes of the channels
    min_isotope = min([j.mz for i in all_iso for j in i])-1
    max_isotope = max([j.mz for i in all_iso for j in i])+1
    dia_spectrum = dia_spectrum[np.logical_and(dia_spectrum[:,0]>min_isotope,dia_spectrum[:,0]<max_isotope)]
    
    merged_coords_idxs = np.searchsorted(dia_spectrum[:,0]+mz_ppm*dia_spectrum[:,0],dia_spectrum[:,0])
    
    # what are the first mz of these peak groups
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs),0]
    merged_intensities = np.zeros(len((merged_coords_idxs)))
    for j,val in zip(merged_coords_idxs,dia_spectrum[:,1]):
        merged_intensities[j]+=val
    merged_intensities = merged_intensities[merged_intensities!=0]
    
    #update spectrum to new values (note mz remains first in group as this will eventually be rounded)
    dia_spectrum = np.array((merged_coords,merged_intensities)).transpose()
    # print(dia_spectrum)
    
    #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)
    centroid_breaks = np.concatenate((dia_spectrum[:,0]-mz_ppm*dia_spectrum[:,0],dia_spectrum[:,0]+mz_ppm*dia_spectrum[:,0]))
    centroid_breaks = np.sort(centroid_breaks)
    bin_centers = np.mean(np.stack((centroid_breaks[::2],centroid_breaks[1::2]),1),1)
    
    ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in ms1_iso_patterns]
    
    lib_peaks_matched = [j%2==1 for j in ref_coords]
    ref_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(ref_coords,lib_peaks_matched)] # NB these are floats
    num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched]) #f1
    ref_spec_col_indices_split = [np.array([idx]*i,dtype=np.int32) for idx,i in zip(range(len(ref_coords)),num_lib_peaks_matched)] 
    ref_spec_values_split = [i[:,1][j] for i,j in zip(ms1_iso_patterns,lib_peaks_matched)]
    
    
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
        not_dia_values = np.array([np.sum([ms1_iso_patterns[:,:,1][idx][peak_idx] for peak_idx in range(len(ms1_iso_patterns[:,:,1][idx])) if ref_coords[idx][peak_idx]%2==0])
                                  for idx in range(len(ref_coords))])
        
        
        
        sparse_row_indices = np.append(ref_spec_row_indices,not_dia_row_indices)
        sparse_col_indices = np.append(ref_spec_col_indices,not_dia_col_indices)
        sparse_values = np.append(ref_spec_values,not_dia_values)
        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows
        sparse_row_indices = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1
        
        max_row = np.max(sparse_row_indices)+1 # plus 1 for indexing
        max_col = np.max(sparse_col_indices)+1
        matrix = np.zeros((max_row,max_col))
        matrix[sparse_row_indices,sparse_col_indices] = sparse_values
        
        dia_spec_int = np.append(dia_spec_int,[0]*(matrix.shape[0]-dia_spec_int.shape[0])) 
        
        # Generate sparse matrix from data
        # sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))
        # dia_spec_int = np.append(dia_spec_int,[0]*(sparse_lib_matrix.shape[0]-dia_spec_int.shape[0])) 
        
        # Fit lib spectra to observed spectra
        # fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix,dia_spec_int,{"show_progress":False})
        # lib_coefficients = np.array(fit_results['x']).flatten()
        
        ### NOT Non-Negative!!
        # matrix = np.array(sparse_lib_matrix.todense())
        # lib_coefficients = np.linalg.lstsq(matrix, dia_spec_int)[0]
        lib_coefficients, residuals = optimize.nnls(matrix, dia_spec_int)
        
    return lib_coefficients, dia_spec_int,  matrix

    # func=np.array
    # plt.vlines(np.arange(len(dia_spec_int)),0,func(dia_spec_int),linewidths=15)
    # place_holder = np.zeros_like(dia_spec_int)
    # for i in range(len(ref_coords)):
    #     _bool = sparse_col_indices==i
    #     plt.vlines(sparse_row_indices[_bool],
    #                func(place_holder[sparse_row_indices[_bool]]),
    #                func((sparse_values[_bool]*lib_coefficients[i])+place_holder[sparse_row_indices[_bool]]),
    #                linewidths=10,colors=jp.colours[i+1])
    #     place_holder[sparse_row_indices[_bool]] += (sparse_values[_bool]*lib_coefficients[i])


   
# ms1_spec_idx = 24409

# spec_idxs= [24319,
# 24349,
# 24379,
# 24409,
# 24439,
# 24469,
# 24499,
# 24529]
"""
all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol=all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol

### remove tags so we can group together
decoy_coeffs["untag_seq"] = [re.sub("(\(.*?\))?","",peptide) for peptide in decoy_coeffs["seq"]]
filtered_decoy_coeffs["untag_seq"] = [re.sub("(\(mTRAQ-\d\))?","",peptide) for peptide in filtered_decoy_coeffs["seq"]]
untag_grouped_decoy_coeffs = decoy_coeffs.groupby(["untag_seq","z"])
grouped_decoy_coeffs = decoy_coeffs.groupby(["seq","z"])

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




# prec_mzs = [661.01196,662.34766,663.68335]
# prec_seqs = ['S(mTRAQ-0)IVPSGASTGVHEALEMR','S(mTRAQ-4)IVPSGASTGVHEALEMR','S(mTRAQ-8)IVPSGASTGVHEALEMR']
# prec_z = 3
# ms1_spec_idx = 24409

### group fdc by untag and z
fdc_group = filtered_decoy_coeffs.groupby(["untag_seq","z"])

key = ('SIVPSGASTGVHEALEMR',3)
key = ("VLPELQGK",2)
key = ("TASGNIIPSSTGAAK",3)

# """

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

# @profile
def ms1_cor_channels(all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol,tag=None,timeplex=False):
    print("Fitting tagged channels together")
    
    decoy_coeffs["untag_seq"] = [re.sub(f"(\({tag.name}-\d+\))?","",peptide) for peptide in decoy_coeffs["seq"]]
    decoy_coeffs["untag_prec"] = ["_".join([i[0],str(int(i[1]))]) for i in zip(decoy_coeffs["untag_seq"],decoy_coeffs["z"])]
    
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
        
    # grouped_decoy_coeffs = decoy_coeffs.groupby(["seq","z"])
    # fdc_group = filtered_decoy_coeffs.groupby(["untag_seq","z"])

    all_ms1= []
    all_coeff = []
    all_iso = []
    all_group_pearson = []
    all_trace = []
    all_fitted = []
    all_group_keys = []
    
    all_scans_len = []
    for key in tqdm.tqdm(list(fdc_group.groups)):
        tag_group = fdc_group.get_group(key)
        prec_mzs = tag_group["mz"]
        prec_seqs = tag_group["seq"]
        prec_z = key[1]
        if timeplex:
            time_channel = key[2]
        largest_id = np.argmax(tag_group["coeff"])
        top_ms1_spec_idx = list(tag_group["Ms1_spec_id"])[largest_id]
        prec_rt = list(tag_group["rt"])[largest_id]
        spec_idx_of_largest =list(tag_group["spec_id"])[largest_id]
        rt_bool = np.abs(ms1_rt-prec_rt)<rt_tol
        # print(prec_rt,rt_tol)
        
        window_mz = tag_group["window_mz"].iloc[largest_id]
        # spec = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]
        # mz_ppm=mz_ppm
        # ms1_spectra=ms1_spectra
        # rt_bool=rt_bool
        num_iso = config.num_iso_ms1
        window_half_width = 10
        # assert spec.scan_num == ms1_spec_idx
        
        ### search for all channels always:
        channel_dict = get_other_channels((prec_seqs.iloc[largest_id],prec_z), prec_mzs.iloc[largest_id], tag)
        prec_seqs,prec_mzs = tuple(zip(*channel_dict.values()))
    
    
        
        # ms2_window_bool = np.logical_and(window_mz>bottom_of_window,window_mz<top_of_window)
        # ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
        # ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
        # # print(sum(ms2_bool))
        # possible_ms2_scans = ms2_spec_idxs[ms2_bool]
        # all_scans = [ms2_ms1_scan_map[i] for i in possible_ms2_scans]
        # ## n scans either side of max
        # idx_of_max =all_scans.index(top_ms1_spec_idx)
        # scans_each_side = np.array(all_scans)[np.arange(max(0,idx_of_max-window_half_width),min(len(all_scans),idx_of_max+window_half_width+1))]
        
        
        if timeplex:
            ## scans where coeff>1
            coeff_scans = decoy_coeffs["Ms1_spec_id"][np.logical_and.reduce((decoy_coeffs["untag_seq"]==key[0],
                                                                             decoy_coeffs["z"]==key[1],
                                                                             decoy_coeffs["time_channel"]==key[2]))]
            ## join
            # all_scans = sorted(list(set(list(scans_each_side)+list(coeff_scans))))
            ## scans of max of each
            largest_coeff_scans = list(filtered_decoy_coeffs["Ms1_spec_id"][np.logical_and.reduce((filtered_decoy_coeffs["untag_seq"]==key[0],
                                                                                                  filtered_decoy_coeffs["z"]==key[1],
                                                                                                  filtered_decoy_coeffs["time_channel"]==key[2]))])
            
        else:
            ## scans where coeff>1
            coeff_scans = decoy_coeffs["Ms1_spec_id"][np.logical_and(decoy_coeffs["untag_seq"]==key[0],decoy_coeffs["z"]==key[1])] 
            ## join
            # all_scans = sorted(list(set(list(scans_each_side)+list(coeff_scans))))
            ## scans of max of each
            largest_coeff_scans = list(filtered_decoy_coeffs["Ms1_spec_id"][np.logical_and(filtered_decoy_coeffs["untag_seq"]==key[0],
                                                                                           filtered_decoy_coeffs["z"]==key[1])])
        
        ## max and min of this list
        max_scan, min_scan = max(largest_coeff_scans), min(largest_coeff_scans)
        ms1_list_idx_min = list(ms1_spec_idxs).index(min_scan)
        ms1_list_idx_max = list(ms1_spec_idxs).index(max_scan)
        scans_each_side = np.array(ms1_spec_idxs)[np.arange(max(0,ms1_list_idx_min-window_half_width),min(len(ms1_spectra),ms1_list_idx_max+window_half_width+1))]
        all_scans = list(scans_each_side)
    
        spectra_subset = [all_spectra.get_by_idx(idx) for idx in all_scans]
        
        # all_scans_len.append([len(all_scans),len(all_scans2)])
        # print(len(all_scans))
        ms1_traces = []
        coeff_traces = []
        is_traces = []
        all_pearson = []
        iso_ratios = []
        obs_ratios = []
        group_iso = []
        group_keys = [] ## collect to ensure we match them up correctly
        all_channel_scans = []
        for prec_mz,prec_seq in zip(prec_mzs,prec_seqs):
            
            ## keep decoys mathching to the correct MS1
            offset = config.decoy_mz_offset if "Decoy" in prec_seq else 0
            
            if timeplex:
                channel_key = (prec_seq,prec_z,time_channel)
            else:
                channel_key = (prec_seq,prec_z)
                
            group_keys.append(channel_key)
            
            ## create dummy 
            ms2_vals = {0:0}
            
            if channel_key in grouped_decoy_coeffs.groups:
                new_data= grouped_decoy_coeffs.get_group(channel_key)   
                ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
                prec_rt = new_data.rt.iloc[np.argmax(new_data.coeff)]
                ms2_window_bool = np.logical_and(prec_mz+offset>bottom_of_window,prec_mz+offset<top_of_window)
                
                min_rt = np.minimum(prec_rt-rt_tol,np.min(new_data.rt)*.99)
                max_rt = np.maximum(prec_rt+rt_tol,np.max(new_data.rt)*1.01)
                ms2_rt_bool = np.logical_and(ms2_rt>=min_rt,ms2_rt<=max_rt)
                
                ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
                # print(sum(ms2_bool))
                possible_ms2_scans = ms2_spec_idxs[ms2_bool]
                ms2_vals = {i:min_int for i in possible_ms2_scans}
            
                # ms2_vals = {i:j for i,j in zip(new_data["Ms1_spec_id"],new_data["coeff"])}
                # ms2_vals = {i:j for i,j in zip(new_data["spec_id"],new_data["coeff"])}
                # plt.plot(ms2_vals.keys(),np.log10(list(ms2_vals.values())))
                for scan,c in zip(new_data["spec_id"],new_data["coeff"]):
                    ms2_vals[scan]=c
            # else:
            #     ms2_vals = {}
                
            
            f = interp1d(list(ms2_vals.keys()), list(ms2_vals.values()), bounds_error=False)
                
            
            
            # ms1_vals = {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
            ms1_vals = {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec in spectra_subset}
            
            
            isotopes = iso.precursor_isotopes(prec_seq,prec_z,num_iso)
            
            delta_mz = 0
            if tag.name in prec_seq:
                delta_mz = prec_mz-isotopes[0].mz
            for i  in isotopes:
                i.mz+=delta_mz
                
            group_iso.append(isotopes)
            
            prec_isotope_traces=[]
            # iso_ratios.append([i.intensity for i in isotopes])
            ## note: we have collected similar values for previous channel if the isotopic envelopes are overlapping. 
            ### However, in cases like diethlyation, isoptopes can differ by > 10 ppm #!!!Maybe investigate wider ppm tol for these cases?
            for isotope in isotopes[1:]:# we already have the monoisotopic trace
                # iso_trace = {spec.scan_num:get_trace_int(spec, isotope.mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
                iso_trace = {spec.scan_num:get_trace_int(spec, isotope.mz,rtol=mz_ppm) for spec in spectra_subset}
                prec_isotope_traces.append(iso_trace)
             
            
            all_ms1_vals = {i:min_int for i in all_scans}
            all_ms2_vals = {i:min_int for i in all_scans}
            all_iso_vals = [{i:min_int for i in all_scans} for _ in range(len(prec_isotope_traces))]
            
            for scan,c in zip(all_scans,f(all_scans)):
                if scan in ms1_vals:
                    all_ms1_vals[scan] = ms1_vals[scan]
                    all_ms2_vals[scan] = c#f(scan)
                # if scan in ms2_vals:
                    # all_ms2_vals[scan] = ms2_vals[scan]
                for iso_idx in range(len(prec_isotope_traces)):
                    if scan in prec_isotope_traces[iso_idx]:
                        all_iso_vals[iso_idx][scan] = prec_isotope_traces[iso_idx][scan]
                        
            # def func(x): return(np.array(list(x)))
            # plt.plot(all_ms2_vals.keys(),func(all_ms2_vals.values()),label="Coeffs")
            # plt.plot(all_ms1_vals.keys(),func(all_ms1_vals.values()),label="Monoiso")
            # plt.plot(all_iso_vals[0].keys(),func(all_iso_vals[0].values()),label="1st Iso")
            # plt.plot(all_iso_vals[1].keys(),func(all_iso_vals[1].values()),label="2nd Iso") 
            # plt.plot(all_iso_vals[2].keys(),func(all_iso_vals[2].values()),label="2nd Iso") 
            ###plt.plot(ms2_vals.keys(),func(ms2_vals.values()),label="Coeffs")
            
            ## use monoiso ms1 prec mz to find the elution ms1 peak
            if ms2_vals=={0:0}:
                ms1_index_of_max = top_ms1_spec_idx ## should I just use the max of MS1???? Need to look into again
            else:
                # ms1_index_of_max = new_data.Ms1_spec_id.iloc[np.argmax(new_data.coeff)]
                ms1_keys = list(all_ms1_vals.keys())
                ms1_index_of_max = ms1_keys[np.argmax(f(ms1_keys))]
                
            ms1_peak_idx,ms1_peak_edge_idxs = get_ms1_peak(list(all_ms1_vals.keys()), list(all_ms1_vals.values()), ms1_index_of_max)
            
            ## redefine all_scans to keep only thoe from the above peak
            channel_scans = all_scans[all_scans.index(ms1_peak_edge_idxs[0]):all_scans.index(ms1_peak_edge_idxs[1])+1]
            all_ms1_vals = {i:all_ms1_vals[i] for i in channel_scans}
            all_iso_vals = [{i:iso_vals[i] for i in channel_scans} for iso_vals in all_iso_vals]
            all_ms2_vals = {i:all_ms2_vals[i] for i in channel_scans}
            
            all_channel_scans.append(channel_scans)
            ms1_traces.append([all_ms1_vals,*all_iso_vals])
            coeff_traces.append(all_ms2_vals)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec_pearsons = [np_pearson_cor(list(all_ms2_vals.values()),list(i.values())).statistic for i in [all_ms1_vals,*all_iso_vals[:config.num_iso_r]]]
                # all_pearson.append(stats.pearsonr(list(all_ms2_vals.values()),list(all_ms1_vals.values())).statistic)
                all_pearson.append(spec_pearsons)
                
                # ms1_spec_idx = filtered_decoy_coeffs.iloc[fdc_idx]["Ms1_spec_id"]
                ms1_spec_idx = channel_scans[np.argmax(list(all_ms2_vals.values()))]
                
                theoretical_pattern = [i.intensity for i in isotopes]
                obs_pattern = [all_ms1_vals[ms1_spec_idx],*[iso_trace[ms1_spec_idx] for iso_trace in all_iso_vals]]
                # obs_ratios.append(obs_pattern)
                iso_ratios.append([np_pearson_cor(theoretical_pattern,obs_pattern),theoretical_pattern,obs_pattern])
        
        # def func(x): return(np.array(list(x)))
        # for d,c in zip(coeff_traces,colours):
        #     plt.plot(d.keys(),func(d.values()),color=c,label="Coeffs")
        # for d,c in zip(ms1_traces,colours):
        #     plt.plot(d[0].keys(),func(d[0].values()),color=c,label="Coeffs",linestyle="--")
        # plt.xlim(35000,36300)
            
        ### need to reduce the number of spectra we fit to
        idx_of_max =all_scans.index(top_ms1_spec_idx)
        scans_to_search = np.array(all_scans)[np.arange(max(0,idx_of_max-window_half_width),min(len(all_scans),idx_of_max+window_half_width+1))]
        ### fit to those from each channel
        scans_to_search = np.sort(np.unique(np.concatenate(all_channel_scans)))
        vals = []
        group_pred = []
        group_obs_peaks=[]
        group_matrices =[]
        group_fit_cor =[]
        ### for ms1_spec_idx in all_scans:
        for ms1_spec_idx in scans_to_search:
            spec = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]
            
            pred_coeff, obs_peaks, fit_matrix = fit_mTRAQ_isotopes(spec,group_iso,mz_ppm)
            if len(obs_peaks)==0:
                fit_cor = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_cor = np_pearson_cor(np.sum(fit_matrix*pred_coeff,1),obs_peaks)
            vals.append([pred_coeff,obs_peaks,fit_matrix,fit_cor])
            
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
    return all_group_pearson, all_ms1, all_coeff, all_iso, all_group_keys, all_fitted









###########################################################################
###########################################################################
###########################################################################
### functionalize the above for multiprocessing
###########################################################################
###########################################################################
###########################################################################

# @profile
def collect_channel(prec_mz,prec_seq,prec_z,grouped_decoy_coeffs, all_spectra ,all_scans,tag,
                    rt_bool,mz_ppm,num_iso,window_edges,ms2_rt,ms2_spec_idxs,rt_tol):
    
    ms1_spectra = all_spectra.ms1scans
    
    
    # ms2_window_bool = np.logical_and(prec_mz>window_edges[:,0],prec_mz<window_edges[:,1])
    # ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
    # # print(sum(ms2_bool))
    # possible_ms2_scans = ms2_spec_idxs[ms2_bool]
    
    ## keep decoys mathching to the correct MS1
    offset = config.decoy_mz_offset if "Decoy" in prec_seq else 0
    
    
    # ms2_vals = {i:min_int for i in possible_ms2_scans}
    
    ## create dummy 
    ms2_vals = {0:0}
    
    channel_key = (prec_seq,prec_z)
    if channel_key in grouped_decoy_coeffs.groups:
        new_data= grouped_decoy_coeffs.get_group(channel_key)       
        
        prec_rt = new_data.rt.iloc[np.argmax(new_data.coeff)]
        ms2_window_bool = np.logical_and(prec_mz>window_edges[:,0],prec_mz<window_edges[:,1])
        ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
        ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
        possible_ms2_scans = ms2_spec_idxs[ms2_bool]
        ms2_vals = {i:min_int for i in possible_ms2_scans}
        
        for scan,c in zip(new_data["spec_id"],new_data["coeff"]):
            ms2_vals[scan]=c
    # else:
    #     ms2_vals = {}
        
    
    f = interp1d(list(ms2_vals.keys()), list(ms2_vals.values()), bounds_error=False)
        
    
    spectra_subset = [all_spectra.get_by_idx(idx) for idx in all_scans]
    # ms1_vals = {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
    ms1_vals = {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec in spectra_subset}
    
    
    isotopes = iso.precursor_isotopes(prec_seq,prec_z,num_iso)
    
    delta_mz = 0
    if tag.name in prec_seq:
        delta_mz = prec_mz-isotopes[0].mz
    for i  in isotopes:
        i.mz+=delta_mz
        
    
    
    prec_isotope_traces=[]
    # iso_ratios.append([i.intensity for i in isotopes])
    ## note: we have collected similar values for previous channel if the isotopic envelopes are overlapping. 
    ### However, in cases like diethlyation, isoptopes can differ by > 10 ppm #!!!Maybe investigate wider ppm tol for these cases?
    for isotope in isotopes[1:]:# we already have the monoisotopic trace
        # iso_trace = {spec.scan_num:get_trace_int(spec, isotope.mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
        iso_trace = {spec.scan_num:get_trace_int(spec, isotope.mz,rtol=mz_ppm) for spec in spectra_subset}
        prec_isotope_traces.append(iso_trace)
     
    
    all_ms1_vals = {i:min_int for i in all_scans}
    all_ms2_vals = {i:min_int for i in all_scans}
    all_iso_vals = [{i:min_int for i in all_scans} for _ in range(len(prec_isotope_traces))]
    
    for scan in all_scans:
        if scan in ms1_vals:
            all_ms1_vals[scan] = ms1_vals[scan]
            all_ms2_vals[scan] = f(scan)
        # if scan in ms2_vals:
            # all_ms2_vals[scan] = ms2_vals[scan]
        for iso_idx in range(len(prec_isotope_traces)):
            if scan in prec_isotope_traces[iso_idx]:
                all_iso_vals[iso_idx][scan] = prec_isotope_traces[iso_idx][scan]
                
    ms1_traces= [all_ms1_vals,*all_iso_vals]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec_pearsons = [np_pearson_cor(list(all_ms2_vals.values()),list(i.values())).statistic for i in [all_ms1_vals,*all_iso_vals]]
        # all_pearson.append(stats.pearsonr(list(all_ms2_vals.values()),list(all_ms1_vals.values())).statistic)
        
        
        # ms1_spec_idx = filtered_decoy_coeffs.iloc[fdc_idx]["Ms1_spec_id"]
        ms1_spec_idx = all_scans[np.argmax(list(all_ms2_vals.values()))]
        
        theoretical_pattern = [i.intensity for i in isotopes]
        obs_pattern = [all_ms1_vals[ms1_spec_idx],*[iso_trace[ms1_spec_idx] for iso_trace in all_iso_vals]]
        # obs_ratios.append(obs_pattern)
        
        isotope_fit = [np_pearson_cor(theoretical_pattern,obs_pattern),theoretical_pattern,obs_pattern]
    
    # plt.plot(all_ms2_vals.keys(),all_ms2_vals.values())
    # plt.plot(all_ms1_vals.keys(),all_ms1_vals.values())
    # [plt.plot(i.keys(),i.values()) for i in all_iso_vals]
    # plt.xlim(27300,27800)
  
    return isotopes ,ms1_traces, all_ms2_vals, spec_pearsons, isotope_fit
   
# @profile
def fit_group(key,fdc_group,
              ms1_rt,ms2_rt,ms1_spec_idxs,ms2_spec_idxs,ms2_ms1_scan_map,window_floors,window_ceilings,tag,
              decoy_coeffs,grouped_decoy_coeffs,all_spectra,filtered_decoy_coeffs,
              rt_tol, mz_ppm,timeplex):
    
    ms1_spectra = all_spectra.ms1scans
    tag_group = fdc_group.get_group(key)
    prec_mzs = tag_group["mz"]
    prec_seqs = tag_group["seq"]
    prec_z = key[1]
    largest_id = np.argmax(tag_group["coeff"])
    top_ms1_spec_idx = list(tag_group["Ms1_spec_id"])[largest_id]
    prec_rt = list(tag_group["rt"])[largest_id]
    spec_idx_of_largest =list(tag_group["spec_id"])[largest_id]
    rt_bool = np.abs(ms1_rt-prec_rt)<rt_tol
    # print(prec_rt,rt_tol)
    
    window_mz = tag_group["window_mz"].iloc[largest_id]
    # spec = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]
    # mz_ppm=mz_ppm
    # ms1_spectra=ms1_spectra
    # rt_bool=rt_bool
    num_iso =6
    window_half_width = 10
    # assert spec.scan_num == ms1_spec_idx
    
    ### search for all channels always:
    channel_dict = get_other_channels((prec_seqs.iloc[largest_id],prec_z), prec_mzs.iloc[largest_id], tag)
    prec_seqs,prec_mzs = tuple(zip(*channel_dict.values()))

    window_edges = np.stack([window_floors,window_ceilings],1)
    
    # ms2_window_bool = np.logical_and(window_mz>window_floors,window_mz<window_ceilings)
    # ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
    # ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
    # print(sum(ms2_bool))
    # possible_ms2_scans = ms2_spec_idxs[ms2_bool]

    # all_scans = [ms2_ms1_scan_map[i] for i in possible_ms2_scans]
    
    # ## n scans either side of max
    # idx_of_max =all_scans.index(top_ms1_spec_idx)
    # scans_each_side = np.array(all_scans)[np.arange(max(0,idx_of_max-window_half_width),min(len(all_scans),idx_of_max+window_half_width+1))]
    
    ## scans where coeff>1
    # coeff_scans = list(decoy_coeffs["Ms1_spec_id"][np.logical_and(decoy_coeffs["untag_seq"]==key[0],decoy_coeffs["z"]==key[1])])
    
    # ## join
    # all_scans = sorted(list(set(list(scans_each_side)+list(coeff_scans))))
    
    ## scans of max of each
    # largest_coeff_scans = list(filtered_decoy_coeffs["Ms1_spec_id"][np.logical_and(filtered_decoy_coeffs["untag_seq"]==key[0],filtered_decoy_coeffs["z"]==key[1])])
    
    ## max and min of this list
    # max_scan, min_scan = max(largest_coeff_scans), min(largest_coeff_scans)
    # ms1_list_idx_min = list(ms1_spec_idxs).index(min_scan)
    # ms1_list_idx_max = list(ms1_spec_idxs).index(max_scan)
    # scans_each_side = np.array(ms1_spec_idxs)[np.arange(max(0,ms1_list_idx_min-window_half_width),min(len(ms1_spectra),ms1_list_idx_max+window_half_width+1))]
    # all_scans = list(scans_each_side)
    if timeplex:
        ## scans where coeff>1
        coeff_scans = decoy_coeffs["Ms1_spec_id"][np.logical_and.reduce((decoy_coeffs["untag_seq"]==key[0],
                                                                         decoy_coeffs["z"]==key[1],
                                                                         decoy_coeffs["time_channel"]==key[2]))]
        ## join
        # all_scans = sorted(list(set(list(scans_each_side)+list(coeff_scans))))
        ## scans of max of each
        largest_coeff_scans = list(filtered_decoy_coeffs["Ms1_spec_id"][np.logical_and.reduce((filtered_decoy_coeffs["untag_seq"]==key[0],
                                                                                              filtered_decoy_coeffs["z"]==key[1],
                                                                                              filtered_decoy_coeffs["time_channel"]==key[2]))])
        
    else:
        ## scans where coeff>1
        coeff_scans = decoy_coeffs["Ms1_spec_id"][np.logical_and(decoy_coeffs["untag_seq"]==key[0],decoy_coeffs["z"]==key[1])] 
        ## join
        # all_scans = sorted(list(set(list(scans_each_side)+list(coeff_scans))))
        ## scans of max of each
        largest_coeff_scans = list(filtered_decoy_coeffs["Ms1_spec_id"][np.logical_and(filtered_decoy_coeffs["untag_seq"]==key[0],filtered_decoy_coeffs["z"]==key[1])])
    
    ## max and min of this list
    max_scan, min_scan = max(largest_coeff_scans), min(largest_coeff_scans)
    ms1_list_idx_min = list(ms1_spec_idxs).index(min_scan)
    ms1_list_idx_max = list(ms1_spec_idxs).index(max_scan)
    scans_each_side = np.array(ms1_spec_idxs)[np.arange(max(0,ms1_list_idx_min-window_half_width),min(len(ms1_spectra),ms1_list_idx_max+window_half_width+1))]
    all_scans = list(scans_each_side)
    
    # print(len(all_scans))
    ms1_traces = []
    coeff_traces = []
    all_pearson = []
    iso_ratios = []
    group_iso = []
    group_keys = [] ## collect to ensure we match them up correctly
      
    for prec_mz,prec_seq in zip(prec_mzs,prec_seqs):
        isotopes ,channel_ms1_traces, channel_ms2_traces, spec_pearsons, isotope_fit = collect_channel(prec_mz,prec_seq,prec_z,grouped_decoy_coeffs, all_spectra,all_scans,tag, rt_bool,mz_ppm,num_iso,
                                                                                                       window_edges,ms2_rt,ms2_spec_idxs,rt_tol)
        ms1_traces.append(channel_ms1_traces)
        coeff_traces.append(channel_ms2_traces)
        all_pearson.append(spec_pearsons)
        iso_ratios.append(isotope_fit)
        group_iso.append(isotopes)
        group_keys.append((prec_seq,prec_z))
        
    ### need to reduce the number of spectra we fit to
    idx_of_max =all_scans.index(top_ms1_spec_idx)
    scans_to_search = np.array(all_scans)[np.arange(max(0,idx_of_max-window_half_width),min(len(all_scans),idx_of_max+window_half_width+1))]
    scans_to_search = all_scans
    vals = []
    group_pred = []
    group_obs_peaks=[]
    group_matrices =[]
    group_fit_cor =[]
    ### for ms1_spec_idx in all_scans:
    for ms1_spec_idx in scans_to_search:
        spec = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]
        
        pred_coeff, obs_peaks, fit_matrix = fit_mTRAQ_isotopes(spec,group_iso,mz_ppm)
        if len(obs_peaks)==0:
            fit_cor = np.nan
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_cor = np_pearson_cor(np.sum(fit_matrix*pred_coeff,1),obs_peaks)
        vals.append([pred_coeff,obs_peaks,fit_matrix,fit_cor])
        
        group_pred.append(pred_coeff)
        group_obs_peaks.append(obs_peaks)
        group_matrices.append(fit_matrix)
        group_fit_cor.append(fit_cor)
        
    # [plt.plot(i.keys(),i.values()) for i in coeff_traces]
    # spec = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]
    # fitted_ms1 = fit_mTRAQ_isotopes(spec,all_iso,mz_ppm)
    fit_results = [np.array(group_pred),group_obs_peaks,group_matrices,group_fit_cor,scans_to_search]
    
    return fit_results,ms1_traces,coeff_traces,iso_ratios,all_pearson,group_keys

# @profile 
def ms1_cor_channels_fn(all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol,tag=None,multiprocess=False,timeplex=False):
    # print("update")
    num_iso = 3
    
    decoy_coeffs["untag_seq"] = [re.sub(f"(\({tag.name}-\d+\))?","",peptide) for peptide in decoy_coeffs["seq"]]
    
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

    grouped_decoy_coeffs = decoy_coeffs.groupby(["seq","z"])
    fdc_group = filtered_decoy_coeffs.groupby(["untag_seq","z"])

    all_ms1= []
    all_coeff = []
    all_iso = []
    all_group_pearson = []
    all_trace = []
    all_fitted = []
    all_group_keys = []
    
      
    all_keys = list(fdc_group.groups)
    
    if multiprocess:
        with multiprocessing.Pool(8) as p:
            # iso_out = p.starmap(fit_group,tqdm.tqdm(zip(all_seqs,all_frags),total=len(all_seqs)))
            outputs = list(tqdm.tqdm(p.imap(partial(fit_group,
                                                    fdc_group=fdc_group,
                                                    ms1_rt=ms1_rt,
                                                    ms2_rt=ms2_rt,
                                                    ms1_spec_idxs=ms1_spec_idxs,
                                                    ms2_spec_idxs=ms2_spec_idxs,
                                                    ms2_ms1_scan_map=ms2_ms1_scan_map,
                                                    window_floors=bottom_of_window,
                                                    window_ceilings=top_of_window,
                                                    tag=tag,
                                                    decoy_coeffs=decoy_coeffs,
                                                    grouped_decoy_coeffs=grouped_decoy_coeffs,
                                                    all_spectra=all_spectra,
                                                    filtered_decoy_coeffs=filtered_decoy_coeffs,
                                                    rt_tol=rt_tol, 
                                                    mz_ppm=mz_ppm,
                                                    timeplex=timeplex),
                                            all_keys,chunksize=100),total=len(all_keys)))
    else:
        for key in tqdm.tqdm(list(fdc_group.groups)):
            
            fit_results,ms1_traces,coeff_traces,iso_ratios,all_pearson,group_keys = fit_group(key,fdc_group,
                          ms1_rt,ms2_rt,ms1_spec_idxs,ms2_spec_idxs,ms2_ms1_scan_map,bottom_of_window,top_of_window,tag,
                          decoy_coeffs,grouped_decoy_coeffs,all_spectra,filtered_decoy_coeffs,
                          rt_tol, mz_ppm,timeplex)
            
            all_fitted.append(fit_results)
            all_ms1.append(ms1_traces)
            all_coeff.append(coeff_traces)
            all_iso.append(iso_ratios)
            all_group_pearson.append(all_pearson)
            all_group_keys.append(group_keys)
          
    return all_group_pearson, all_ms1, all_coeff, all_iso, all_group_keys, all_fitted

"""

func = np.log10#np.array
### plot the above estimates
for i in range(len(vals[0])):
    plt.plot(scans_to_search,func([j[i] for j in vals]))
    plt.plot(scans_to_search,func([coeff_traces[i][j] for j in scans_to_search]))
    
plt.ylim(-1,10)
plt.title(f"Num Isotopes: {num_iso}")


## oredr by mtraq0
order = np.argsort([-max(i[0]) for i in all_fitted])

func =np.log10#np.array# 
### plot the above estimates
idx = order[0]
key = (grouped_decoy_coeffs.get_group(all_group_keys[idx][0])["untag_seq"].iloc[0],grouped_decoy_coeffs.get_group(all_group_keys[idx][0])["z"].iloc[0])
tag_group = fdc_group.get_group(key)
all_scans = list(all_coeff[idx][0].keys())
group_keys = all_group_keys[idx]
ms1_spec_idx = int(grouped_decoy_coeffs.get_group(all_group_keys[idx][0])["Ms1_spec_id"].iloc[0])
idx_of_max =all_scans.index( ms1_spec_idx)
scans_to_search = np.array(all_scans)[np.arange(max(0,idx_of_max-window_half_width),min(len(all_scans),idx_of_max+window_half_width+1))]
wide_window=10
wider_scans_to_search = np.array(all_scans)[np.arange(max(0,idx_of_max-wide_window),min(len(all_scans),idx_of_max+wide_window+1))]

channels = ["0","4","8"]
channels = ["6","7","8"]

vals = all_fitted[idx]
coeff_traces = all_coeff[idx]
for i in range(len(vals[0])):
    channel = re.findall("\d",group_keys[i][0])[0]
    color = colours[channels.index(channel)]
    # plt.plot(scans_to_search,func([j[i] for j in vals]))
    plt.plot(wider_scans_to_search,func([coeff_traces[i][j] for j in wider_scans_to_search]),linestyle="--",marker="x",color=color)
    plt.plot(wider_scans_to_search,func([all_ms1[idx][i][0][j] for j in wider_scans_to_search]),marker="o",color=color)
    plt.plot(wider_scans_to_search,func([all_ms1[idx][i][1][j] for j in wider_scans_to_search]),color=color)
    # [plt.plot(wider_scans_to_search,func([all_ms1[idx][i][k][j] for j in wider_scans_to_search])) for k in range(len(all_ms1[idx][i]))]
    plt.hlines(func([285435.5, 217832.3125]),min(wider_scans_to_search),max(wider_scans_to_search))
    
# plt.ylim(-1,10)
plt.title(f"Num Isotopes: {num_iso}")


# all_scans = list(all_coeff[idx][0].keys())
for i in range(len(all_ms1[idx])):
    # plt.plot(scans_to_search,func([j[i] for j in vals]))
    plt.plot(all_scans,func([coeff_traces[i][j] for j in all_scans]),linestyle="--")
    plt.plot(all_scans,func([all_ms1[idx][i][0][j] for j in all_scans]))
    
# plt.ylim(-1,10)
plt.title(f"Num Isotopes: {num_iso}")


## Keys
group_keys

## MS2 quant
tag_group["coeff"]

## Ms1 quant






## compare the estimated MS1 quants
[i[2][0] for i in iso_ratios]
    v = np.array(vals)
[max(v[i]) for i in range(len(v))]
plt.scatter(func([i[2][0] for i in iso_ratios]), func([max(v[i]) for i in range(len(v))]))

first_x = np.array([iso_ratios[0][2][0] for iso_ratios in all_iso])
first_y = np.array([np.array(v)[1,0] for v in all_fitted])

all_x = np.concatenate([[i[2][0] for i in iso_ratios] for iso_ratios in all_iso])
# all_y = np.concatenate([[max(np.array(v)[:,i]) for i in range(len(v[0]))] for v in all_fitted])
all_y = np.concatenate([[np.array(v)[1,i] for i in range(len(v[0]))] for v in all_fitted])
all_channel = np.concatenate([[re.findall("\d",all_group_keys[i][j][0])[-1] for j in range(len(all_group_keys[i]))] for i in range(len(fdc_group.groups))])
all_color = np.array([colours[0] if i==channels[0] else colours[1] if i==channels[1] else colours[2] for i in all_channel])
all_alpha = 0.1#np.array([.0 if i=="0" else 0.0 if i=="4" else 0.1 for i in all_channel])
# all_mz = np.concatenate([i["mz"] for _,i in fdc_group])
# all_alpha = 0.1*np.logical_and(np.greater(all_mz,700),np.less(all_mz,8000))

labels = labels = [f"(mTRAQ-{j})" for j in channels]
plt.scatter(np.log10(all_x),np.log10(all_y),c=all_color,s=1,alpha=all_alpha)
plt.plot([0,9],[0,9])
plt.plot([0,9],[1,10],color=colours[0],linestyle=":")
plt.plot([0,9],[.3,9.3],color=colours[0],linestyle="--")
plt.legend(handles=[plt.Line2D([], [], color=colours[i], marker='o', linestyle='None', markersize=10) for i in range(len(labels))],
    labels=labels)
plt.xlabel("Original MS1 Quant")
plt.ylabel("Isotope Fitted MS1 Quant")
plt.xlim(2,9)
plt.ylim(2,9)



labels = labels = [f"(mTRAQ-{j})" for j in [0,4,8]]

c = "0"
x = np.log10(all_x)[all_channel==c]
x1 =  np.log10(all_y)[all_channel==c]
_b = np.isfinite(x)*np.isfinite(x1)
x = x[_b]
x1 = x1[_b]
xx = np.vstack([x,x1])
z = stats.gaussian_kde(xx)(xx)
plt.scatter(x,x1,c=z,s=1)
plt.plot([0,9],[0,9])
plt.legend(handles=[plt.Line2D([], [], color=colours[i], marker='o', linestyle='None', markersize=10) for i in range(len(labels))],
    labels=labels)
plt.xlabel("Original MS1 Quant")
plt.ylabel("Isotope Fitted MS1 Quant")
plt.xlim(2,9)
plt.ylim(2,9)

plt.hist((np.log10(all_x)[all_y!=0]-np.log10(all_y)[all_y!=0])[all_channel[all_y!=0]=="4"],40)

lys_bool = np.array([re.sub("\(mTRAQ-\d\)","",j[0])[-1]=="K" for i in all_group_keys for j in i])
labels = labels = [f"(mTRAQ-{j})" for j in [0,4,8]]
plt.scatter(np.log10(all_x)[lys_bool],np.log10(all_y)[lys_bool],c=all_color[lys_bool],s=1)
plt.plot([0,9],[0,9])
plt.legend(handles=[plt.Line2D([], [], color=colours[i], marker='o', linestyle='None', markersize=10) for i in range(len(labels))],
    labels=labels)
plt.xlabel("Original MS1 Quant")
plt.ylabel("Isotope Fitted MS1 Quant")
plt.xlim(2,9)
plt.ylim(2,9)
"""

"""   


# protect the entry point
if __name__ == '__main__':

    # with open("/Users/kevinmcdonnell/Programming/jmod_test_data/ms1_quant/all_spectra","rb") as read_file:
    #     all_spectra =  dill.load(read_file)
    print("Loading Spectra")
    all_spectra = load_files.SpectrumFile('/Users/kevinmcdonnell/Programming/Data/Diethyl/2024-08-25_SS_DE-6plex_1000ng_3.mzML')
    print("Finished")
    # with open("/Users/kevinmcdonnell/Programming/jmod_test_data/ms1_quant/tag","rb") as read_file:
    #     tag = dill.load(read_file)
    from mass_tags import diethyl_6plex
    tag = diethyl_6plex
        
    filtered_decoy_coeffs = pd.read_csv("/Users/kevinmcdonnell/Programming/jmod_test_data/ms1_quant/filtered_decoy_coeffs.csv")
    decoy_coeffs = pd.read_csv("/Users/kevinmcdonnell/Programming/jmod_test_data/ms1_quant/decoy_coeffs.csv")
    
    mz_ppm = 4.057922902890217e-06
    rt_tol = 7.779594375565848
    import sys
    
    group_p_corrs,group_ms1_traces,group_ms2_traces,group_iso_ratios, group_keys, group_fitted = ms1_cor_channels(all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol,tag=tag)
    # print(sys.getsizeof(group_p_corrs))
    # print(sys.getsizeof(group_ms1_traces))
    # print(sys.getsizeof(group_ms2_traces))
    # print(sys.getsizeof(group_iso_ratios))
    # print(sys.getsizeof(group_keys))
    # print(sys.getsizeof(group_fitted))
    # print(group_fitted[-1])
    group_p_corrs,group_ms1_traces,group_ms2_traces,group_iso_ratios, group_keys, group_fitted = ms1_cor_channels_fn(all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol,tag=tag)
    # print(sys.getsizeof(group_p_corrs))
    # print(sys.getsizeof(group_ms1_traces))
    # print(sys.getsizeof(group_ms2_traces))
    # print(sys.getsizeof(group_iso_ratios))
    # print(sys.getsizeof(group_keys))
    # print(sys.getsizeof(group_fitted))
    # print(group_fitted[-1])
    
    # from load_files import Spectrum
    # prec_mz= 543.42993
    # spec = Spectrum
    # spec.mz = np.load("/Users/kevinmcdonnell/Programming/jmod_test_data/ms1_quant/spec_mz.npy")
    # spec.intens = np.load("/Users/kevinmcdonnell/Programming/jmod_test_data/ms1_quant/spec_intens.npy")
    
    # for i in range(10000):
    #     get_trace_int2(spec,prec_mz,rtol=mz_ppm)





fdc = filtered_decoy_coeffs


mz_ppm = 4.057922902890217e-06
# mz_ppm = 20e-06
### extract the top n keys
big_n = 2000
fdc_n = fdc[-big_n:][::-1]
top_n_keys_d = {tuple([i,j]):k for i,j,k in zip(fdc_n.untag_seq,fdc_n.z,fdc_n.coeff)}
top_n_keys = list(top_n_keys_d)

all_ms1= []
all_coeff = []
all_iso = []
all_group_pearson = []
all_trace = []
all_fitted = []
all_group_keys = []
for key in tqdm.tqdm(list(top_n_keys)):
    
    fit_results,ms1_traces,coeff_traces,iso_ratios,all_pearson,group_keys = fit_group(key,fdc_group,
                  ms1_rt,ms2_rt,ms1_spec_idxs,ms2_spec_idxs,ms2_ms1_scan_map,bottom_of_window,top_of_window,tag,
                  decoy_coeffs,grouped_decoy_coeffs,all_spectra,
                  rt_tol, mz_ppm)
    
    all_fitted.append(fit_results)
    all_ms1.append(ms1_traces)
    all_coeff.append(coeff_traces)
    all_iso.append(iso_ratios)
    all_group_pearson.append(all_pearson)
    all_group_keys.append(group_keys)

orgs =np.array(['Human', 'Ecoli', 'Yeast'])

channels = [0,2,4,6,8,10]
channel_mix = ["A","C","B","D","A","C"]
tag_name = "diethyl_6plex"

theoretical_yeast_amounts = {"A":6,"B":12,"C":18,"D":24}
theoretical_human_amounts = {"A":100,"B":100,"C":100,"D":100}

linker_dict = {key:[group_idx,key_idx] for group_idx,keys in enumerate(all_group_keys) for key_idx,key in enumerate(keys)}
linker_dict[("Q(diethyl_6plex-0)ITVNDLPVGR",2)]

idx = 68589
# idx = 9
org = list(orgs[[top_n_keys[idx][0] in i for i in all_fasta_seqs]])
print(org)
[plt.plot([],label=re.search("\(.*?\)",i[0])[0]) for i in all_group_keys[idx]]
plt.legend();plt.gca().set_prop_cycle(None)
plt.title([org,top_n_keys[idx]])
[plt.plot(all_fitted[idx][-1],i,color=colours[n],linestyle="--") for n,i in enumerate(all_fitted[idx][0].T)] ## ms1 fit

[plt.plot(all_fitted[idx][-1],[i[0][j] for j in all_fitted[idx][-1]],color=colours[n]) for n,i in enumerate(all_ms1[idx])] #Ms1 mono iso of each channel
[plt.plot(all_fitted[idx][-1],[i[j] for j in all_fitted[idx][-1]],color=colours[n]) for n,i in enumerate(all_coeff[idx])] # coeff predicted


predicted_coeffs = np.max([list(i.values()) for i in all_coeff[idx]],axis=1)
norm_predicted_coeffs = predicted_coeffs/predicted_coeffs[0]
predicted_ms1 = np.max(np.array([i for i in all_fitted[idx][0]]),axis=0)
norm_predicted_ms1 = predicted_ms1/predicted_ms1[0]
obs_mono_ms1 = np.max([list(i[0].values()) for i in all_ms1[idx]],axis=1)
norm_obs_mono =obs_mono_ms1/obs_mono_ms1[0]
norm_theoretical = np.array([theoretical_yeast_amounts[i] for i in channel_mix])/6
print([org,keys[idx]])
print("Fit MS2"," \t".join(['%.2f'% elem for elem in np.round(norm_predicted_coeffs,2)]))
print("Fit MS1"," \t".join(['%.2f'% elem for elem in np.round(norm_predicted_ms1,2)]))
print("Obs MS1"," \t".join(['%.2f'% elem for elem in np.round(norm_obs_mono,2)]))
print("Theoret"," \t".join(['%.2f'% elem for elem in np.round(norm_theoretical,2)]))
print()




key = (re.sub("\(.*?\)","",all_group_keys[idx][0][0]),all_group_keys[idx][0][1])

fit_results,ms1_traces,coeff_traces,iso_ratios,all_pearson,group_keys = fit_group(key,fdc_group,
              ms1_rt,ms2_rt,ms1_spec_idxs,ms2_spec_idxs,ms2_ms1_scan_map,bottom_of_window,top_of_window,tag,
              decoy_coeffs,grouped_decoy_coeffs,all_spectra,
              rt_tol, mz_ppm)


def scan_rt(scan_num,all_scans,return_rt=True):
    if return_rt:
        return all_scans.get_by_idx(scan_num).RT
    else:
        return scan_num
    
def scan_rts(scan_nums,all_scans,return_rt=True):
    if type(scan_nums)==int:
        return scan_rt(scan_nums, all_scans,return_rt)
    elif type(scan_nums)==list:
        rts = [scan_rt(i, all_scans,return_rt) for i in scan_nums]
        return rts
    else:
        raise ValueError("Incorrect input type")
        
        
max_intensity = 1.5e8
plot_rt = True
[plt.plot(scan_rts(list(all_fitted[idx][-1]),DIAspectra,plot_rt),[i[0][j] for j in all_fitted[idx][-1]],color=colours[n]) for n,i in enumerate(all_ms1[idx])];plt.ylim(0,max_intensity) #Ms1 mono iso of each channelp
[plt.plot(scan_rts(list(all_fitted[idx][-1]),DIAspectra,plot_rt),[i[1][j] for j in all_fitted[idx][-1]],color=colours[n]) for n,i in enumerate(all_ms1[idx])];plt.ylim(0,max_intensity) #Ms1 mono iso of each channel
[plt.plot(scan_rts(list(all_fitted[idx][-1]),DIAspectra,plot_rt),[i[2][j] for j in all_fitted[idx][-1]],color=colours[n]) for n,i in enumerate(all_ms1[idx])];plt.ylim(0,max_intensity) #Ms1 mono iso of each channel
[plt.plot(scan_rts(list(all_fitted[idx][-1]),DIAspectra,plot_rt),[i[3][j] for j in all_fitted[idx][-1]],color=colours[n]) for n,i in enumerate(all_ms1[idx])];plt.ylim(0,max_intensity) #Ms1 mono iso of each channel
[plt.plot(scan_rts(list(all_fitted[idx][-1]),DIAspectra,plot_rt),[i[4][j] for j in all_fitted[idx][-1]],color=colours[n]) for n,i in enumerate(all_ms1[idx])];plt.ylim(0,max_intensity) #Ms1 mono iso of each channel
[plt.plot(scan_rts(list(all_fitted[idx][-1]),DIAspectra,plot_rt),[i[5][j] for j in all_fitted[idx][-1]],color=colours[n]) for n,i in enumerate(all_ms1[idx])];plt.ylim(0,max_intensity) #Ms1 mono iso of each channel


org = list(orgs[[keys[idx][0] in i for i in all_fasta_seqs]])
org = ["none"] if org==[] else org
predicted_coeffs = np.max([list(i.values()) for i in all_coeff[idx]],axis=1)
norm_predicted_coeffs = predicted_coeffs/predicted_coeffs[0]
predicted_ms1 = np.max(np.array([i for i in all_fitted[idx][0]]),axis=0)
norm_predicted_ms1 = predicted_ms1/predicted_ms1[0]
obs_mono_ms1 = np.max([list(i[0].values()) for i in all_ms1[idx]],axis=1)
norm_obs_mono =obs_mono_ms1/obs_mono_ms1[0]
norm_theoretical = np.array([theoretical_yeast_amounts[i] for i in channel_mix])/6
print([org,keys[idx]])
print("Fit MS2"," \t".join(['%.2f'% elem for elem in np.round(_func(norm_predicted_coeffs),2)]))
print("Fit MS1"," \t".join(['%.2f'% elem for elem in np.round(_func(norm_predicted_ms1),2)]))
print("Obs MS1"," \t".join(['%.2f'% elem for elem in np.round(_func(norm_obs_mono),2)]))
print("Theoret"," \t".join(['%.2f'% elem for elem in np.round(_func(norm_theoretical),2)]))
print()


print("Fit MS2"," \t".join(['%.2f'% elem for elem in np.round(_func(norm_predicted_coeffs)-_func(norm_theoretical),2)]))
print("Fit MS1"," \t".join(['%.2f'% elem for elem in np.round(_func(norm_predicted_ms1)-_func(norm_theoretical),2)]))
print("Obs MS1"," \t".join(['%.2f'% elem for elem in np.round(_func(norm_obs_mono)-_func(norm_theoretical),2)]))
print("Theoret"," \t".join(['%.2f'% elem for elem in np.round(_func(norm_theoretical)-_func(norm_theoretical),2)]))
print()


# [plt.plot(i[0].keys(),i[0].values()) for i in all_ms1[idx]] #mono iso of each channel
# [plt.plot(i.keys(),i.values()) for i in all_coeff[idx]]
all_group_pearson, all_ms1, all_coeff, all_iso, all_group_keys, all_fitted=group_p_corrs,group_ms1_traces,group_ms2_traces,group_iso_ratios, group_keys, group_fitted

linker_dict = {key:[group_idx,key_idx] for group_idx,keys in enumerate(all_group_keys) for key_idx,key in enumerate(keys)}

all_keys = [(re.sub("\(.*?\)","",i[0][0]),i[0][1]) for i in group_keys]

keys = all_keys
keys = top_n_keys
d = {}
# for idx in range(74648):
for idx in range(len(keys)):  
    # org = list(orgs[[all_keys[idx][0] in i for i in all_fasta_seqs]]) 
    org = list(orgs[[keys[idx][0] in i for i in all_fasta_seqs]])
    org = ["none"] if org==[] else org
    predicted_coeffs = np.max([list(i.values()) for i in all_coeff[idx]],axis=1)
    norm_predicted_coeffs = predicted_coeffs/predicted_coeffs[0]
    predicted_ms1 = np.max(np.array([i for i in all_fitted[idx][0]]),axis=0)
    norm_predicted_ms1 = predicted_ms1/predicted_ms1[0]
    obs_mono_ms1 = np.max([list(i[0].values()) for i in all_ms1[idx]],axis=1)
    norm_obs_mono =obs_mono_ms1/obs_mono_ms1[0]
    norm_theoretical = np.array([theoretical_yeast_amounts[i] for i in channel_mix])/6
    # print([org,keys[idx]])
    # print("Fit MS2"," \t".join(['%.2f'% elem for elem in np.round(norm_predicted_coeffs,2)]))
    # print("Fit MS1"," \t".join(['%.2f'% elem for elem in np.round(norm_predicted_ms1,2)]))
    # print("Obs MS1"," \t".join(['%.2f'% elem for elem in np.round(norm_obs_mono,2)]))
    # print("Theoret"," \t".join(['%.2f'% elem for elem in np.round(norm_theoretical,2)]))
    # print()
    d.setdefault(org[0],{"norm_predicted_coeffs":[],"norm_predicted_ms1":[],"norm_obs_mono":[],"norm_theoretical":[]})
    d[org[0]]["norm_predicted_coeffs"].append(norm_predicted_coeffs)
    d[org[0]]["norm_predicted_ms1"].append(norm_predicted_ms1)
    d[org[0]]["norm_obs_mono"].append(norm_obs_mono)
    d[org[0]]["norm_theoretical"].append(norm_theoretical)



_func = np.log2

[plt.hist(i,np.linspace(0,10,20),alpha=.2) for i in np.stack(d["Yeast"]["norm_predicted_coeffs"],1)]
print([np.nanmedian(i) for i in np.stack(d["Yeast"]["norm_predicted_coeffs"],1)])
print([np.nanmedian(i) for i in np.stack(d["Yeast"]["norm_predicted_ms1"],1)])
print([np.nanmedian(i) for i in np.stack(d["Yeast"]["norm_obs_mono"],1)])
print([np.nanmedian(i) for i in np.stack(d["Yeast"]["norm_theoretical"],1)])

org = "Human"
print("Fit MS2"," \t".join(['%.2f'% elem for elem in np.round([np.nanmedian(i) for i in _func(np.stack(d[org]["norm_predicted_coeffs"],1))],2)]))
print("Fit MS1"," \t".join(['%.2f'% elem for elem in np.round([np.nanmedian(i) for i in _func(np.stack(d[org]["norm_predicted_ms1"],1))],2)]))
print("Obs MS1"," \t".join(['%.2f'% elem for elem in np.round([np.nanmedian(i) for i in _func(np.stack(d[org]["norm_obs_mono"],1))],2)]))
print("Theoret"," \t".join(['%.2f'% elem for elem in np.round([np.nanmedian(i) for i in _func(np.stack(d[org]["norm_theoretical"],1))],2)]))
print()








# """