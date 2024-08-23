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

from miscFunctions import  closest_ms1spec, curve_fit, gaussian
from pyteomics import mass
from read_output import get_large_prec

import config 

import dill
import load_files
import tqdm

import iso_functions as iso
import warnings

from scipy import sparse
import sparse_nnls
import Jplot as jp


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

def get_trace_int(spec,mz,atol=0,rtol=0,base=min_int):
    closest_idx = np.argmin(np.abs(spec.mz-mz))
    # order_idx = np.searchsorted(spec.mz,mz)
    # closest_idx = order_idx-1 if order_idx >= len(spec.mz) or abs(spec.mz[order_idx]-mz)>abs(spec.mz[order_idx-1]-mz) else order_idx-1
    # if np.isclose(spec.mz[closest_idx],mz,atol=atol,rtol=rtol):

    if (abs(spec.mz[closest_idx] - mz)/mz)<rtol:
        return spec.intens[closest_idx]
    else:
        return base



# all_spectra = 1#DIAspectra

# filtered_decoy_coeffs = 3#fdc
# rt_tol = 0
# mz_ppm = 10

def ms1_cor(all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol,timeplex=False):
    num_iso = 6
    
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
    
        ms2_window_bool = np.logical_and(prec_mz>bottom_of_window,prec_mz<top_of_window)
        ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
        ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
        possible_ms2_scans = ms2_spec_idxs[ms2_bool]
        
        if timeplex:
            time_channel = filtered_decoy_coeffs["time_channel"][fdc_idx]
            new_data= grouped_decoy_coeffs.get_group((prec_seq,prec_z,time_channel))
        else:
            new_data= grouped_decoy_coeffs.get_group((prec_seq,prec_z))
        
        rt_bool = np.abs(ms1_rt-prec_rt)<rt_tol
        
        ms1_scans = ms1_spec_idxs[rt_bool]
        ms1_scans_from_coeff = new_data["Ms1_spec_id"]
        all_scans = sorted(set(ms1_scans).union(set(ms1_scans_from_coeff)))
        all_scans = [ms2_ms1_scan_map[i] for i in possible_ms2_scans]
        all_scans = sorted(set(ms1_scans_from_coeff))
        ms1_vals = {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
        ms2_vals = {i:j for i,j in zip(new_data["Ms1_spec_id"],new_data["coeff"])}
        
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
            if scan in ms2_vals:
                all_ms2_vals[scan] = ms2_vals[scan]
            for iso_idx in range(len(prec_isotope_traces)):
                if scan in prec_isotope_traces[iso_idx]:
                    all_iso_vals[iso_idx][scan] = prec_isotope_traces[iso_idx][scan]
                    
                    
        ms1_traces.append([all_ms1_vals,*all_iso_vals])
        coeff_traces.append(all_ms2_vals)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec_pearsons = [0]*num_iso
            if len(all_scans)>2:
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
        
        # not_dia_col_indices = np.arange(len(ref_coords))
        # not_dia_row_indices = np.array([last_row]*len(not_dia_col_indices),dtype=int)
        # not_dia_values = np.array([np.sum([ms1_iso_patterns[:,:,1][idx][peak_idx] for peak_idx in range(len(ms1_iso_patterns[:,:,1][idx])) if ref_coords[idx][peak_idx]%2==0])
        #                           for idx in range(len(ref_coords))])
        
        
        #### Type B
        not_dia_col_indices = np.arange(len(ref_coords))
        not_dia_row_indices = [last_row+1]*len(not_dia_col_indices)+not_dia_col_indices
        not_dia_values = np.array([np.sum([ms1_iso_patterns[:,:,1][idx][peak_idx] for peak_idx in range(len(ms1_iso_patterns[:,:,1][idx])) if ref_coords[idx][peak_idx]%2==0])
                                  for idx in range(len(ref_coords))])
        
        #### Type C
        # all_unmatched_peaks = [[ms1_iso_patterns[:,:,1][idx][peak_idx] for peak_idx in range(len(ms1_iso_patterns[:,:,1][idx])) if ref_coords[idx][peak_idx]%2==0 and ms1_iso_patterns[:,:,1][idx][peak_idx]>lower_limit]
        #                           for idx in range(len(ms1_iso_patterns[:,:,1]))]
        # num_unmatched_to_fit = [len(i) for i in all_unmatched_peaks]
        # not_dia_col_indices = np.array(np.concatenate([[idx]*i for idx,i in enumerate(num_unmatched_to_fit)]),dtype=int)
        # not_dia_row_indices = np.array(np.arange(np.sum(num_unmatched_to_fit))+last_row+1,dtype=int)
        # not_dia_values = np.concatenate(all_unmatched_peaks)
        
        
        
        sparse_row_indices = np.append(ref_spec_row_indices,not_dia_row_indices)
        sparse_col_indices = np.append(ref_spec_col_indices,not_dia_col_indices)
        sparse_values = np.append(ref_spec_values,not_dia_values)
        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows
        sparse_row_indices = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1
        
        # Generate sparse matrix from data
        sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))
        dia_spec_int = np.append(dia_spec_int,[0]*(sparse_lib_matrix.shape[0]-dia_spec_int.shape[0])) 
        
        # Fit lib spectra to observed spectra
        # fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix,dia_spec_int,{"show_progress":False})
        # lib_coefficients = np.array(fit_results['x']).flatten()
        
        matrix = np.array(sparse_lib_matrix.todense())
        lib_coefficients = np.linalg.lstsq(matrix, dia_spec_int)[0]
        
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


def ms1_cor_channels(all_spectra,filtered_decoy_coeffs,decoy_coeffs,mz_ppm,rt_tol,tag=None):
    num_iso = 3
    
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
    
    for key in tqdm.tqdm(list(fdc_group.groups)):
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
        window_half_width = 2
        # assert spec.scan_num == ms1_spec_idx
        
        ### search for all channels always:
        channel_dict = get_other_channels((prec_seqs.iloc[largest_id],prec_z), prec_mzs.iloc[largest_id], tag)
        prec_seqs,prec_mzs = tuple(zip(*channel_dict.values()))
    
    
        
        ms2_window_bool = np.logical_and(window_mz>bottom_of_window,window_mz<top_of_window)
        ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
        ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
        # print(sum(ms2_bool))
        possible_ms2_scans = ms2_spec_idxs[ms2_bool]
 
        all_scans = [ms2_ms1_scan_map[i] for i in possible_ms2_scans]
        # print(len(all_scans))
        ms1_traces = []
        coeff_traces = []
        is_traces = []
        all_pearson = []
        iso_ratios = []
        obs_ratios = []
        group_iso = []
        group_keys = [] ## collect to ensure we match them up correctly
        for prec_mz,prec_seq in zip(prec_mzs,prec_seqs):
            
            ## keep decoys mathching to the correct MS1
            offset = config.decoy_mz_offset if "Decoy" in prec_seq else 0
            
            group_keys.append((prec_seq,prec_z))
            
            # ms2_window_bool = np.logical_and(prec_mz>bottom_of_window,prec_mz<top_of_window)
            # # ms2_window_bool = np.logical_and(window_mz>bottom_of_window,window_mz<top_of_window)
            # ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
            # ms2_bool = np.logical_and(ms2_window_bool,ms2_rt_bool)
            # possible_ms2_scans = ms2_spec_idxs[ms2_bool]
            
            
            
            # all_scans = [ms2_ms1_scan_map[i] for i in possible_ms2_scans]
            # # ms1_scans_from_coeff = new_data["Ms1_spec_id"]
            # # all_scans = sorted(set(ms1_scans_from_coeff))
            
            channel_key = (prec_seq,prec_z)
            if channel_key in grouped_decoy_coeffs.groups:
                new_data= grouped_decoy_coeffs.get_group(channel_key)           
                ms2_vals = {i:j for i,j in zip(new_data["Ms1_spec_id"],new_data["coeff"])}
                # plt.plot(ms2_vals.keys(),np.log10(list(ms2_vals.values())))
            else:
                ms2_vals = {}
                
            traces= []
            ms1_vals = {spec.scan_num:get_trace_int(spec, prec_mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
            
            
            isotopes = iso.precursor_isotopes(prec_seq,prec_z,num_iso)
            
            delta_mz = 0
            if "mTRAQ" in prec_seq:
                delta_mz = prec_mz-isotopes[0].mz
            for i  in isotopes:
                i.mz+=delta_mz
                
            group_iso.append(isotopes)
            
            prec_isotope_traces=[]
            # iso_ratios.append([i.intensity for i in isotopes])
            for isotope in isotopes[1:]:# we already have the monoisotopic trace
                iso_trace = {spec.scan_num:get_trace_int(spec, isotope.mz,rtol=mz_ppm) for spec,use in zip(ms1_spectra,rt_bool) if use}
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
                        
            
            # def func(x): return(np.array(list(x)))
            # plt.plot(all_ms2_vals.keys(),func(all_ms2_vals.values()),label="Coeffs")
            # plt.plot(all_ms1_vals.keys(),func(all_ms1_vals.values()),label="Monoiso")
            # plt.plot(all_iso_vals[0].keys(),func(all_iso_vals[0].values()),label="1st Iso")
            # plt.plot(all_iso_vals[1].keys(),func(all_iso_vals[1].values()),label="2nd Iso")
                        
            ms1_traces.append([all_ms1_vals,*all_iso_vals])
            coeff_traces.append(all_ms2_vals)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec_pearsons = [stats.pearsonr(list(all_ms2_vals.values()),list(i.values())).statistic for i in [all_ms1_vals,*all_iso_vals]]
                # all_pearson.append(stats.pearsonr(list(all_ms2_vals.values()),list(all_ms1_vals.values())).statistic)
                all_pearson.append(spec_pearsons)
                
                # ms1_spec_idx = filtered_decoy_coeffs.iloc[fdc_idx]["Ms1_spec_id"]
                ms1_spec_idx = all_scans[np.argmax(list(all_ms2_vals.values()))]
                
                theoretical_pattern = [i.intensity for i in isotopes]
                obs_pattern = [all_ms1_vals[ms1_spec_idx],*[iso_trace[ms1_spec_idx] for iso_trace in all_iso_vals]]
                # obs_ratios.append(obs_pattern)
                iso_ratios.append([stats.pearsonr(theoretical_pattern,obs_pattern),theoretical_pattern,obs_pattern])
            
        ### need to reduce the number of spectra we fit to
        idx_of_max =all_scans.index(top_ms1_spec_idx)
        scans_to_search = np.array(all_scans)[np.arange(max(0,idx_of_max-window_half_width),min(len(all_scans),idx_of_max+window_half_width+1))]
        vals = []
        group_pred = []
        group_obs_peaks=[]
        group_matrices =[]
        group_fit_cor =[]
        ### for ms1_spec_idx in all_scans:
        for ms1_spec_idx in scans_to_search:
            spec = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]
            
            pred_coeff, obs_peaks, fit_matrix = fit_mTRAQ_isotopes(spec,group_iso,mz_ppm)
            if obs_peaks==[]:
                fit_cor = np.nan
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_cor = stats.pearsonr(np.sum(fit_matrix*pred_coeff,1),obs_peaks)
            vals.append([pred_coeff,obs_peaks,fit_matrix,fit_cor])
            
            group_pred.append(pred_coeff)
            group_obs_peaks.append(obs_peaks)
            group_matrices.append(fit_matrix)
            group_fit_cor.append(fit_cor)
            
        # spec = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]
        # fitted_ms1 = fit_mTRAQ_isotopes(spec,all_iso,mz_ppm)
        
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



# """