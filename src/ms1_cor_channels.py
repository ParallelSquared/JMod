#  Copyright (c) 2026 Parallel Squared Technology Institute
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  """

import numpy as np
import polars as pl
import re
import sys
import os
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config 

import tqdm
from src.utils.misc_functions import  closest_ms1spec,np_pearson_cor
from scipy.interpolate import interp1d
from src.utils import misc_functions as mf
from src import iso_functions as iso

import warnings
from scipy.signal import find_peaks
import logging

from scipy import optimize
import numba as nb



min_int = 1e-3
import csv
import pickle
from src.logger import logger

## To enable fitting whole MS1 spectrum: in FDR analysis, set fit_whole_MS1 to true
    # Uncomment the two blocks of code that follow if fit_whole_MS1:
    # Uncomment functions fit_whole_MS1_sepctrum, add_to_MS1_spec_dict, and fit_all_ms1_specs



def ms1_cor_channels(all_spectra,
                     filtered_decoy_coeffs,
                     decoy_coeffs,
                     mz_ppm,
                     rt_tol,
                     tag=None,
                     SILAC=None,
                     timeplex=False,
                     num_iso = None,
                     num_iso_r = None,
                     additional_scans = None,
                     vote_sigma = 1.0,
                     fit_whole_MS1 = False,
                     dump_precursors = None
                     ):

    window_half_width = 10

    if tag.name != "no_tag":
        logger.info("Fitting tagged channels together")
    else:
        logger.info("Fitting Precursors Individually")

    ms1_spectra = all_spectra.ms1scans
    ms2_spectra = all_spectra.ms2scans
    
    ## array of ms1 and ms2 retention time
    ms2_rt = np.array([i.RT for i in ms2_spectra])
    ms1_rt = np.array([i.RT for i in ms1_spectra])
    
    ## array of scan numbers for ms1 and ms2 spectra
    ms1_spec_idxs = np.array([i.scan_num for i in ms1_spectra])
    ms2_spec_idxs = np.array([i.scan_num for i in ms2_spectra])
    
    ## get ms2 info for filtering
    bottom_of_window, top_of_window = np.array([i.ms1window for i in ms2_spectra]).T
    ms2_rt = np.array([i.RT for i in ms2_spectra])

    ## list of scan nums of the closest ms1 scan for each ms2 scan (IM-aware if available)
    if all_spectra.ms2_to_ms1_map is not None:
        resp_ms1scans = [ms1_spec_idxs[all_spectra.ms2_to_ms1_map[i]] for i in range(len(ms2_spectra))]
    else:
        resp_ms1scans = [ms1_spec_idxs[closest_ms1spec(ms2_rt[i], ms1_rt)] for i in range(len(ms2_rt))]

    ## mapping of ms2 scan nums to ms1 scan nums
    ms2_ms1_scan_map = {spec.scan_num:resp_ms1scans[i] for i,spec in enumerate(ms2_spectra)}

    
    if timeplex:
        dc_group_keys = ["seq", "z", "time_channel"]
        fdc_group = filtered_decoy_coeffs.groupby(["untag_seq","z","time_channel"])
    else:
        dc_group_keys = ["seq", "z"]
        fdc_group = filtered_decoy_coeffs.groupby(["untag_seq","z"])

    if isinstance(decoy_coeffs, pl.DataFrame):
        # Production path: ``decoy_coeffs`` is a slim polars frame pre-sorted
        # by ``dc_group_keys`` (see read_output.get_large_prec). Build a {key:
        # (start, length)} offsets dict so per-precursor lookups in
        # get_ms2_vals slice a contiguous range and convert just that slice to
        # pandas. Full pandas materialization of the post-NNLS frame OOMs at
        # ~60GB on multiplexed low-input runs.
        _bounds = (decoy_coeffs.with_row_index("__idx")
                               .group_by(dc_group_keys, maintain_order=True)
                               .agg(pl.col("__idx").min().alias("start"),
                                    pl.col("__idx").len().alias("n")))
        dc_offsets = {
            tuple(row[:-2]): (int(row[-2]), int(row[-1]))
            for row in _bounds.iter_rows()
        }
        dc_source = (decoy_coeffs, dc_offsets)
    else:
        # Legacy pandas path — kept for unit-test fixtures that pass small
        # pandas frames directly. These mutations are dead within this
        # function but exposed in test assertions.
        decoy_coeffs["untag_seq"] = [re.sub(rf"(\({tag.name}-\d+\))?","",peptide) for peptide in decoy_coeffs["seq"]]
        decoy_coeffs["untag_prec"] = ["_".join([i[0],str(int(i[1]))]) for i in zip(decoy_coeffs["untag_seq"],decoy_coeffs["z"])]

        if "med_frag_error" not in decoy_coeffs.columns:
            frag_errors = [np.array(x, dtype=float) if x is not None and len(x) > 0 else np.array([]) for x in decoy_coeffs.frag_errors]
            non_empty = [i for i in frag_errors if len(i) > 0]
            median = np.median(np.concatenate(non_empty)) if non_empty else 0.0
            decoy_coeffs["med_frag_error"] = [np.median(np.abs(median-i)) if len(i) > 0 else np.nan for i in frag_errors]

        if "abs_rt_error" not in decoy_coeffs.columns:
            decoy_coeffs["abs_rt_error"] = np.abs(decoy_coeffs.rt_error)

        if "abs_mz_error" not in decoy_coeffs.columns:
            decoy_coeffs["abs_mz_error"] = np.abs(decoy_coeffs.mz_error)

        dc_source = decoy_coeffs.groupby(dc_group_keys)

    all_ms1, all_coeff, all_iso, all_group_pearson, all_trace, all_fitted, all_group_keys, all_scans_len = ([] for _ in range(8))
    dump_data = {"ms1_ppm": mz_ppm, "num_iso": num_iso, "precursors": {}} if dump_precursors else None

    #these are for fit_whole_ms1
    ms1_spec_dict = {k: {"fdc_idx": [], "peak_mz": [], "rel_iso_int": [], "monoiso_groups": [], "MS1_spectra": None} for k in ms1_spec_idxs}
    dummy_idx_list = list(filtered_decoy_coeffs.index)
    fake_fdc_dict = {}


    GUI_print_idxs = [int(((len(fdc_group.groups)-1)/10)*y) for y in range(1,11)]
    fdc_group_idx = -1
    for key in tqdm.tqdm(list(fdc_group.groups)):
        fdc_group_idx += 1
        if config.ran_from_GUI:
            if fdc_group_idx in GUI_print_idxs:
                frac_done = (GUI_print_idxs.index(fdc_group_idx)+1) * 10
                logger.info(f"Fitting - {frac_done}%")

        tag_group = fdc_group.get_group(key)
        group_protein = tag_group["protein"].iloc[0] if "protein" in tag_group.columns else ""

        prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel = get_seqs_and_mzs(fdc_group, timeplex, tag, key, SILAC)
        all_scans, spectra_subset = minmax_spec_window(largest_coeff_scans, ms1_spec_idxs, ms1_spectra, all_spectra, window_half_width)

        ### Gaussian-weighted voting: each channel contributes its max-coeff scan
        # as a Gaussian centered on that scan, with sigma = vote_sigma cycles
        # (derived from the global elution SD). Adjacent-cycle votes reinforce
        # instead of competing, and outlier channels only contribute weakly far
        # from the consensus region.
        all_scans_arr = np.asarray(all_scans)
        positions = np.arange(len(all_scans_arr), dtype=np.float64)
        votes = np.zeros(len(all_scans_arr), dtype=np.float64)
        for seq in tag_group["seq"].unique():
            ch_rows = tag_group[tag_group["seq"] == seq]
            best_row = ch_rows.loc[ch_rows["coeff"].idxmax()]
            s_idx = float(np.searchsorted(all_scans_arr, int(best_row["Ms1_spec_id"])))
            votes += float(best_row["coeff"]) * np.exp(-0.5 * ((positions - s_idx) / vote_sigma) ** 2)
        voted_apex_idx = int(np.argmax(votes))
        voted_apex = int(all_scans_arr[voted_apex_idx])

        # Fit a wider ±fit_radius window around voted_apex so process_ms1_quant
        # can monotonically slide each channel's apex away from the group's voted
        # apex toward a local maximum. The slide is capped at the second-from-edge
        # position, so a channel can drift at most config.args.apex_jitter cycles
        # from the voted apex. apex_jitter == 0 yields a 3-scan window — no walk.
        fit_radius = max(int(config.args.apex_jitter) + 1, additional_scans + 1)
        apex_idx = int(np.searchsorted(all_scans, voted_apex))
        lo = max(0, apex_idx - fit_radius)
        hi = min(len(all_scans), apex_idx + fit_radius + 1)
        apex_scans = all_scans[lo:hi]

        ms1_traces, coeff_traces, is_traces, all_pearson, iso_ratios = ([] for _ in range(5))
        obs_ratios, group_iso, group_keys, all_channel_scans, interp_funcs, best_coeff = ([] for _ in range(6))

        for prec_mz,prec_seq in zip(prec_mzs,prec_seqs):

            ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, dc_source, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
            group_keys.append(channel_key)

            interp_func = build_ms2_interpolator(ms2_vals)
            interp_funcs.append(interp_func)

            all_ms1_vals, all_ms2_vals, all_iso_vals, isotopes, interp_func = get_isotopes_and_vals(prec_seq, prec_z, num_iso, [tag,SILAC], all_scans, prec_mz, mz_ppm, spectra_subset, interp_func)
            group_iso.append(isotopes)

            ## filter to voted apex neighborhood
            channel_scans = apex_scans
            all_ms1_vals = {s: all_ms1_vals[s] for s in channel_scans}
            all_iso_vals = [{s: iso[s] for s in channel_scans} for iso in all_iso_vals]
            all_ms2_vals = {s: all_ms2_vals[s] for s in channel_scans}
            all_channel_scans.append(channel_scans)
            ms1_traces.append([all_ms1_vals,*all_iso_vals])
            coeff_traces.append(all_ms2_vals)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                all_pearson_to_append, iso_ratios_to_append = compute_ms1_ms2_cors(all_ms2_vals, all_ms1_vals, all_iso_vals, num_iso_r, channel_scans, isotopes)
                all_pearson.append(all_pearson_to_append)
                iso_ratios.append(iso_ratios_to_append)

        scans_to_search = apex_scans

        group_pred, group_obs_peaks, group_matrices, group_fit_cor, group_kept_mz = ([] for _ in range(5))

        for ms1_spec_idx in scans_to_search:
            pred_coeff, obs_peaks, fit_matrix, fit_cor, kept_mz = fit_isotopes_and_score(ms1_spectra, ms1_spec_idxs, ms1_spec_idx, group_iso, mz_ppm)
            group_pred.append(pred_coeff)
            group_obs_peaks.append(obs_peaks)
            group_matrices.append(fit_matrix)
            group_fit_cor.append(fit_cor)
            group_kept_mz.append(kept_mz)

        # Collect dump data for matching precursors
        if dump_precursors is not None:
            untag_prec_str = f"{key[0]}_{int(key[1])}"
            if untag_prec_str in dump_precursors:
                channels_info = []
                for ch_i, (seq, mz_val) in enumerate(zip(prec_seqs, prec_mzs)):
                    iso_list = [(p.mz, p.intensity) for p in group_iso[ch_i]] if ch_i < len(group_iso) else []
                    channels_info.append({"seq": seq, "mz": float(mz_val), "isotopes": iso_list})

                scans_info = []
                for scan_i, ms1_spec_idx in enumerate(scans_to_search):
                    pred_c = group_pred[scan_i]
                    obs_p = group_obs_peaks[scan_i]
                    mat = group_matrices[scan_i]
                    fc = group_fit_cor[scan_i]
                    km = group_kept_mz[scan_i]

                    # Build row labels: (channel_idx, isotope_idx, theoretical_mz)
                    row_labels = []
                    if len(obs_p) > 0:
                        for row_i in range(len(obs_p)):
                            ch_idx = int(np.argmax(mat[row_i, :]))
                            iso_idx = -1
                            theo_mz = 0.0
                            if ch_idx < len(group_iso):
                                diffs = [abs(km[row_i] - p.mz) for p in group_iso[ch_idx]]
                                iso_idx = int(np.argmin(diffs))
                                theo_mz = group_iso[ch_idx][iso_idx].mz
                            row_labels.append((ch_idx, iso_idx, theo_mz))

                    scans_info.append({
                        "scan_num": int(ms1_spec_idx),
                        "matrix": mat,
                        "observed_intensities": obs_p,
                        "observed_mz": km,
                        "row_labels": row_labels,
                        "current_coefficients": pred_c,
                        "fit_cor": fc,
                    })

                csv_meta = dump_precursors[untag_prec_str] if isinstance(dump_precursors, dict) else {}
                dump_data["precursors"][untag_prec_str] = {
                    "charge": int(key[1]),
                    "protein": group_protein,
                    "group": csv_meta.get("group", ""),
                    "csv_metadata": csv_meta,
                    "channels": channels_info,
                    "scans": scans_info,
                }

        # all_fitted.append(vals)
        all_fitted.append([np.array(group_pred),group_obs_peaks,group_matrices,group_fit_cor,scans_to_search])
        all_ms1.append(ms1_traces)
        all_coeff.append(coeff_traces)
        all_iso.append(iso_ratios)
        all_group_pearson.append(all_pearson)
        all_group_keys.append(group_keys)

    new_output_dict = {}
    # if fit_whole_MS1:
    #     new_output_dict = fit_all_ms1_specs(ms1_spec_dict, new_output_dict, mz_ppm)

    if dump_data is not None and dump_data["precursors"]:
        dump_path = os.path.join(os.getcwd(), "ms1_fitting_data.pkl")
        with open(dump_path, "wb") as f:
            pickle.dump(dump_data, f)
        logger.info(f"Wrote MS1 fitting data for {len(dump_data['precursors'])} precursors to {dump_path}")

    return all_group_pearson, all_ms1, all_coeff, all_iso, all_group_keys, all_fitted, new_output_dict, fake_fdc_dict


def get_seqs_and_mzs(fdc_group, timeplex, tag, key, SILAC):
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
    SILAC : massTag
        A massTag instance
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
    group_prec_mzs = tag_group["mz"]
    group_prec_seqs = tag_group["seq"]
    group_prec_z = key[1]
    if timeplex:
        time_channel = key[2]
    else:
        time_channel = None
    largest_id = np.argmax(tag_group["coeff"])
    top_ms1_spec_idx = list(tag_group["Ms1_spec_id"])[largest_id]
    prec_rt = list(tag_group["rt"])[largest_id]
    
    if (tag.name != "no_tag" and SILAC is not None):
        
        all_comb = get_other_channels((group_prec_seqs.iloc[largest_id],group_prec_z), group_prec_mzs.iloc[largest_id], [tag,SILAC])
        prec_seqs,prec_mzs = tuple(zip(*all_comb))
    elif tag.name != "no_tag":
        all_comb = get_other_channels((group_prec_seqs.iloc[largest_id],group_prec_z), group_prec_mzs.iloc[largest_id], [tag])
        prec_seqs,prec_mzs = tuple(zip(*all_comb))
    elif SILAC is not None:
        all_comb = get_other_channels((group_prec_seqs.iloc[largest_id],group_prec_z), group_prec_mzs.iloc[largest_id], [SILAC])
        prec_seqs,prec_mzs = tuple(zip(*all_comb))
    else:
        prec_seqs = (group_prec_seqs.iloc[largest_id],)
        prec_mzs = (group_prec_mzs.iloc[largest_id],)

    largest_coeff_scans = list(tag_group["Ms1_spec_id"])

    return prec_seqs, prec_mzs, group_prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel


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
    bounded by `window_half_width`.

    When IM data is present, filters to spectra sharing the same IM bin as the
    reference scans, using RT-based windowing within that bin.
    """
    # Check if IM data is present
    has_im = len(ms1_spectra) > 0 and getattr(ms1_spectra[0], 'im_lo', None) is not None

    if has_im:
        # Determine target IM bin from reference scan
        ref_scan = largest_coeff_scans[0]
        ref_idx = list(ms1_spec_idxs).index(ref_scan)
        ref_spec = ms1_spectra[ref_idx]
        target_im_lo, target_im_hi = ref_spec.im_lo, ref_spec.im_hi

        # Filter ms1 indices to same IM bin
        im_filtered_idxs = [i for i, s in enumerate(ms1_spectra)
                            if s.im_lo == target_im_lo and s.im_hi == target_im_hi]
        im_filtered_scan_nums = np.array([ms1_spec_idxs[i] for i in im_filtered_idxs])

        # Find min/max within filtered set
        max_scan, min_scan = max(largest_coeff_scans), min(largest_coeff_scans)
        # Find positions within the IM-filtered list
        filtered_list = list(im_filtered_scan_nums)
        if min_scan in filtered_list:
            idx_min = filtered_list.index(min_scan)
        else:
            idx_min = np.searchsorted(im_filtered_scan_nums, min_scan)
        if max_scan in filtered_list:
            idx_max = filtered_list.index(max_scan)
        else:
            idx_max = np.searchsorted(im_filtered_scan_nums, max_scan)

        lo = max(0, idx_min - window_half_width)
        hi = min(len(im_filtered_scan_nums), idx_max + window_half_width + 1)
        scans_each_side = im_filtered_scan_nums[lo:hi]
        all_scans = list(scans_each_side)
        spectra_subset = [all_spectra.get_by_idx(idx) for idx in all_scans]
    else:
        ## max and min of this list
        max_scan, min_scan = max(largest_coeff_scans), min(largest_coeff_scans)
        ms1_list_idx_min = list(ms1_spec_idxs).index(min_scan)
        ms1_list_idx_max = list(ms1_spec_idxs).index(max_scan)
        scans_each_side = np.array(ms1_spec_idxs)[np.arange(max(0,ms1_list_idx_min-window_half_width),min(len(ms1_spectra),ms1_list_idx_max+window_half_width+1))]
        all_scans = list(scans_each_side)
        spectra_subset = [all_spectra.get_by_idx(idx) for idx in all_scans]

    return all_scans, spectra_subset



def _lookup_group(source, key):
    """Return a pandas DataFrame slice for ``key`` or ``None`` if absent.

    Accepts either a pandas DataFrameGroupBy (legacy callers and tests) or a
    ``(polars.DataFrame, offsets_dict)`` tuple. The tuple form avoids
    materializing the full post-NNLS frame as pandas — only the per-precursor
    slice gets converted.
    """
    if isinstance(source, tuple):
        polars_df, offsets = source
        if key not in offsets:
            return None
        start, n = offsets[key]
        return polars_df.slice(start, n).to_pandas()
    if key not in source.groups:
        return None
    return source.get_group(key)


def get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs):
    """
    For a given precursor, get the ms2 JMod coefficients at MS2 scan #s and return this dictionary,
    along with the ms1 scan number corresponding to the highest scan

    Parameters
    ----------
    prec_seq : str
        Precursor non-stripped sequence
    prec_z : int or float
        precursor charge
    prec_rt : float
        Precursor retention time, not currently used
    time_channel : int or str or None
        Precursor time channel
    timeplex : bool
        Bool for timeplex
    grouped_decoy_coeffs : pandas DataFrameGroupBy or (polars.DataFrame, dict)
        Either a pandas groupby (legacy) or a ``(slim_polars_df, offsets)``
        tuple where ``offsets`` maps the channel key to ``(start, length)`` row
        ranges in the polars frame. The tuple form is used in production to
        avoid materializing the full post-NNLS frame as pandas.
    ms2_rt : 1D Numpy array
        Array of retention time of all ms2 scans in MS run
    rt_tol : float
        Retention time tolerance
    prec_mz : float
        Precursor mz
    bottom_of_window: 1D Numpy array
        Array of the bottom of the isolation window for every ms2 scan (same size as ms2_rt)
    top_of_window: 1D Numpy array
        Array of the top of the isolation window for every ms2 scan (same size as ms2_rt)
    ms2_spec_idxs: 1D Numpy array
        Array of the scan index of every ms2 scan (same size as ms2_rt)


    Returns
    -------
    ms2_vals : dict[int] == float
        A dictionary with keys as ms2 scan indexes and values as the JMod MS2 coeff
    highest_ranked_spec : int
        The ms1 scan index from the highest ranked ms2 Jmod coeff
    channel_key : tuple of (str, int) or (str, int, int)
        Tuple of unstripped sequence, charge, and time channel if applicable

    """
    offset = 0

    if timeplex:
        channel_key = (prec_seq,prec_z,time_channel)
    else:
        channel_key = (prec_seq,prec_z)

    ## create dummy
    ms2_vals = {0:0}

    group = _lookup_group(grouped_decoy_coeffs, channel_key)
    if group is not None:

        # use coeff as rank score directly
        highest_ranked_spec = group.loc[group["coeff"].idxmax(), "Ms1_spec_id"]

        # ms2_rt_bool = np.abs(ms2_rt-prec_rt)<rt_tol
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

    return ms2_vals, highest_ranked_spec, channel_key



def get_isotopes_and_vals(prec_seq, prec_z, num_iso, tags, all_scans, prec_mz, mz_ppm, spectra_subset, interp_func):
    isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tags)
    all_isotope_traces = get_isotope_traces_vectorized(isotopes, mz_ppm, spectra_subset)
    all_ms2_vals = fill_scan_values(all_scans, interp_func, all_isotope_traces[0])

    return all_isotope_traces[0], all_ms2_vals, all_isotope_traces[1:], isotopes, interp_func


def build_ms2_interpolator(ms2_vals):
    return interp1d(list(ms2_vals.keys()), np.array(list(ms2_vals.values())), bounds_error=False)   



def get_isotope_traces_vectorized(isotopes, mz_ppm, spectra_subset):
    """
    Get a trace of each isotopic peak across a set of MS1 spectra

    Parameters
    ----------
    isotopes : list of brainpy theoretical peak objects
        Peak objects (with p.mz, p.intensity, and p.charge) for one precursor
    mz_ppm : float
        MS1 tolerance
    spectra_subset : list of Spectrum Objects
        A list of Sprectrum Objects
    
    Returns
    -------
    all_isotope_traces : list of dicts[int] = float
        A list where each element corresponds to one isotope.
        Each element is a dictionary mapping MS1 scan numbers (`spec.scan_num`)
        to matched peak intensities for that isotope.

    Notes
    -------
    note: we have collected similar values for previous channel if the isotopic envelopes are overlapping. 
    However, in cases like diethlyation, isoptopes can differ by > 10 ppm #!!!Maybe investigate wider ppm tol for these cases?
    """
    all_isotope_traces=[{} for _ in isotopes]

    for spec in spectra_subset:
        iso_ints = get_trace_int_numba(spec.mz, spec.intens, np.array([isotope.mz for isotope in isotopes]), mz_ppm, min_int)
        for i in range(len(isotopes)):
            all_isotope_traces[i][spec.scan_num] = iso_ints[i]

    
    return all_isotope_traces

@nb.njit(cache=False)
def get_trace_int_numba(spec_mz, spec_intens, mz_array, rtol, base):
    """
    Match isotope mz values to the closest peaks in a spectrum and return their intensities.

    Parameters
    ----------
    spec_mz : 1D numpy array
        Array of mz floats from spec
    spec_intens : 1D numpy array
        Array of intensity floats from spec
    mz_array : 1D numpy array
        Array of mzs of the isotopes to be matched
    rtol : float
        The mz ppm
    base : float
        Minimum intensity assigned when the peak is unmatched
    
    Returns
    -------
    vals_to_return : 1D numpy array
        Array of the intensities of matched peaks (or base for unmatched peaks) of length mz_array

    """
    if spec_mz.size == 0 or spec_intens.size == 0:
        return np.full(mz_array.shape, base, dtype=np.float64)
    
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



def compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tags):
    """
    Gets a list of brainpy theoretical peak isotopes for a precursor

    Parameters
    ----------
    prec_seq : str
        precursor unstripped Sequence
    prec_mz : float
        percursor_mz
    prec_z : float or int
        precursor charge
    num_iso : int
        number of isotopes to return
    tag : massTag
        massTag instance
    
    Returns
    -------
    isotopes : list of brainpy theoretical peak objects
        A list of peak objects with p.mz, p.intensity, and p.charge of length num_iso

    Notes
    -------
    Mostly this is just the function iso.precursor_isotopes. However, if a tag doesn't have compositions, 
    (or the mass if otherwise off for some reason?) then the other parts of the code correct and make sure
    isotopes[0] == prec_mz

    """
    isotopes = iso.precursor_isotopes(prec_seq,prec_z,tags,num_iso, decoys=False)

    delta_mz = prec_mz-isotopes[0].mz
    for i in isotopes:
        i.mz+=delta_mz

    return isotopes


def fill_scan_values(all_scans, interp_func, ms1_vals):
    """
    Interpolate MS2 intensities at the scan numbers corresponding to MS1 spectra.

    Parameters
    ----------
    all_scans : list of int
        List of scans to consider
    interp_func : scipy interp1d object 
        An interpolation function from ms2 vals
    ms1_vals : dict[int] = float
        A dictionary with MS1 scan #s as keys and intesnity as values
        ms1_vals.keys() == all_scans for all cases as far as I can tell but I've left the conditional in
    
    Returns
    -------
    all_ms2_vals : dict[int] = float
        A dictionary with MS1 scan #s as keys and interpolated MS2 intensities as values

    Notes
    -------
    This function lines up the MS1 and MS2 traces by interpolating the value of the MS2 trace at the MS1 scan #

    """
    all_ms2_vals = {i: min_int for i in all_scans}

    interp_vals = interp_func(all_scans)

    for scan, c in zip(all_scans, interp_vals):
        if scan in ms1_vals:
            all_ms2_vals[scan] = c

    return all_ms2_vals



def get_ms1_index_of_max(ms2_vals, top_ms1_spec_idx, highest_ranked_spec):
    """
    Return the best MS1 spec idx to be used

    Parameters
    ----------
    ms2_vals : dict[int] = float
        Dictionary with keys of MS1 spec idxs and values of corresponding intensities
    top_ms1_spec_idx : int 
        MS1 spec idx from get_seqs_and_mzs
    highest_ranked_spec : int
        MS1 spec idx from get_ms2_vals
    
    Returns
    -------
    ms1_index_of_max : int
        The max index to be used going forwards

    """

    ## use monoiso ms1 prec mz to find the elution ms1 peak
    if ms2_vals=={0:0}:
        ms1_index_of_max = top_ms1_spec_idx ## should I just use the max of MS1???? Need to look into again
    else:
        ms1_index_of_max = highest_ranked_spec
    return ms1_index_of_max


def filter_all_scans(all_scans, ms1_peak_edge_idxs, all_ms1_vals, all_iso_vals, all_ms2_vals):
    """
    Filter all scans and corresponding values to only those within the MS1 peak edges.
    """
    channel_scans = all_scans[all_scans.index(ms1_peak_edge_idxs[0]):all_scans.index(ms1_peak_edge_idxs[1])+1]
    all_ms1_vals = {i:all_ms1_vals[i] for i in channel_scans}
    all_iso_vals = [{i:iso_vals[i] for i in channel_scans} for iso_vals in all_iso_vals]
    all_ms2_vals = {i:all_ms2_vals[i] for i in channel_scans}

    return channel_scans, all_ms1_vals, all_iso_vals, all_ms2_vals


def compute_ms1_ms2_cors(all_ms2_vals, all_ms1_vals, all_iso_vals, num_iso_r, channel_scans, isotopes):
    """
    Return ms1/ms2 pearson correlation and isotope ratio

    Parameters
    ----------
    all_ms2_vals, all_ms1_vals, all_iso_vals: dict[int] : float
        Dictionaries of intensities for ms2 (interpolated), the first isotopic 
        ms1 peak, and other ms1 isotopic peaks at given ms1 scan idxs
    num_iso_r : int
        From config, the number of isotopes to get a pearson correlation against ms2 for
    channel_scans : list of int
        A list of ms1 spec idxs
    isotopes : list of Brainpy theoretical peak objects
        A list of peaks with peak.mz, peak.intensity, and peak.charge
    
    Returns
    -------
    spec_pearsons : list of float
        A list of pearson correlations between the ms2 intensity and the ms1 intensity of each isotope up to num_iso_r+1
    iso_ratio : list of [PearsonRResult, list of float, list of float]
        list[0] : PearsonRResult object with .statistic and .pvalue for the correlation between list[1] and list[2]
        list[1] : Theoretical Isotopic Intensities from the theoretical peak isotopes
        list[2] : Observed isotopic intensities of each isotope
    
    Notes
    ------
    Spec_pearsons is a result that encompasses all relevant scans
    Iso_ratios is extracted from the scan with the highest ms2 value only

    """
    spec_pearsons = [np_pearson_cor(list(all_ms2_vals.values()),list(i.values())).statistic for i in [all_ms1_vals,*all_iso_vals[:num_iso_r]]]

    ms1_spec_idx = channel_scans[np.argmax(list(all_ms2_vals.values()))]

    theoretical_pattern = [i.intensity for i in isotopes]
    obs_pattern = [all_ms1_vals[ms1_spec_idx],*[iso_trace[ms1_spec_idx] for iso_trace in all_iso_vals]]
    iso_ratio = [np_pearson_cor(theoretical_pattern,obs_pattern),theoretical_pattern,obs_pattern]

    return spec_pearsons, iso_ratio


def select_scans_to_search(top_ms1_spec_idx, all_scans, all_channel_scans, window_half_width):
    "Aggregate a sorted list of all unique scans collected from every channel"
    # idx_of_max =all_scans.index(top_ms1_spec_idx)
    # scans_to_search = np.array(all_scans)[np.arange(max(0,idx_of_max-window_half_width),min(len(all_scans),idx_of_max+window_half_width+1))]
    scans_to_search = np.sort(np.unique(np.concatenate(all_channel_scans)))
    return scans_to_search


def fit_isotopes_and_score(ms1_spectra, ms1_spec_idxs, ms1_spec_idx, group_iso, mz_ppm):
    spec = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]

    pred_coeff, obs_peaks, fit_matrix, kept_mz = fit_channel_isotopes_numba(spec,group_iso,mz_ppm)

    if len(obs_peaks)==0:
        fit_cor = np.nan
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_cor = np_pearson_cor(np.sum(fit_matrix*pred_coeff,1),obs_peaks)

    return pred_coeff, obs_peaks, fit_matrix, fit_cor, kept_mz

# def add_to_ms1_spec_dict(ms1_spec_dict, ms1_spectra, ms1_spec_idxs, group_iso, key, fdc_group, filtered_decoy_coeffs, tag, scans_to_search, fdc_group_idx, dummy_idx_list, fake_fdc_dict):
#     """
#     As each fdc group is iterated through, it will be passed into this function to be
#     added to one ms1_spec_dict that contains all the fitting information for all spectra

#     Parameters
#     ----------
#     ms1_spec_dict : dict
#         Has been initialized as {k: {"fdc_idx": [], "peak_mz": [], "rel_iso_int": [], "monoiso_groups": [], "MS1_spectra": None} for k in ms1_spec_idxs}
#     ms1_spectra : Spectrum
#         The spectrum object
#     ms1_spec_idxs : list of int
#         List of ms1 spec idxs
#     group_iso : list of list of brainpy theoretical peak objects (one for each channel)
#         group_iso contains a list of outputs from compute_isotopes corresponding to each channel
#     key : tuple
#         The key identifying the group in fdc_group:
#         - (seq, z) if not timeplex
#         - (seq, z, time_channel) if timeplex
#     fdc_group : pandas.core.groupby.generic.DataFrameGroupBy
#         Grouped FDC DataFrame, either by ('seq', 'z') or ('seq', 'z', 'time_channel') if timeplex.
#     filtered_decoy_coeffs : pandas df
#     tag : massTag instance
#     scans_to_search: list of int
#         list of scan indexes for each precursor to be searched in
#     fdc_group_idx: int
#         the index of the fdc group
#     dummy_idx_list: list of int
#         Starts as list of fdc indexes. Dummy (+1) gets appended every time a channel is not found
#     fake_fdc_dict: dict
#         starts as an empty dict, and through iterations get filled with (for channels that are not identified)
#         key -> dummy_idx 
#         value -> (fdc_group_idx, channel)
        

#     Returns
#     -------
#     ms1_spec_dict : dict of dict
#         keys = ms1_spec_idxs
#         values = dict as shown in same entry above but filled
#     fake_fdc_dict : dict
#         see same entry above
#     dummy_idx_list : list of int
#         a list of indicies that gets [-1]+1 appended to it every time a channel is not identified

#     Notes: See comments below for important assumptions that might not always be true, and for more granular information

#     """

#     #see which channels are observed in an fdc group
#     idxs = fdc_group.groups[key]
#     rows = filtered_decoy_coeffs.loc[idxs]
#     # This all assumes tag.channel_names is ordered, as well as group_iso is ordered
#     row_channels = rows["channel"].to_numpy()
#     observed_channels = []
#     for channel in tag.channel_names:
#         if int(channel) in row_channels:
#             observed_channels.append(channel)
    
#     # for each channel in the tag:
#         #if it isnt observed at to dummy_idx_list and fake_fdc_dict
#         #if it is observed get the fdc index of the channel
#     for i, channel in enumerate(tag.channel_names):
#         if not channel in observed_channels:
#             # This assumes dummy_idx_list is sorted
#             if dummy_idx_list[-1]+1 in dummy_idx_list:
#                 raise ValueError("Dummy Index List +1 is in Dummy Index List Already")
#             dummy_idx_list.append(dummy_idx_list[-1]+1)
#             fdc_index = dummy_idx_list[-1]
#             fake_fdc_dict[fdc_index] = (fdc_group_idx, channel)
#         else:
#             fdc_index = rows.index[rows["channel"] == int(channel)]
#             if len(fdc_index) == 0:
#                 raise ValueError("This is a valueError")
#             elif len(fdc_index) > 1:
#                 raise ValueError(f"Multiple matches for channel {channel}")
#             else:
#                 fdc_index = fdc_index[0]

#         # get the entry from group_iso that corresponds to said channel
#         iso_group = group_iso[i]

#         #then, add these as corresponding mzs, intensities, and fdc_idxs to three lists in the ms1_spec_dict list
#         for ms1_spec_idx in scans_to_search:
#             for peak in iso_group:
#                 ms1_spec_dict[ms1_spec_idx]["peak_mz"].append(peak.mz)
#                 ms1_spec_dict[ms1_spec_idx]["rel_iso_int"].append(peak.intensity)
#                 ms1_spec_dict[ms1_spec_idx]["fdc_idx"].append(fdc_index)

#         #finally if the spectra hasnt been added, add the spectra to the dict
#         if ms1_spec_dict[ms1_spec_idx]["MS1_spectra"] is None:
#             ms1_spec_dict[ms1_spec_idx]["MS1_spectra"] = ms1_spectra[np.where(ms1_spec_idxs==ms1_spec_idx)[0][0]]
    
#     #Im not sure why monoiso groups is in here. It could probably be removed
#     for ms1_spec_idx in scans_to_search:
#         ms1_spec_dict[ms1_spec_idx]["monoiso_groups"].append([iso_group[0].mz for iso_group in group_iso])

#     return ms1_spec_dict, fake_fdc_dict, dummy_idx_list


# def fit_all_ms1_specs(ms1_spec_dict, new_output_dict, mz_ppm):
#     """
#     Calls fit_whole_ms1_spectum for each spectra in the dict and adds it to new_output_dict

#     Parameters
#     ----------
#     ms1_spec_dict : dict
#         Has been initialized as {k: {"fdc_idx": [], "peak_mz": [], "rel_iso_int": [], "monoiso_groups": [], "MS1_spectra": None} for k in ms1_spec_idxs}
#         and filled by add_to_ms1_spec_dict()
#     new_output_dict : dict
#         an empty dict
#     mz_ppm : float
#         m/z tolerance in ppm

#     Returns
#     -------
#     new_output_dict : dict
#         a dictionary with keys of ms1_spec_idx and values of a dictionary with
#             key value pairs of the outputs from fit_whole_ms1_spectrum for that spec
#     """
#     for ms1_spec_idx, dicts in tqdm.tqdm(ms1_spec_dict.items()):
#         if dicts["MS1_spectra"] == None:
#             continue
#         mapped_pred_coeffs, spectra_obs_peaks, spectra_sparse_matrix = fit_whole_MS1_spectrum(dicts, mz_ppm, ms1_spec_idx)
#         new_output_dict[ms1_spec_idx] = {
#                                         "mapped_pred_coeffs": mapped_pred_coeffs,
#                                         "spectra_obs_peaks": spectra_obs_peaks,
#                                         "spectra_sparse_matrix": spectra_sparse_matrix
#                                         }
#     return new_output_dict

# from scipy.sparse import coo_matrix
# from scipy.optimize import lsq_linear
# def fit_whole_MS1_spectrum(dicts, mz_ppm, ms1_spec_idx, fit_type="b"):
#     """
#     Reads an MS1_spec_dict entry to fit an entire MS1 spectrum at a time as a sparse matrix

#     Parameters
#     ----------
#     dicts : dict
#         {"fdc_idx": [], "peak_mz": [], "rel_iso_int": [], "MS1_spectra": Spectrum}
#     mz_ppm : float
#         The m/z tolerance in ppm
#     ms1_spec_idx : int
#         The index of the current ms1 scan. Not currently used
#     fit_type : str
#         The type of fit being used. Currently supports "b", "c", and "no_penalty"

#     Returns
#     -------
#     mapped_lib_coefficients : dict
#         A dictionary with keys of fdc_idx and values of their coefficient
#     dia_spec_int : list of floats
#         A list containing all observed peak intensities (or padded 0s) in the spectrum that made it into the matrix
#     dense_matrix : 2d numpy array
#         The matrix used for fitting
#     """

#     spec = dicts["MS1_spectra"]
#     mz_peaks = spec.mz
#     dia_spec_int = spec.intens

#     offsets = mz_ppm * mz_peaks
#     centroid_breaks = np.empty(mz_peaks.size * 2, dtype=mz_peaks.dtype)
#     centroid_breaks[:mz_peaks.size] = mz_peaks - offsets
#     centroid_breaks[mz_peaks.size:] = mz_peaks + offsets
#     centroid_breaks.sort()


#     all_mz = dicts["peak_mz"]
#     ref_coords = np.searchsorted(centroid_breaks, all_mz)

#     lib_peaks_matched = np.array((ref_coords % 2 == 1).tolist())

#     num_lib_peaks_unmatched = np.count_nonzero(~lib_peaks_matched)

#     unique_fdcs = (set(dicts["fdc_idx"]))
#     fdc_to_col = {fdc: i for i, fdc in enumerate(unique_fdcs)}

#     last_unnasigned_peak_idx = len(mz_peaks)
#     if not len(ref_coords) == len(lib_peaks_matched) == len(dicts["rel_iso_int"]) == len(dicts["peak_mz"]) == len(dicts["fdc_idx"]):
#         raise ValueError("Length of lists for fitting is not the same")
    
#     sparse_rows = []
#     sparse_cols = []
#     sparse_vals = []
#     fdc_idx_dummy_peak_dict = {k: None for k in unique_fdcs}
#     for fdc_idx, mz, intensity, ref_coord, matched in zip(dicts["fdc_idx"],
#                                                             dicts["peak_mz"],
#                                                             dicts["rel_iso_int"],
#                                                             ref_coords,
#                                                             lib_peaks_matched
#                                                             ):

#         if fit_type == "c":
#             if matched:
#                 ms1_peak_coord = (ref_coord-1)//2
#             else:
#                 last_unnasigned_peak_idx += 1
#                 ms1_peak_coord = last_unnasigned_peak_idx

#         elif fit_type == "b":
#             if matched:
#                 ms1_peak_coord = (ref_coord-1)//2
#             else:
#                 if fdc_idx_dummy_peak_dict[fdc_idx] == None:
#                     last_unnasigned_peak_idx += 1
#                     fdc_idx_dummy_peak_dict[fdc_idx] = last_unnasigned_peak_idx
#                 ms1_peak_coord = fdc_idx_dummy_peak_dict[fdc_idx]
#         elif fit_type == "no_penalty":
#             if matched: 
#                 ms1_peak_coord = (ref_coord-1)//2


#         sparse_rows.append(ms1_peak_coord)
#         sparse_cols.append(fdc_to_col[fdc_idx])
#         sparse_vals.append(intensity)

#     matrix_sparse = coo_matrix((sparse_vals, (sparse_rows, sparse_cols)))
#     dia_spec_int = np.append(dia_spec_int,[0]*(matrix_sparse.shape[0]-dia_spec_int.shape[0])) 

#     # matrix_sparse = coo_matrix((sparse_vals, (sparse_rows, sparse_cols)), shape=(len(mz_peaks), len(unique_fdcs)))

#     matrix_csr = matrix_sparse.tocsr()
#     mask = np.diff(matrix_csr.indptr) != 0
#     matrix_sparse = matrix_csr[mask].tocoo()
#     dia_spec_int = dia_spec_int[mask]


#     res = lsq_linear(matrix_sparse, dia_spec_int, bounds=(0, np.inf), method="trf", lsmr_tol="auto")
#     lib_coefficients = res.x
#     lib_coefficients = lib_coefficients.flatten()
#     mapped_lib_coefficients = {k: v for k, v in list(zip(unique_fdcs, lib_coefficients))}

    
#     return mapped_lib_coefficients, dia_spec_int, matrix_sparse





    

# def get_other_channels(prec,mz,tag):
#     """
#     Return the m/z and sequences for all channels of a given precursor.

#     Parameters
#     ----------
#     prec : tuple
#         A tuple (unstripped_seq, z) containing the peptide sequence (with tags) 
#         and its charge state.
#     mz : float
#         The m/z value of the precursor in the current channel.
#     tag : massTag
#         A massTag instance.

#     Returns
#     -------
#     channel_dict : dict
#         A dictionary mapping each channel name (e.g. 'PSMtag_5plex-0') to a list:
#         [unstripped_seq, float(mz)] for that channel.
#     """
    
#     ## identify what channel the current prec is in
#     channels = re.findall(rf"({tag.name}-\d+)",prec[0])
#     num_tags = len(channels)
#     assert len(set(channels))==1, f"{channels}"
#     channel = channels[0]
#     assert channel in tag.mass_dict
#     channel_dict = {i:[] for i in tag.mass_dict}
    
#     for c in channel_dict:
#         if c==channel:
#             channel_dict[channel] = [prec[0],mz]
#         else:
#             c_seq = re.sub(channel,c,prec[0])
#             c_mz = mz + (num_tags*(tag.mass_dict[c]-tag.mass_dict[channel])/prec[1])
#             channel_dict[c] = [c_seq,c_mz]
#     return channel_dict





def get_other_channels(prec, mz, tags):
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
    all_prec_comb : list
        A list of tuples with each precursor-channel combination and their m/z:
        [unstripped_seq, float(mz)] for that channel.
    """
    ## check which tags are in use and update
    tags = [tag for tag in tags if len(re.findall(rf"{tag.name}-(\d+)",prec[0]))]
    ## identify what channel the current prec is in
    all_channels = [re.findall(rf"{tag.name}-(\d+)",prec[0]) for tag in tags]
    channels_top = tuple([i[0] for i in all_channels])
    num_tags = [len(i) for i in all_channels]
    
    # Get all channel combinations (one channel per tag)
    channel_options = [tag.channel_names for tag in tags]
    all_combinations = list(product(*channel_options))
    
    all_prec_comb = []
    for combination in all_combinations:
        # break
        # if channels==combination:
        #     all_prec_comb.append([prec[0],mz])
        # else:
        last_channels = channels_top
        new_seq = prec[0]
        new_mz = mz
        # For each tag, replace ALL occurrences with the same channel
        for tag, channel, n, l in zip(tags, combination, num_tags, last_channels):
            # break
            # Pattern to match tag-channel combinations
            pattern = rf'{tag.name}-\d+'
            replacement = f'{tag.name}-{channel}'
            new_seq = re.sub(pattern, replacement, new_seq)
            
            new_mz = new_mz + (n*(tag.mass_dict[tag.name+"-"+channel]-tag.mass_dict[tag.name+"-"+l])/prec[1])
            # print(new_seq, combination,new_mz)
        all_prec_comb.append([new_seq,new_mz])
    
    return all_prec_comb




def get_ms1_peak(x,y,idx,additional_scans):
    """
    Get the peak apex and bounds for a given set of monoisotopic intensities

    Parameters
    ----------
    x : list of int
        A list of MS1 spec idxs (from ms1_vals)
    y : list or array of float
        The moving averages of MS1 spec intensities (from ms1_vals)
    idx : int
        The MS1 spec idx used to determine the peak (important when there is more than one "peak" found)
    additional_scans : int
        The number of scans to add to each side of the peak edges

    Returns
    -------
    ms1_peak_idx : int
        The MS1 index of the peak
    ms1_peak_edge_idxs : 1D numpy array of int (len==2)
        The first and last MS1 spec idx that is a part of the peak
    """
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
    peak_edge_idxs = [max(0,peak_attr["left_bases"][closest_idx]-additional_scans),min(len(x)-1,peak_attr["right_bases"][closest_idx]+additional_scans)]
    # peak_edge_idxs = [peak_attr["left_bases"][closest_idx],peak_attr["right_bases"][closest_idx]]
    
    return x[peak_idx],x[peak_edge_idxs]



def moving_average(x, w=4):
    return np.convolve(x, np.ones(w), 'same') / w


def _compute_cooks_d(fit_matrix, obs_peaks, pred_coeff):
    """Compute Cook's distance and residuals for each observation in the NNLS fit."""
    n, p = fit_matrix.shape
    predicted = fit_matrix @ pred_coeff
    residuals = obs_peaks - predicted
    if n <= p or np.sum(residuals**2) == 0:
        return np.zeros(n), residuals
    mse = np.sum(residuals**2) / (n - p)
    try:
        hat_diag = np.einsum('ij,ji->i', fit_matrix,
                             np.linalg.pinv(fit_matrix.T @ fit_matrix) @ fit_matrix.T)
        hat_diag = np.clip(hat_diag, 0, 1 - 1e-12)
        cooks_d = (residuals**2 * hat_diag) / (p * mse * (1 - hat_diag)**2)
    except np.linalg.LinAlgError:
        cooks_d = np.zeros(n)
    return cooks_d, residuals



def fit_channel_isotopes_numba(spec, all_iso, mz_ppm):
    """
    Takes observed peaks fom spectra and expected peaks from from all_iso for one plex-group and builds a matrix to fit them

    Parameters
    ----------
    spec : Spectrum Object
        The spectrum to look in
    all_iso : a list of list of brainpy theoretical peak instances
        i.e.
        [
        [Channel_1_Peak_1, Channel_1_Peak_2, ...]
        [Channel_2_Peak_1, Channel_2_Peak_2, ...]
        etc
        ]
        where the peaks are brainpy theoretical peaks with p.mz, p.intensity, and p.charge
    mz_ppm : float
        MS1 ppm tolerance

    Returns
    -------
    lib_coefficients : 1d numpy array
        An array containing the coefficient for each channel
    dia_spec_int : list of floats
        A list containing all observed peak intensities in the spectrum that made it into the matrix
    dense_matrix : 2d numpy array
        The matrix used for fitting
    kept_mz : 1d numpy array
        The m/z values of the kept rows
    Notes
    -------
    Spectra are matched within mz tolerance between observed (from spectrum) and theoretical (from all_iso) peaks.
    A matrix is created with columns for each channel and rows for each observed peak. Only rows with both
    observed signal and at least one library entry are kept. The matrix is fit with plain NNLS.

    """

    flat = np.array([(p.mz, p.intensity) for iso in all_iso for p in iso], dtype=np.float64)
    ms1_iso_patterns = flat.reshape(len(all_iso), len(all_iso[0]), 2)

    dia_spectrum = np.array(spec.peak_list(), dtype=np.float64).T

    group_lengths = np.array([len(g) for g in all_iso])

    dense_matrix, dia_spec_int, lib_peaks_matched, fdc_idxs, grp_lengths = get_matrix_to_fit_numba(ms1_iso_patterns, group_lengths, dia_spectrum, len(all_iso), mz_ppm)

    # Exclude channels where neither of the two most intense expected peaks
    # was observed.  ms1_iso_patterns[ch, :, 1] gives the theoretical
    # intensities for channel ch; the top-2 indices into that array identify
    # the two most intense peaks.  lib_peaks_matched is flat (one entry per
    # peak across all channels), so we slice it per channel using group_lengths.
    n_iso_per_ch = ms1_iso_patterns.shape[1]
    offset = 0
    for ch in range(len(all_iso)):
        intensities = ms1_iso_patterns[ch, :, 1]
        top2 = np.argsort(intensities)[-2:]  # indices of two most intense peaks
        matched_slice = lib_peaks_matched[offset:offset + group_lengths[ch]]
        if not np.any(matched_slice[top2]):
            dense_matrix[:, ch] = 0.0
        offset += group_lengths[ch]

    # Replicate the spectrum windowing from get_matrix_to_fit_numba to get mz values
    mz_full = dia_spectrum[:, 0]
    lo = np.searchsorted(mz_full, ms1_iso_patterns[:, :, 0].min() - 1, side="right")
    hi = np.searchsorted(mz_full, ms1_iso_patterns[:, :, 0].max() + 1, side="left")
    windowed_mz = mz_full[lo:hi]
    n_dummy = dense_matrix.shape[0] - len(windowed_mz)
    all_mz = np.append(windowed_mz, np.zeros(max(0, n_dummy)))

    # Keep rows with a library entry (regardless of observed signal)
    keep = np.any(dense_matrix != 0, axis=1)
    dense_matrix = dense_matrix[keep, :]
    dia_spec_int = dia_spec_int[keep]
    kept_mz = all_mz[keep]

    if dense_matrix.shape[0] == 0:
        return (
            np.zeros(len(all_iso), dtype=float),
            dia_spec_int,
            dense_matrix,
            kept_mz,
        )

    lib_coefficients, _ = optimize.nnls(dense_matrix, dia_spec_int)
    lib_coefficients[lib_coefficients < 1] = 0.0

    return lib_coefficients, dia_spec_int, dense_matrix, kept_mz



@nb.njit(cache=False)
def get_matrix_to_fit_numba(ms1_iso_patterns, group_lengths, dia_spectrum, all_iso_len, mz_ppm):    
    ### we only need to conseider the part of the spectrum that falls within the isotopic envelopes of the channels
    min_isotope = ms1_iso_patterns[:, :, 0].min() - 1
    max_isotope = ms1_iso_patterns[:, :, 0].max() + 1

    mz = dia_spectrum[:, 0]
    lo = np.searchsorted(mz, min_isotope, side="right")
    hi = np.searchsorted(mz, max_isotope, side="left")
    dia_spectrum = dia_spectrum[lo:hi]

    mz_peaks = dia_spectrum[:, 0]

    fdc_idxs = np.repeat(np.arange(all_iso_len), group_lengths)
    all_mz = ms1_iso_patterns[:,:,0].ravel()
    rel_iso_int = ms1_iso_patterns[:, :, 1].ravel()

    # For each library peak, find ALL observed peaks within ppm tolerance,
    # sum their intensities, and record the closest as the canonical row index.
    n_lib = all_mz.size
    n_obs = mz_peaks.size
    lib_peaks_matched = np.empty(n_lib, dtype=np.bool_)
    matched_obs_idx = np.empty(n_lib, dtype=np.int64)
    # Summed observed intensity for each library peak (across all matching obs peaks)
    summed_obs = np.zeros(n_lib, dtype=np.float64)
    obs_int = dia_spectrum[:, 1]

    for i in range(n_lib):
        lib_mz = all_mz[i]
        pos = np.searchsorted(mz_peaks, lib_mz)

        best_idx = -1
        best_dist = np.inf
        intensity_sum = 0.0

        # Scan left from insertion point
        j = pos - 1
        while j >= 0:
            obs_mz = mz_peaks[j]
            dist = abs(lib_mz - obs_mz)
            tol = mz_ppm * obs_mz
            if dist > tol:
                break
            intensity_sum += obs_int[j]
            if dist < best_dist:
                best_dist = dist
                best_idx = j
            j -= 1

        # Scan right from insertion point
        j = pos
        while j < n_obs:
            obs_mz = mz_peaks[j]
            dist = abs(lib_mz - obs_mz)
            tol = mz_ppm * obs_mz
            if dist > tol:
                break
            intensity_sum += obs_int[j]
            if dist < best_dist:
                best_dist = dist
                best_idx = j
            j += 1

        lib_peaks_matched[i] = (best_idx >= 0)
        matched_obs_idx[i] = best_idx
        summed_obs[i] = intensity_sum

    last_unassigned_peak_idx = len(mz_peaks)

    unique_fdcs = (set(fdc_idxs))

    # Group unmatched library peaks by m/z proximity so that overlapping
    # peaks from different channels share a matrix row (just like matched
    # peaks that hit the same observed peak share a row).
    n_unmatched = np.sum(~lib_peaks_matched)

    # Collect indices of unmatched peaks
    unmatched_idx = np.empty(n_unmatched, dtype=np.int64)
    k = 0
    for i in range(n_lib):
        if not lib_peaks_matched[i]:
            unmatched_idx[k] = i
            k += 1

    # Sort unmatched peaks by m/z
    unmatched_mz = np.empty(n_unmatched, dtype=np.float64)
    for k in range(n_unmatched):
        unmatched_mz[k] = all_mz[unmatched_idx[k]]
    sort_order = np.argsort(unmatched_mz)

    # Assign group IDs: consecutive peaks within tolerance share a group
    unmatched_group = np.empty(n_unmatched, dtype=np.int64)
    if n_unmatched > 0:
        current_group = 0
        unmatched_group[sort_order[0]] = 0
        for j in range(1, n_unmatched):
            prev_mz = unmatched_mz[sort_order[j - 1]]
            cur_mz = unmatched_mz[sort_order[j]]
            tol = mz_ppm * prev_mz
            if abs(cur_mz - prev_mz) <= tol:
                unmatched_group[sort_order[j]] = current_group
            else:
                current_group += 1
                unmatched_group[sort_order[j]] = current_group
        num_unmatched_groups = current_group + 1
    else:
        num_unmatched_groups = 0

    # Map group IDs back to full library-peak array
    unmatched_group_for_all = np.empty(n_lib, dtype=np.int64)
    k = 0
    for i in range(n_lib):
        if not lib_peaks_matched[i]:
            unmatched_group_for_all[i] = unmatched_group[k]
            k += 1
        else:
            unmatched_group_for_all[i] = 0  # unused

    n_rows = len(mz_peaks) + num_unmatched_groups
    n_cols = len(unique_fdcs)

    dense_matrix = np.zeros((n_rows, n_cols), dtype=np.float64)

    ms1_peak_coord = np.empty(n_lib, dtype=np.int64)
    for i in range(n_lib):
        if lib_peaks_matched[i]:
            ms1_peak_coord[i] = matched_obs_idx[i]
        else:
            ms1_peak_coord[i] = last_unassigned_peak_idx + unmatched_group_for_all[i]

    # np.add.at(dense_matrix, (ms1_peak_coord, fdc_idxs), rel_iso_int)
    for i in range(rel_iso_int.size):
        dense_matrix[ms1_peak_coord[i], fdc_idxs[i]] += rel_iso_int[i]

    # Build dia_spec_int using summed intensities: for each observed row that
    # was matched by a library peak, replace its intensity with the sum of all
    # observed peaks within tolerance of that library peak.
    dia_spec_int = dia_spectrum[:, 1].copy()
    for i in range(n_lib):
        if lib_peaks_matched[i]:
            row = matched_obs_idx[i]
            if summed_obs[i] > dia_spec_int[row]:
                dia_spec_int[row] = summed_obs[i]

    n_rows = dense_matrix.shape[0]
    if len(dia_spec_int) > n_rows:
        dia_spec_int = dia_spec_int[:n_rows]
    elif len(dia_spec_int) < n_rows:
        dia_spec_int = np.append(dia_spec_int, [0]*(n_rows - dia_spec_int.shape[0]))

    # dia_spec_int = np.append(dia_spec_int,[0]*(dense_matrix.shape[0]-dia_spec_int.shape[0])) 

    return dense_matrix, dia_spec_int, lib_peaks_matched, fdc_idxs, group_lengths

