"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""



from src.utils.io.read_output import get_large_prec

from sklearn.model_selection import KFold,GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve,auc 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from scipy import stats
import xgboost as xgb

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm
import re
import os
import pandas as pd

# from src.ms1_cor_channels import ms1_cor_channels
import src.ms1_cor_channels as ms1_mod
from src.utils.io.load_files import loadSpectra
from src.models.spec_lib.spec_lib import loadSpecLib

#from .mass_tags import mTRAQ, mTRAQ_02468, mTRAQ_678, tag_library
from src.utils.misc_functions import unstring_floats
from src.utils.frag_encoding import get_index

from src import config
from src.logger import logger
from src.utils.misc_functions import p_result
from src.mass_tags import massTag
from pyteomics import mass
from numba import njit



# Grouping key for the cross-channel scoring features: one plexDIA set, i.e. the
# co-isolated mass-tag channels of a precursor within a single acquisition. Built
# by _plex_set_key.
PLEX_SET_COL = "plex_set"


@njit(nogil=True)
def _compute_shared_frac_jit(target_mz, target_int, other_mz_flat, mz_tol):
    """For one precursor, compute fraction of library intensity shared with other channels.

    target_mz, target_int: 1D arrays for this channel's matched library fragments
    other_mz_flat: concatenated m/z arrays from all OTHER channels of same untag_prec
    mz_tol: relative tolerance (e.g. 20e-6)

    Returns: fraction of target_int that has a match in any other channel
    """
    total_int = 0.0
    shared_int = 0.0
    for i in range(len(target_mz)):
        total_int += target_int[i]
        mz = target_mz[i]
        found = False
        for j in range(len(other_mz_flat)):
            if abs(other_mz_flat[j] - mz) / mz < mz_tol:
                found = True
                break
        if found:
            shared_int += target_int[i]
    if total_int == 0.0:
        return 0.0
    return shared_int / total_int


def _compute_frac_shared_intensity(fdc, mz_tol, group_col="untag_prec"):
    """Compute fraction of library intensity from fragments shared across channels.

    For each ``group_col`` group with 2+ channels, compares fragment m/z lists
    across channels using relative PPM tolerance. Single-channel groups get -1.

    ``group_col`` is the plexDIA set (see :func:`_plex_set_key`): on a timeplex
    run it is scoped to one time channel, so a precursor is only ever compared
    against the channels it was co-isolated with.
    """
    result = np.full(len(fdc), -1.0)
    groups = fdc.groupby(group_col).indices

    for _group_key, indices in groups.items():
        if len(indices) < 2:
            continue

        # Collect frag_mz and frag_int arrays for all rows in this group
        mz_arrays = []
        int_arrays = []
        for idx in indices:
            mz_arr = np.array(fdc["frag_mz"].iloc[idx], dtype=np.float64)
            int_arr = np.array(fdc["frag_int"].iloc[idx], dtype=np.float64)
            mz_arrays.append(mz_arr)
            int_arrays.append(int_arr)

        for k, idx in enumerate(indices):
            # Build other_mz_flat from all channels except this one
            other_parts = [mz_arrays[j] for j in range(len(indices)) if j != k]
            if len(other_parts) == 0 or all(len(p) == 0 for p in other_parts):
                result[idx] = 0.0
                continue
            other_mz_flat = np.concatenate(other_parts)
            target_mz = mz_arrays[k]
            target_int = int_arrays[k]
            if len(target_mz) == 0:
                result[idx] = 0.0
                continue
            result[idx] = _compute_shared_frac_jit(target_mz, target_int, other_mz_flat, mz_tol)

    return result


def _compute_frac_shared_intensity_from_polars(fdc, fdc_list_cols, mz_tol,
                                               group_col="untag_prec"):
    """Variant that reads frag_mz/frag_int from a polars list-cols frame.

    Avoids materializing the list columns as pandas (which inflates 4-5× over
    polars). Per-group iteration converts only the small per-precursor slice
    to numpy arrays — the inner JIT function is unchanged.

    ``group_col`` has the same meaning as in :func:`_compute_frac_shared_intensity`.
    """
    n = len(fdc)
    result = np.full(n, -1.0)

    # Join the group key into the list-cols frame so we can group_by in polars.
    untag_lookup = pl.from_pandas(fdc[["__fdc_idx", group_col]])
    joined = fdc_list_cols.join(untag_lookup, on="__fdc_idx").select(
        "__fdc_idx", group_col, "frag_mz", "frag_int"
    )

    for _, group in joined.group_by(group_col):
        indices = group["__fdc_idx"].to_list()
        if len(indices) < 2:
            continue
        mz_lists = group["frag_mz"].to_list()
        int_lists = group["frag_int"].to_list()
        mz_arrays = [np.asarray(x, dtype=np.float64) if x else np.array([], dtype=np.float64)
                     for x in mz_lists]
        int_arrays = [np.asarray(x, dtype=np.float64) if x else np.array([], dtype=np.float64)
                      for x in int_lists]
        for k, fdc_idx in enumerate(indices):
            other_parts = [mz_arrays[j] for j in range(len(indices)) if j != k]
            if len(other_parts) == 0 or all(len(p) == 0 for p in other_parts):
                result[fdc_idx] = 0.0
                continue
            other_mz_flat = np.concatenate(other_parts)
            target_mz = mz_arrays[k]
            target_int = int_arrays[k]
            if len(target_mz) == 0:
                result[fdc_idx] = 0.0
                continue
            result[fdc_idx] = _compute_shared_frac_jit(
                target_mz, target_int, other_mz_flat, mz_tol
            )
    return result


def _compute_med_frag_error_from_polars(fdc_list_cols):
    """Polars-native med_frag_error computation. Returns a pandas DataFrame
    indexed by ``__fdc_idx`` with a single ``med_frag_error`` column."""
    explode_frame = fdc_list_cols.select(
        pl.col("frag_errors").explode().drop_nulls().alias("e")
    )
    if explode_frame.height > 0:
        global_median = float(explode_frame["e"].median())
    else:
        global_median = 0.0
    return fdc_list_cols.select(
        pl.col("__fdc_idx"),
        pl.col("frag_errors").list.eval(
            (pl.element() - global_median).abs()
        ).list.median().alias("med_frag_error")
    ).to_pandas()


# The fragment ordinal lives in bits [10:3] of the packed int32 fragment code
# (src/utils/frag_encoding.py). For b/y/a/c/x/z ions that ordinal is the number
# of residues in the fragment, so it is the fragment's length in amino acids.
# Spelled as //8 %256 rather than >>3 &0xFF so the same arithmetic works inside
# a polars expression.
_FRAG_IDX_DIV = 1 << 3    # == 1 << _IDX_SHIFT
_FRAG_IDX_MOD = 0xFF + 1  # == _IDX_MASK + 1


def _compute_wtd_frag_len_from_polars(fdc_list_cols):
    """Intensity-weighted mean fragment length per PSM, computed in polars.

    ``sum((b / x) * a)`` over a precursor's matched fragments, where ``a`` is
    the fragment's length in residues, ``b`` its observed MS2 intensity, and
    ``x`` the precursor's total matched MS2 intensity. Because the weights sum
    to one this is the mean fragment length weighted by observed signal.

    Stays list-native (no explode) to keep peak memory down. Returns a pandas
    frame keyed by ``__fdc_idx``.
    """
    frag_len = pl.col("frag_names").list.eval(
        (pl.element() // _FRAG_IDX_DIV) % _FRAG_IDX_MOD
    ).cast(pl.List(pl.Float64))
    # obs_int is a measured intensity so it is never negative, but a fragment
    # can come back null; those contribute no signal.
    obs = pl.col("obs_int").list.eval(pl.element().fill_null(0.0))
    total = obs.list.sum()
    return fdc_list_cols.select(
        pl.col("__fdc_idx"),
        pl.when(total > 0)
          .then((frag_len * obs).list.sum() / total)
          .otherwise(None)
          .alias("wtd_frag_len"),
    ).to_pandas()


def _compute_wtd_frag_len(frag_names, obs_int):
    """Pandas-path equivalent of :func:`_compute_wtd_frag_len_from_polars`.

    NaN where a PSM has no matched fragments or no observed signal.
    """
    frag_names = list(frag_names)
    obs_int = list(obs_int)
    out = np.full(len(frag_names), np.nan)
    for i, (codes, ints) in enumerate(zip(frag_names, obs_int)):
        if codes is None or ints is None or len(codes) == 0 or len(codes) != len(ints):
            continue
        b = np.nan_to_num(np.asarray(ints, dtype=float), nan=0.0)
        a = get_index(np.asarray(codes, dtype=np.int32)).astype(float)
        x = b.sum()
        if x > 0:
            out[i] = float((b * a).sum() / x)
    return out


def _central(vals, n):
    """The central 2n+1 values of a ";"-joined trace (all of it when n is None)."""
    v = np.array(list(map(float, vals.split(";")))) if isinstance(vals, str) else np.asarray(vals, float)
    if n is None or v.size <= 2 * n + 1:
        return v
    c = v.size // 2
    return v[c - n:c + n + 1]


def area(x, apex_idx):
    top_3 = x[max(0, apex_idx-1):apex_idx+2]
    return np.sum(top_3)


def argmax_within_radius(fitted, start_pos, radius=None):
    """Pick the index with the highest ``fitted`` value within
    ``start_pos ± radius`` cycles, capped one position in from either edge so
    ``area`` always has both neighbors for the top-3 sum. When ``radius`` is
    ``None`` the cap is the full interior of ``fitted``.
    """
    n = len(fitted)
    if n < 3:
        return start_pos
    lo = 1
    hi = n - 2
    if radius is not None:
        lo = max(lo, start_pos - radius)
        hi = min(hi, start_pos + radius)
    return lo + int(np.argmax(fitted[lo:hi + 1]))


def walk_to_local_max(fitted, start_pos, apex_jitter=None):
    """Slide the apex from ``start_pos`` toward a local maximum, one cycle at a
    time, while strictly increasing. Capped one position in from either edge so
    the picked apex always has both neighbors available for ``area`` to sum the
    top-3 window. When ``apex_jitter`` is given, the walk additionally stops
    once it has moved ``apex_jitter`` cycles from ``start_pos`` in either
    direction.

    Returns the new apex index. Identical to ``start_pos`` when the voted apex
    is already a local maximum.
    """
    n = len(fitted)
    if n < 3:
        return start_pos
    lo_cap = 1
    hi_cap = n - 2
    if apex_jitter is not None:
        lo_cap = max(lo_cap, start_pos - apex_jitter)
        hi_cap = min(hi_cap, start_pos + apex_jitter)
    i = start_pos
    while True:
        left_ok = (i - 1 >= lo_cap)
        right_ok = (i + 1 <= hi_cap)
        left_higher = left_ok and (fitted[i - 1] > fitted[i])
        right_higher = right_ok and (fitted[i + 1] > fitted[i])
        if not (left_higher or right_higher):
            return i
        if left_higher and right_higher:
            # Voted apex is in a saddle — step toward the higher side.
            i = i - 1 if fitted[i - 1] >= fitted[i + 1] else i + 1
        elif left_higher:
            i -= 1
        else:
            i += 1



# lp,fdc,dc = get_large_prec(file,condense_output=False,timeplex=bool(params["timeplex"]))

# all_lp.append(lp)
# all_prec_labels.append(lp)
                       
       

### check if this processing was already done
### If so load it
### if not create it

# ID_attributes_file  = "precursor_attributes.csv"
# ID_attributes_path = results_folder+"/"+ID_attributes_file 
# if os.path.exists(ID_attributes_path):
#     fdc = pd.read_csv(ID_attributes_path)
# else:

    
# def add_attributes(fdc):
     
#     ## Add additional features
#     # X["prec_z"] = fdc["z"]
#     fdc["pep_len"] = [len(re.findall("([A-Z](?:\(.*?\))?)",i.split("_")[-1])) for i in fdc["seq"]]
#     fdc["stripped_seq"] = np.array([re.sub("\(.*?\)","",i) for i in fdc["seq"]])
#     # X["rt"] = fdc["rt"]
#     # X["coeff"] = fdc["coeff"]
#     fdc["sq_rt_error"] = np.power(fdc["rt_error"],2)
#     fdc["sq_mz_error"] = np.power(fdc["mz_error"],2)
    
    
#     return fdc


def ms1_quant(dat,lp,dc,mass_tag,SILAC,DIAspectra,mz_ppm,rt_tol,timeplex=False,vote_sigma=1.0):
    # X = fdc.iloc[:,6:-5]
    fit_whole_MS1 = False
   
    logger.info("")
    logger.info("Performing MS1 Quantitation") 
    
    fdc = dat[~dat["is_decoy"]].copy().reset_index(drop=True)  #remove decoys
    
    #only quantify confident precs
    if config.args.unfiltered_quant: #this will not execute if you specificy --unfiltered_quant (inherently stored as false)
        fdc = fdc[fdc["BestChannel_Qvalue"] < 0.01].reset_index(drop=True)

    if timeplex:
        all_keys = [(i,j,k) for i,j,k in zip(fdc.seq,fdc.z,fdc.time_channel)]
    else:
        all_keys = [(i,j) for i,j in zip(fdc.seq,fdc.z)]

    if mass_tag:
        tag_to_use = mass_tag
    else:
        tag_to_use = massTag(rules="nK",
                            base_mass=0.00,
                            delta=[0],
                            channel_names=["0"],
                            name="no_tag",
                            compositions=mass.Composition())

    (
    group_p_corrs,
    group_ms1_traces,
    group_ms2_traces,
    group_iso_ratios,
    group_keys,
    group_fitted,
    new_output_dict,
    fake_fdc_dict
    ) = ms1_mod.ms1_cor_channels(DIAspectra, 
                                fdc, 
                                dc, 
                                mz_ppm=mz_ppm, 
                                rt_tol = rt_tol,
                                tag=tag_to_use,
                                SILAC=SILAC,
                                timeplex=timeplex,
                                num_iso = config.num_iso_ms1,
                                num_iso_r = config.num_iso_r,
                                additional_scans = config.args.additional_scans,
                                vote_sigma = vote_sigma,
                                fit_whole_MS1=fit_whole_MS1
                                )
    
    dat = process_ms1_quant(dat,
                            fdc,
                            all_keys,
                            group_p_corrs,
                            group_ms1_traces,
                            group_ms2_traces,
                            group_iso_ratios,
                            group_keys,
                            group_fitted,
                            new_output_dict,
                            fake_fdc_dict,
                            DIAspectra,
                            fit_whole_MS1=fit_whole_MS1
                            )

    return dat



def process_ms1_quant(dat, fdc, all_keys, group_p_corrs, group_ms1_traces, group_ms2_traces, group_iso_ratios, group_keys, group_fitted, new_output_dict, fake_fdc_dict, DIAspectra, fit_whole_MS1=False):
    
    ## create dictionary  that links keys to data so we can match the order of "fdc"
    linker_dict = {key:[group_idx,key_idx] for group_idx,keys in enumerate(group_keys) for key_idx,key in enumerate(keys)}
    
    p_corrs = [group_p_corrs[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
    ms1_traces = [group_ms1_traces[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
    ms2_traces = [group_ms2_traces[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
    iso_ratios = [group_iso_ratios[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
    extracted_keys = [group_keys[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]

    if fit_whole_MS1 is False:
        extracted_fitted = [group_fitted[linker_dict[key][0]][0][:,linker_dict[key][1]] for key in all_keys]
        extracted_fitted_specs = [group_fitted[linker_dict[key][0]][4] for key in all_keys]
        extracted_fitted_p = [group_fitted[linker_dict[key][0]][3] for key in all_keys]

    else:
        ##TODO deal with fake_fdc_dict values make sure they get properly assigned
        all_fdc_idxs = list(set(key for ms1_spec_idx in new_output_dict.keys() for key in new_output_dict[ms1_spec_idx]["mapped_pred_coeffs"].keys()))
        extracted_fitted_specs = [[] for k in all_fdc_idxs if k not in fake_fdc_dict.keys()]
        extracted_fitted_p = [[] for k in all_fdc_idxs if k not in fake_fdc_dict.keys()]
        for ms1_spec_idx in new_output_dict.keys():
            for key in new_output_dict[ms1_spec_idx]["mapped_pred_coeffs"].keys():
                if key not in fake_fdc_dict.keys():
                    extracted_fitted_specs[key].append(ms1_spec_idx)
                    extracted_fitted_p[key].append(p_result(r_sq=np.nan, p=np.nan))

        extracted_fitted = []
        for fdc_idx in all_fdc_idxs:
            if fdc_idx not in fake_fdc_dict.keys():
                fdc_coeffs = []
                for scan in extracted_fitted_specs[fdc_idx]:
                    fdc_coeffs.append(new_output_dict[scan]["mapped_pred_coeffs"][fdc_idx])
                
                extracted_fitted.append(np.array(fdc_coeffs))
    
    
    fdc["plexfitMS1"] = [np.max(i) for i in extracted_fitted]
    fdc["plexfitMS1_p"] = [j[np.argmax(i)].statistic  if type(j[np.argmax(i)])!=float else 0 for i,j in zip(extracted_fitted,extracted_fitted_p)]

    plexfittrace_idxs = [np.where([e in set(k) for e in j])[0] for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
    plexfittrace = [i[j] for i,j in zip(extracted_fitted,plexfittrace_idxs)]
    plexfit_ps = [[i[k].statistic if type(i[k])!=float else 0 for k in j] for i,j in zip(extracted_fitted_p,plexfittrace_idxs)]
    # fdc["plexfitMS1_new"] = [np.max(i) for i in plexfittrace]
    fdc["plexfittrace"] = [";".join(map(str,i)) for i in plexfittrace] ###spec ids come from ms2_traces
    fdc["plexfit_ps"] = [";".join(map(str,i)) for i in plexfit_ps]
    
    fdc["plexfittrace_spec_all"] = [";".join(map(str,j)) for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
    fdc["plexfittrace_all"] = [";".join(map(str,i)) for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
    fdc["plexfittrace_ps_all"] = [";".join(map(str,[pi.statistic if pi==pi else np.nan for pi in p])) for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
    # ms1_cor_channels fit a ±fit_radius window around the group's voted apex.
    # The voted apex is at the center of fitted/specs. Each channel can drift
    # away from that center monotonically (only while values are strictly
    # increasing), capped one position in from the edge so area() always has
    # both neighbors for its top-3 sum.
    plex_areas = []
    apex_scans = []
    for idx in range(len(fdc)):
        fitted = extracted_fitted[idx]
        specs = extracted_fitted_specs[idx]
        center = len(fitted) // 2
        # Per-channel apex selection: optionally argmax-pick a seed within
        # ±additional_scans of the group-voted center, then let the monotonic
        # walk drift from that seed within ±apex_jitter. The two knobs are
        # independent — apex_jitter still matters when free_apex is on, and
        # vice versa.
        if config.args.free_apex:
            seed = argmax_within_radius(fitted, center, radius=int(config.args.additional_scans))
        else:
            seed = center
        apex_pos = walk_to_local_max(fitted, seed, apex_jitter=int(config.args.apex_jitter))
        plex_areas.append(area(fitted, apex_pos))
        apex_scans.append(int(specs[apex_pos]))
    fdc["plex_Area"] = plex_areas
    fdc["ms1_apex_scan"] = apex_scans

       


    
    
    fdc["ms1_cor"] = [i[0] for i in p_corrs]
    
    for idx in range(config.num_iso_r):
        iso_num = idx+1
        fdc[f"iso{iso_num}_cor"] = [i[iso_num] for i in p_corrs]
    # fdc["iso1_cor"] = [i[1] for i in p_corrs]
    # fdc["iso2_cor"] = [i[2] for i in p_corrs]
    
    # Rescale correlations from [-1,1] to [0,1] before computing product
    _ms1_cor_scaled = (fdc["ms1_cor"] + 1) / 2
    _iso1_cor_scaled = (fdc["iso1_cor"] + 1) / 2
    _iso2_cor_scaled = (fdc["iso2_cor"] + 1) / 2
    fdc["traceproduct"] = np.log10(_ms1_cor_scaled * _iso1_cor_scaled * _iso2_cor_scaled + 1e-6)
    
    # fdc["MS1_is1cor"] = [stats.pearsonr(list(i[0].values())[:10], list(i[1].values())[:10]).statistic for i in ms1_traces]
    
    
    fdc["iso_cor"] = [i[0].statistic for i in iso_ratios]
    
    fdc["MS1_Int"] = [i[2][0] for i in iso_ratios]
    fdc["MS1_Int"] = [np.linalg.lstsq(np.array(i[1])[:,np.newaxis], i[2], rcond=-1)[0][0] for i in iso_ratios]
    
    # X[np.isnan(X)]=0 ## set nans to zero (mostly for r2 values)
    fdc["all_ms1_specs"] = [";".join(map(str,trace[0].keys())) for trace in ms1_traces]
    for i in range(config.num_iso_ms1):
        fdc[f"all_ms1_iso{i}vals"] = [";".join(map(str,trace[i].values())) for trace in ms1_traces]
    # fdc["ms2_trace"] = [";".join(map(str,trace.values())) for trace in ms2_traces]
    
    # Integrate over scan *index*, not spectrum id. Scans are evenly spaced in
    # time, but the spectrum-id gap between consecutive MS1 scans is the number
    # of MS2 windows in the cycle, which is not constant: measured 8-12 on
    # JD0311_re. Using it as the x axis multiplied each precursor's area by its
    # own local cycle length, injecting pure artifact spread between precursors
    # that has nothing to do with abundance (robust SD 1.203 -> 1.181).
    # Integrate the whole extracted MS1 trace. _central also PARSES the
    # ";"-joined trace strings into arrays, so it is still needed with n=None.
    fdc["MS1_Area"] = [
        float(np.trapz(_central(vals, None)))
        for vals in fdc.all_ms1_iso0vals
    ]


        # Define selected columns that we want to merge
    selected_cols = [
        "plexfitMS1", "plexfitMS1_p", "plexfittrace", "plexfit_ps",
        "plexfittrace_spec_all", "plexfittrace_all", "plexfittrace_ps_all",
        "plex_Area", "ms1_apex_scan", "ms1_cor", "traceproduct", "iso_cor", "MS1_Int",
        "all_ms1_specs", "MS1_Area",
    ] + [f"all_ms1_iso{i}vals" for i in range(config.num_iso_ms1)]
    
    # Ensure we only select columns that actually exist in fdc
    existing_cols = [col for col in selected_cols if col in fdc.columns]
    
    # Perform the merge safely
    dat = dat.merge(fdc[["untag_prec", "channel","silac_channel"] + existing_cols], how="left", on=["untag_prec", "channel","silac_channel"]).fillna(0)


    return dat



def estimate_pep(scores, is_decoy):
    """Estimate Posterior Error Probability via isotonic regression.

    Fits a non-decreasing decoy probability curve over the score distribution.
    PEP = decoy_prob / (1 - decoy_prob), clamped to [0, 1].
    """
    order = np.argsort(-scores)  # descending
    labels = is_decoy[order].astype(float)  # decoy=1, target=0

    ir = IsotonicRegression(y_min=0, y_max=1, increasing=True)
    fitted = ir.fit_transform(np.arange(len(labels)), labels)

    pep = fitted / (1 - fitted + 1e-10)
    pep = np.clip(pep, 0, 1)

    # Map back to original order
    result = np.empty_like(pep)
    result[order] = pep
    return result


class model_instance():
    def __init__(self,model_type):
        self.mode_type = model_type
        
    def predict(self,X):
        pred = self.__predict_fn__(X)
        #First column of pred is target probabilities and second column is decoys
        #If for some reason there were no decoys in one of the training folds
        #only a single column is returned. Handle this case...
        if len(pred.shape)==2:
            if pred.shape[1]==2:
                output = pred[:,1]
            else:
                output = pred[:,0]
        else:
            output = pred
        return output
        
        
class score_model():
    
    def __init__(self,model_type,n_splits=5,folder=None):
        self.model_type=model_type
        self.n_splits = n_splits
        self.folder = folder
                
    def run_model(self,X,y,sample_weight=None,groups=None):
        # logger.info(f"{config.tree_max_depth}")
        if self.model_type=="rf":
            
            ### Random Forest
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = RandomForestClassifier(n_estimators = 200,max_depth=config.tree_max_depth,n_jobs=-1, random_state=config.RANDOM_SEED)
                    m.model.fit(X,y,sample_weight=sample_weight)
                    m.__predict_fn__ = m.model.predict_proba

                    if self.folder:
                        feature_importance = m.model.feature_importances_
                        sorted_indices = np.argsort(feature_importance, kind='stable')
                        sorted_features = np.array(X.columns)[sorted_indices]  
                        sorted_importance = feature_importance[sorted_indices]  
                    
                        fig, ax = plt.subplots(figsize=(8, len(X.columns)*0.3))                    
                        ax.barh(sorted_features, sorted_importance)
                        ax.set_title("Feature Importance")
                    
                        # Save plot
                        plt.savefig(self.folder + f"/scoring/RF{idx}_feature_importance.png", dpi=600, bbox_inches="tight")
                        # For RF models, log feature importance
                        plt.close(fig)
                    return m
                
            # self.model = fit_model(X,y)
            
        
        elif self.model_type=="lda":
            
            ## Linear Disriminant Analysis
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = LinearDiscriminantAnalysis()
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
                
            # self.model = fit_model(X,y)
            
            
        elif self.model_type == "xg":
            
            ## XGBoost
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    dTrain = xgb.DMatrix(X,y,weight=sample_weight,feature_names=list(X.columns))
                    # param = {
                    #     'max_depth': config.tree_max_depth, 
                    #     'eta': .1, 
                    #     'objective': 'binary:logistic',}
                    param = {
                        # 'max_depth': config.tree_max_depth, 
                        # 'eta': .1, 
                        # 'objective': 'binary:logistic',
                        
                        'objective': 'binary:logistic',  
                         'eval_metric': 'aucpr',    
                         'eta': 0.1,
                         'max_depth': 10,          
                         'subsample': 0.8,
                         'colsample_bytree': 0.8,   
                         'tree_method': 'hist',         
                         'nthread': -1,                   
                         'seed': config.RANDOM_SEED,
                         'min_child_weight': .5
                        }

                    # param['nthread'] = 4
                    # param['eval_metric'] = 'pre'
                    
                    m.model = xgb.train(param, dtrain=dTrain,num_boost_round=500)
                    def xg_predict(X):
                        X_convert = xgb.DMatrix(X,feature_names=list(X.columns))
                        return m.model.predict(X_convert)
                    m.__predict_fn__ = xg_predict
                    
                    if self.folder:
                        os.makedirs(os.path.join(self.folder, "scoring"), exist_ok=True)
                        fi = m.model.get_score(importance_type="gain")
                        feature_importance = np.array([fi.get(c, 0) for c in X.columns])
                        sorted_indices = np.argsort(feature_importance, kind='stable')
                        sorted_features = np.array(X.columns)[sorted_indices]
                        sorted_importance = feature_importance[sorted_indices]

                        fig, ax = plt.subplots(figsize=(8, len(X.columns)*0.3))
                        ax.barh(sorted_features, sorted_importance)
                        ax.set_title("Feature Importance")
                        plt.savefig(self.folder+f"/scoring/XGBoost{idx}_feature_importance.png",dpi=600,bbox_inches="tight")
                        plt.close(fig)

                    return m
                
            # self.model = fit_model(X,y)
        
            
        elif self.model_type == "nn":
            columns = X.columns
            X = pd.DataFrame(preprocessing.StandardScaler().fit(X).transform(X),columns=columns)
            ## Neural network
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    # m.model = MLPClassifier((32,16,8,4),activation="relu")
                    m.model = MLPClassifier((20,20,4),activation="relu")
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
                
        else:
            from src.utils.gui_utils import send_raise_to_TK
            send_raise_to_TK("ValueError - Unsupported Model Type")
            raise ValueError("Unsupported model type")
        
        logger.debug(f"Total samples: {len(y)}, Positive: {sum(y)}, Negative: {len(y) - sum(y)}")
        
        kf = KFold(n_splits=self.n_splits,shuffle=True, random_state = config.RANDOM_SEED)
        k_orders = [i for i in kf.split(X,y)]
        rev_order = np.argsort(np.concatenate([i[1] for i in k_orders], ), kind='stable') # collapse test sets and get order

        if groups is not None:
            unique_groups = np.unique(groups)
            if len(unique_groups) < 5:
                logger.warning(f"Warning: Only {len(unique_groups)} unique groups for 5-fold CV. Using KFold instead.")
                gfk = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
            else:
                gfk = GroupKFold(n_splits=5)
        
            #k_orders = [i for i in kf.split(X,y)] old way
            k_orders = [i for i in gfk.split(X, y, groups=groups)]
            rev_order = np.argsort(np.concatenate([i[1] for i in k_orders]), kind='stable') # collapse test sets and get order
            
            # permutation = np.random.permutation(len(X))
            # X_shuffled = X.iloc[permutation]
            # y_shuffled = y[permutation]
            # groups_shuffled = np.array(self.groups)[permutation]
            # k_orders = [i for i in gfk.split(X_shuffled,y_shuffled,groups=groups_shuffled)]
            # rev_order = np.argsort(np.concatenate([i[1] for i in k_orders])) # collapse test sets and get order

        self.models = []
        self.predictions=[]
        for model_idx, (train_idx, test_idx) in enumerate(tqdm.tqdm(k_orders)):
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            weights = sample_weight[train_idx] if sample_weight is not None else None
            m = fit_model(X_train, y_train, sample_weight=weights, idx=model_idx)
            self.models.append(m)
            self.predictions.append(m.predict(X.iloc[test_idx]))

        return np.concatenate(self.predictions)[rev_order]

    def run_model_filtered(self, X, y, keep_mask, y_filtered, sample_weight=None, groups=None):
        """Train with per-fold filtering: CV folds on full data, filter/relabel training portion only.

        For each CV fold:
          - Take the training indices from the full dataset
          - Apply keep_mask to retain only selected training samples
          - Use y_filtered (with negative-mined labels) for training
          - Predict on the FULL test split (unfiltered, unbiased)
        """
        if self.model_type=="rf":
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = RandomForestClassifier(n_estimators = 200,max_depth=config.tree_max_depth,n_jobs=-1, random_state=config.RANDOM_SEED)
                    m.model.fit(X,y,sample_weight=sample_weight)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
        elif self.model_type=="lda":
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = LinearDiscriminantAnalysis()
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
        elif self.model_type == "xg":
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    dTrain = xgb.DMatrix(X,y,weight=sample_weight,feature_names=list(X.columns))
                    param = {
                        'objective': 'binary:logistic',
                         'eval_metric': 'aucpr',
                         'eta': 0.1,
                         'max_depth': 10,
                         'subsample': 0.8,
                         'colsample_bytree': 0.8,
                         'tree_method': 'hist',
                         'nthread': -1,
                         'seed': config.RANDOM_SEED,
                         'min_child_weight': .5
                        }
                    m.model = xgb.train(param, dtrain=dTrain,num_boost_round=500)
                    def xg_predict(X):
                        X_convert = xgb.DMatrix(X,feature_names=list(X.columns))
                        return m.model.predict(X_convert)
                    m.__predict_fn__ = xg_predict
                    return m
        elif self.model_type == "nn":
            columns = X.columns
            X = pd.DataFrame(preprocessing.StandardScaler().fit(X).transform(X),columns=columns)
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = MLPClassifier((20,20,4),activation="relu")
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
        else:
            raise ValueError("Unsupported model type")

        logger.debug(f"Total samples: {len(y)}, Kept for training: {keep_mask.sum()}")

        # CV folds on the FULL dataset (same as run_model)
        kf = KFold(n_splits=self.n_splits,shuffle=True, random_state = config.RANDOM_SEED)
        k_orders = [i for i in kf.split(X,y)]

        if groups is not None:
            unique_groups = np.unique(groups)
            if len(unique_groups) < 5:
                gfk = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
            else:
                gfk = GroupKFold(n_splits=5)
            k_orders = [i for i in gfk.split(X, y, groups=groups)]

        rev_order = np.argsort(np.concatenate([i[1] for i in k_orders]), kind='stable')

        self.models = []
        self.predictions = []
        model_idx = 0
        for train_idx, test_idx in tqdm.tqdm(k_orders):
            # Filter training portion: keep only samples in keep_mask, use relabeled y
            fold_keep = keep_mask[train_idx]
            filtered_train_idx = train_idx[fold_keep]
            X_train_fold = X.iloc[filtered_train_idx]
            y_train_fold = y_filtered[filtered_train_idx]
            sw = sample_weight[filtered_train_idx] if sample_weight is not None else None

            m = fit_model(X_train_fold, y_train_fold, sample_weight=sw, idx=model_idx)
            self.models.append(m)
            # Predict on FULL test split (unfiltered)
            self.predictions.append(m.predict(X.iloc[test_idx]))
            model_idx += 1

        return np.concatenate(self.predictions)[rev_order]


def score_precursors(fdc,model_type="rf",fdr_t=0.01, folder=None):
    """
    Parameters
    ----------
    fdc : pandas.DataFrame
        All PSMs identified.
    model_type : string [autogluon]
                 Type of ML model used to discriminate targets and decoys.
                 Only 'autogluon' is supported in this implementation.
    fdr_t : float
        False discovery rate threshold.
    folder : str, optional
        Folder path for saving plots.

    Returns
    -------
    fdc : pandas.DataFrame
        Updated dataframe with prediction values and Q-values.
    """

    assert model_type in ["lda", "rf", "xg"], 'model_type must be one of ["lda", "rf", "xg"]'

    logger.info("Scoring IDs")


    ## Only decoys are negatives - all targets are positives
    y = np.array(~fdc["is_decoy"], dtype=int)

    # exclude necessary columns
    drop_colums = ['spec_id', 'Ms1_spec_id', 'seq', 'window_mz', 'frag_names', 'frag_errors', 'frag_mz', 'frag_int', 'obs_int', 'stripped_seq',
                  'untag_seq', 'is_decoy', 'all_ms1_specs', 'all_ms1_iso0vals', 'all_ms1_iso1vals', 'all_ms1_iso2vals','all_ms1_iso3vals', 'all_ms1_iso4vals',
                  'all_ms1_iso5vals','all_ms1_iso6vals','all_ms1_iso7vals',"plexfittrace","plexfit_ps","untag_prec","plexfittrace_spec_all","plexfittrace_all",
                  "plexfittrace_ps_all",
                  "unique_frag_mz", "untag_prec",
                  "channels_matched", PLEX_SET_COL,
                  "unique_obs_int", 'MS1_Int',"MS1_Area", "iso_cor", "cosine", "traceproduct","iso1_cor","iso2_cor","ms1_cor","plexfitMS1","plexfitMS1_p","plex_Area", "untag_prec","channel","time_channel",
                  "silac_channel",
                  "unique_frag_mz",
                  "unique_obs_int",
                  "file_name",
                  "protein",
                  # Parquet row index, carried through so the held-back list
                  # columns can be merged back later.  It must never be a
                  # feature: rows are interleaved but not uniformly, and on
                  # JD0413 the decoy fraction runs monotonically from 0.512 in
                  # the first decile of row order to 0.353 in the last, which a
                  # tree can split on.
                  "__fdc_idx",
                  # MS2 areas are reported quantities, not evidence of
                  # correctness -- dropped for the same reason plex_Area,
                  # MS1_Area and MS1_Int are, and because they measured
                  # neutral-to-worse as features.
                  "coeff_Area", "coeff_ChannelUnique_Area",
                  # single-scan value at the apex the whole plex group shares.
                  # Reported so it can be compared against the per-channel-scan
                  # value; an abundance, so never a scoring feature.
                  "coeff_CommonApex", "coeff_ChannelUnique_CommonApex",
                  # fragment-purity diagnostic: reported so it can be compared
                  # against coeff_ChannelUnique, never scored -- it is the same
                  # abundance measured two ways, so scoring on it would leak
                  # the quantity into the identification it is meant to test
                  "coeff_CU_cleanFrags", "coeff_CU_dirtyFrags",
                  "coeff_CU_cleanFrags_a0", "coeff_CU_dirtyFrags_a0",
                  "cu_purity_median_s",
                  "coeff_CU_pw1", "coeff_CU_pw2", "coeff_CU_pw4",
                  "coeff_CU_pw8", "coeff_CU_pw16",
                  "coeff_CU_pwInv", "coeff_CU_pwInv2"]
    X = fdc.drop([c for c in drop_colums if c in fdc.columns], axis=1)

    # Compute predicted RT (library RT) from observed RT minus RT error
    if 'rt' in X.columns and 'rt_error' in X.columns:
        X['predicted_rt'] = X['rt'] - X['rt_error']
    X = X.drop(columns=[c for c in ['rt'] if c in X.columns])

    # Quantile-bin positional/intensity features into 100 bins
    QBIN_FEATURES = ['predicted_rt', 'mz', 'coeff', 'tic']
    for col in QBIN_FEATURES:
        if col in X.columns:
            vals = X[col].values.astype(float)
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                edges = np.quantile(valid, np.linspace(0, 1, 101))
                binned = np.searchsorted(edges, vals, side='right').astype(float)
                binned = np.clip(binned, 1, 100)
                X[col + '_qbin'] = binned
            X = X.drop(columns=[col])

    X[np.isnan(X)]=0 ## set nans to zero (mostly for r2 values)

    # Iterative training: 3 iterations with target filtering and negative mining
    n_iterations = 3
    pred = None
    fdc_qvalues = None

    for itr in range(n_iterations):
        logger.info(f"  scoring iteration {itr + 1}/{n_iterations}")

        if itr == 0:
            sc_model = score_model(model_type, folder=folder)
            pred = sc_model.run_model(X, y, groups=fdc.stripped_seq)
        else:
            pep_values = estimate_pep(pred, fdc["is_decoy"].values)

            y_filtered = y.copy()
            neg_mine_mask = (y_filtered == 1) & (pep_values >= 0.90)
            y_filtered[neg_mine_mask] = 0

            keep_mask = fdc["is_decoy"].values | (fdc_qvalues <= 0.01) | neg_mine_mask

            n_kept = keep_mask.sum()
            n_relabeled = neg_mine_mask[keep_mask].sum()
            logger.info(f"    Kept {n_kept}/{len(keep_mask)} samples, relabeled {n_relabeled} as decoys")

            sc_model = score_model(model_type, folder=folder)
            pred = sc_model.run_model_filtered(X, y, keep_mask, y_filtered, groups=fdc.stripped_seq)

        # Compute q-values for this iteration
        score_order = np.argsort(-pred, kind='stable')
        decoy_order = fdc["is_decoy"].values[score_order]
        fdc_qvalues_ordered = (1 + np.cumsum(decoy_order)) / np.cumsum(~decoy_order) * config.target_decoy_ratio
        fdc_qvalues_ordered = np.minimum.accumulate(fdc_qvalues_ordered[::-1])[::-1]
        fdc_qvalues = np.empty_like(fdc_qvalues_ordered)
        fdc_qvalues[score_order] = fdc_qvalues_ordered

    model_name = model_type

    if len(pred.shape)==2:
        output = pred[:,1]
    else:
        output = pred

    ## Use the scores to estimate the #IDs as 1% FDR
    fpr, tpr, _ = roc_curve(y, output)

    score_order = np.argsort(-output, kind='stable')
    orig_order = np.argsort(score_order, kind='stable')
    decoy_order = fdc["is_decoy"][score_order]
    frac_decoy = (1 + np.cumsum(decoy_order)) / np.cumsum(~decoy_order) * config.target_decoy_ratio
    frac_decoy = np.minimum.accumulate(frac_decoy[::-1])[::-1]  # Monotonize: q-value = min of downstream q-values
    T = output[score_order[min(len(score_order)-1,np.searchsorted(frac_decoy,0.01))]]
    above_t = output>T
    fdc["PredVal"] = output
    fdc["Qvalue"] = frac_decoy[orig_order]

    if folder:

        plt.subplots()
        y_log=False
        vals,bins,_ = plt.hist(output,50,log=y_log,label="All")
        plt.hist(output[y.astype(bool)],bins,alpha=.5,log=y_log,label="Targets")
        plt.hist(output[~y.astype(bool)],bins,alpha=.5,log=y_log,label="Decoys")
        plt.legend()
        plt.title(model_name+ f" - Type {config.args.unmatched}")
        plt.vlines(T,0,max(vals))
        plt.savefig(folder+"/scoring/ModelScore.png",dpi=600,bbox_inches="tight")
        plt.close()



        feat = 'rt_error'
        func = np.array#np.log10#
        plt.subplots()
        vals,bins,_ = plt.hist(func([i for i in fdc[feat]]),40,label="All")
        # plt.hist([],[])
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][above_t]]),bins,alpha=.5,label="1%FDR")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][np.logical_and(~above_t,~fdc.is_decoy)]]),bins,alpha=.5,label="Low scoring")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][fdc.is_decoy]]),bins,alpha=.5,label="Decoy")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        # plt.title(model_name+ f" - Type {config.args.unmatched}")
        plt.title(model_name)
        plt.legend()
        plt.savefig(folder+"/scoring/RT_error.png",dpi=600,bbox_inches="tight")
        plt.close()


        feat = 'mz_error'
        func = np.array#np.log10#
        plt.subplots()
        vals,bins,_ = plt.hist(func([i for i in fdc[feat]]),40,label="All")
        # plt.hist([],[])
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][above_t]]),bins,alpha=.5,label="1%FDR")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][np.logical_and(~above_t,~fdc.is_decoy)]]),bins,alpha=.5,label="Low scoring")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][fdc.is_decoy]]),bins,alpha=.5,label="Decoy")

        # putting a xlim so that you can see entire distribution of the mz errors better
        xmin, xmax = plt.xlim()
        max_abs = max(abs(xmin), abs(xmax))
        plt.xlim(-max_abs, max_abs)

        plt.xlabel(feat)
        plt.ylabel("Frequency")
        # plt.title(model_name+ f" - Type {config.args.unmatched}")
        plt.title(model_name)
        plt.legend()
        plt.savefig(folder+"/scoring/mz_error.png",dpi=600,bbox_inches="tight")

        plt.close()
        plt.close("all")

    return fdc


def log_df(df):
    filenames = df.iloc[:, 0].unique()
    filenames = [fn.rstrip("0123456789") for fn in filenames]
    filenames = list(set(filenames))
    for fn in filenames:
        logger.info(f"File: {fn}")

    df_no_first = df.iloc[:, 1:]
    for line in df_no_first.to_string(index=False).splitlines():
        logger.info(line)

def compute_protein_FDR(df,results_folder=None):
    logger.info("")
    logger.info("Computing Protein FDR")

  
    df["run_chan"] = df["file_name"].astype(str) + df["channel"].astype(str)+ df["silac_channel"].astype(str)
    df_seqchargeqvals = df[df["Qvalue"] < 0.01].copy().reset_index(drop=True) #filter
    df_seqchargeqvals["maxPredval"] = df_seqchargeqvals.groupby(["protein", "is_decoy"])["PredVal"].transform("max")
    df_seqchargeqvals = df_seqchargeqvals.drop_duplicates(subset=["protein", "is_decoy"]).reset_index(drop=True)

    # Rank by descending maxPredval and compute accum_decoys & Protein_Qvalue
    df_seqchargeqvals = df_seqchargeqvals.sort_values(by="maxPredval", ascending=False).reset_index(drop=True)
    df_seqchargeqvals["prot_rank"] = df_seqchargeqvals.index + 1  # Equivalent to row_number()
    df_seqchargeqvals["accum_decoys"] = df_seqchargeqvals["is_decoy"].cumsum()
    df_seqchargeqvals["Protein_Qvalue"] = (1 + df_seqchargeqvals["accum_decoys"]) / (~df_seqchargeqvals["is_decoy"]).cumsum() * config.target_decoy_ratio
    df_seqchargeqvals["Protein_Qvalue"] = df_seqchargeqvals["Protein_Qvalue"].iloc[::-1].cummin().iloc[::-1]  # Monotonize: q-value = min of downstream q-values
    
    # Filter for non-decoy proteins and select distinct protein values
    df_seqchargeqvals_distinct = (
        df_seqchargeqvals[~df_seqchargeqvals["is_decoy"]]
        .drop_duplicates(subset=["protein"])
        [["protein", "Protein_Qvalue"]]
    )
    
    df = df.drop(columns=["Protein_Qvalue"], errors="ignore")
    df = df.merge(df_seqchargeqvals_distinct, on="protein", how="left")
        
    df_counts_prec = (
        df[(df["is_decoy"] == False) & (df["Qvalue"] < 0.01)]
        .drop_duplicates(subset=["run_chan", "untag_prec"])
        .groupby(["file_name", "channel", "silac_channel"])
        .size()
        .reset_index(name="Precursor_IDs")
        .sort_values("channel")
    )    
    logger.info("")
    logger.info("Number of precursors at 1% FDR:")
    logger.info(f"All Channels:{np.sum(df_counts_prec.Precursor_IDs)}")
    log_df(df_counts_prec)
    

    df_counts_prots = (
        df[(df["Protein_Qvalue"] < 0.01) & (df["is_decoy"] == False) & (df["Qvalue"] < 0.01)]
        .drop_duplicates(subset=["run_chan", "protein"])
        .groupby(["file_name", "channel", "silac_channel"])
        .size()
        .reset_index(name="Protein_IDs")
        .sort_values("channel")
        )
    logger.info("")
    logger.info("Number of proteins at 1% FDR:")
    logger.info(f"All Channels:{np.sum(df_counts_prots.Protein_IDs)}")
    log_df(df_counts_prots)

    if results_folder is not None:
        with open(results_folder+'/Summary.txt', 'a') as f:
            print("Number of precursors at 1% FDR:", file=f)
            print("All Channels:",np.sum(df_counts_prec.Precursor_IDs), file=f)
            print(df_counts_prec.to_string(index=False), file=f)
            
            print("\nNumber of proteins at 1% FDR:", file=f)
            print("All Channels:",np.sum(df_counts_prots.Protein_IDs), file=f)
            print(df_counts_prots.to_string(index=False), file=f)
    
    # if config.args.plexDIA:
    #     if config.args.timeplex:
    #         df["BestChannel_Protein_Qvalue"] = df.groupby(["time_channel", "protein", "decoy"])["Protein_Qvalue"].transform("min")
    #     else:
    #         df["BestChannel_Protein_Qvalue"] = df.groupby(["file_name", "protein", "decoy"])["Protein_Qvalue"].transform("min")

    if config.args.plexDIA:
        logger.info("")
        logger.info("After plexDIA identification propagation based on best channel Q-value:")
        
        # Compute number of precursor IDs at 1% FDR
        df_counts_prec = (
            df[(df["is_decoy"] == False) & (df["BestChannel_Qvalue"] < 0.01)]
            .drop_duplicates(subset=["run_chan", "untag_prec"])
            .groupby(["file_name", "channel", "silac_channel"])
            .size()
            .reset_index(name="Precursor_IDs")
            .sort_values("channel")
        )
        
        # Print precursor ID counts
        logger.info("")
        logger.info("Number of precursors at 1% FDR (best channel):")
        logger.info(f"All Channels:{np.sum(df_counts_prec.Precursor_IDs)}")
        log_df(df_counts_prec)
        
        # Compute number of protein IDs at 1% FDR
        df_counts_prots = (
            df[(df["Protein_Qvalue"] < 0.01) & (df["is_decoy"] == False) & (df["BestChannel_Qvalue"] < 0.01)]
            .drop_duplicates(subset=["run_chan", "protein"])
            .groupby(["file_name", "channel", "silac_channel"])
            .size()
            .reset_index(name="Protein_IDs")
            .sort_values("channel")
        )
        
        # Print protein ID counts
        logger.info("")
        logger.info("Number of proteins at 1% FDR (best channel):")
        logger.info(f"All Channels:{np.sum(df_counts_prots.Protein_IDs)}")
        log_df(df_counts_prots)
        
        if results_folder is not None:
            with open(results_folder+'/Summary.txt', 'a') as f:
                print("Number of precursors at 1% FDR (best channel):", file=f)
                print("All Channels:",np.sum(df_counts_prec.Precursor_IDs), file=f)
                print(df_counts_prec.to_string(index=False), file=f)
                
                print("\nNumber of proteins at 1% FDR (best channel):", file=f)
                print("All Channels:",np.sum(df_counts_prots.Protein_IDs), file=f)
                print(df_counts_prots.to_string(index=False), file=f)


    return df

def _plex_set_key(fdc, timeplex):
    """Grouping key for the cross-channel features: one plexDIA set.

    A plexDIA set is the group of channels that are multiplexed *together in one
    acquisition* — the mass-tag (mTRAQ/SILAC) channels of a given precursor, which
    are co-isolated and fragmented in the same scan. ``untag_prec`` alone is not
    that set on a timeplex run: the same precursor also appears once per TIME
    channel, and those are separate injections. Grouping on ``untag_prec`` therefore
    pools a precursor with copies of itself from other acquisitions, so
    ``median_*``/``diff_*_from_median`` measure run-to-run variation as much as
    channel-to-channel, and ``frac_shared_intensity`` counts fragment m/z overlap
    with channels that were never co-isolated (which is trivially near-total, since
    it is largely the same precursor).

    Appending ``time_channel`` scopes the key to a single plexDIA set. Off timeplex
    (or when the column is absent) there is only one acquisition and the key is
    ``untag_prec`` unchanged.

    Built as a string so that a null ``time_channel`` becomes its own stable group
    rather than being silently dropped by pandas' groupby NaN handling.
    """
    key = fdc["untag_prec"].astype(str)
    if timeplex and "time_channel" in fdc.columns:
        key = key + "|tc" + fdc["time_channel"].astype(str)
    return key


def add_median_based_features(df, metric_columns, group_col="untag_prec", count_col="channels_matched", verbose=True):
    """
    Calculate median-based features for specified metrics across groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the metric columns
    metric_columns : list
        List of column names to calculate medians and differences for
    group_col : str, default="untag_prec"
        Column to group by for median calculations
    count_col : str, default="channels_matched"
        Column indicating how many channels each group has
    verbose : bool, default=True
        Whether to print summary statistics
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added median and difference columns
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    if verbose:
        logger.debug(f"Adding median-based features for {len(metric_columns)} metrics...")
    
    for metric_col in metric_columns:
        # Calculate median for each group
        col_name = f"median_{metric_col}"
        result_df[col_name] = result_df.groupby(group_col)[metric_col].transform("median")
        
        # Set to NA for single-channel entries
        result_df.loc[result_df[count_col] == 1, col_name] = pd.NA
        
        # Calculate difference from median
        diff_col = f"diff_{metric_col}_from_median"
        result_df[diff_col] = result_df[metric_col] - result_df[col_name]
        
        # Fill NA with mean of non-NA values
        mean_val = result_df[diff_col].mean()
        result_df[diff_col] = result_df[diff_col].fillna(mean_val)
        
        if verbose:
            logger.debug(f"  Added {diff_col} (mean for NA values: {mean_val:.5f})")
            logger.debug(f"  Summary stats: min={result_df[diff_col].min():.5f}, max={result_df[diff_col].max():.5f}, mean={result_df[diff_col].mean():.5f}")
    
    return result_df


def process_data(file,spectra,library,mass_tag=None,timeplex=False,SILAC=None,elution_fwhm=None,vote_sigma=1.0):

    # results_folder = os.path.dirname(file)
    results_folder = os.path.dirname(os.path.dirname(file))
    mz_ppm = config.opt_ms1_tol
    rt_tol = config.opt_rt_tol

    # After loading data and adding basic features
    _glp = get_large_prec(file, condense_output=False, timeplex=timeplex)
    # Tests monkey-patch ``get_large_prec`` to return a legacy 3-tuple; the
    # production path returns a 4-tuple with a separate polars list-cols frame.
    if len(_glp) == 4:
        lp, fdc, dc, fdc_list_cols = _glp
    else:
        lp, fdc, dc = _glp
        fdc_list_cols = None

    # Add standard features
    fdc["stripped_seq"] = np.array([re.sub("Decoy_","",re.sub("\(.*?\)","",i)) for i in fdc["seq"]])
    fdc["pep_len"] = [len(re.findall("([A-Z](?:\(.*?\))?)",re.sub("Decoy","",i))) for i in fdc["stripped_seq"]]
    fdc["sq_rt_error"] = np.power(fdc["rt_error"],2)
    fdc["sq_mz_error"] = np.power(fdc["mz_error"],2)

    # RT error standardised against the spread of the TIME CHANNEL it came from.
    # Each time channel gets its own RT model and its own residual sigma (they
    # differ by ~25% within a run), so a raw rt_error of 0.3 min is ordinary in a
    # wide channel and a 2-sigma outlier in a narrow one. Scoring on rt_error
    # alone cannot tell those apart; this makes the comparison per channel.
    _sig = getattr(config, "rt_narrow_sigmas", None)
    if not _sig:
        try:
            import json
            with open(os.path.join(results_folder, "first_search",
                                   "rt_channel_sigmas.json")) as _fh:
                _sig = json.load(_fh).get("narrow_sigmas")
        except Exception:
            _sig = None
    if _sig and "time_channel" in fdc.columns and "rt_error" in fdc.columns:
        _s = np.asarray(_sig, dtype=float)
        _s[~np.isfinite(_s) | (_s <= 0)] = np.nan
        _tc = pd.to_numeric(fdc["time_channel"], errors="coerce").to_numpy()
        _ok = np.isfinite(_tc) & (_tc >= 0) & (_tc < len(_s))
        _den = np.full(len(fdc), np.nan)
        _den[_ok] = _s[_tc[_ok].astype(int)]
        # fall back to the pooled spread where a channel sigma is unusable, so the
        # column is never silently all-NaN for a run
        _fallback = np.nanmedian(_s) if np.isfinite(_s).any() else np.nan
        _den = np.where(np.isfinite(_den), _den, _fallback)
        fdc["rt_error_z_time_channel"] = fdc["rt_error"].to_numpy() / _den
        fdc["abs_rt_error_z_time_channel"] = np.abs(fdc["rt_error_z_time_channel"])
        logger.info(f"per-time-channel RT z-score added (sigmas {[round(float(x),3) for x in _s]})")

    # Handle untag_seq
    if mass_tag and SILAC:
        fdc["untag_seq"] = [re.sub(f"(\({SILAC.name}-\d+\))?","",re.sub(f"(\({mass_tag.name}-\d+\))?","",peptide)) for peptide in fdc["seq"]]
    elif mass_tag:
        fdc["untag_seq"] = [re.sub(f"(\({mass_tag.name}-\d+\))?","",peptide) for peptide in fdc["seq"]]
    elif SILAC:
        fdc["untag_seq"] = [re.sub(f"(\({SILAC.name}-\d+\))?","",peptide) for peptide in fdc["seq"]]
    else:
        fdc["untag_seq"] = fdc["seq"]
    #print(fdc.columns)  # Ensure 'seq' is in fdc

    # Add untag_prec and channels_matched
    fdc["untag_prec"] = ["_".join([i[0],str(int(i[1]))]) for i in zip(fdc["untag_seq"],fdc["z"])]
    
    # The cross-channel features below compare a precursor against the other
    # channels of its own plexDIA set -- the channels it was co-isolated with --
    # and NOT against its copies in other timeplex time channels, which are
    # separate acquisitions. See _plex_set_key.
    fdc[PLEX_SET_COL] = _plex_set_key(fdc, timeplex)

    channel_matches_counts = fdc[PLEX_SET_COL].value_counts()
    channel_matches_counts_dict = {i:j for i,j in zip(channel_matches_counts.index,channel_matches_counts)}
    fdc["channels_matched"] = [channel_matches_counts_dict[i] for i in fdc[PLEX_SET_COL]]

    # Use the helper function to add median-based features
    metrics_to_process = ["gof_stats", "scribe_scores", "max_matched_residuals", "manhattan_distances"]
    fdc = add_median_based_features(fdc, metrics_to_process, group_col=PLEX_SET_COL)

    # Compute frac_shared_intensity: fraction of library intensity from fragments
    # shared across the channels of the same plexDIA set
    if fdc_list_cols is not None:
        fdc["frac_shared_intensity"] = _compute_frac_shared_intensity_from_polars(
            fdc, fdc_list_cols, mz_tol=(config.args.ppm * 1e-6), group_col=PLEX_SET_COL
        )
    else:
        fdc["frac_shared_intensity"] = _compute_frac_shared_intensity(
            fdc, mz_tol=(config.args.ppm * 1e-6), group_col=PLEX_SET_COL
        )

    if timeplex:
        if mass_tag:
            tag_name = mass_tag.name
            tag_channel = [re.findall(f"{tag_name}-(\d+)",i) for i in fdc.seq]
            fdc["channel"] = [str(int(t))+"_"+i[0] if len(i)>0 else str(int(t)) for i,t in zip(tag_channel,fdc.time_channel)]
        else:
            fdc["channel"] = fdc["time_channel"]
            
    elif mass_tag:
        tag_name = mass_tag.name
        ## mTRAQ label
        tag_channel = [re.findall(f"{tag_name}-(\d+)",i) for i in fdc.seq]
        fdc["channel"] = [int(i[0]) if len(i)>0 else np.nan for i in tag_channel]

    else: 
        fdc["channel"] = 0 #if LF
        
    if SILAC is not None:
        silac_channel = [re.findall(f"{SILAC.name}-(\d+)",i) for i in fdc.seq]
        fdc["silac_channel"] = [int(i[0]) if len(i)>0 else np.nan for i in silac_channel] ### Note: This needs to change for multichannel SILAC
    else:
        fdc["silac_channel"] = np.nan 
        
    #this was previously in ms1_quant function.. we need it for the target/decoy classification
    # frag_errors stored as list columns in parquet — convert to numpy arrays
    if fdc_list_cols is not None:
        _med_pd = _compute_med_frag_error_from_polars(fdc_list_cols)
        fdc = fdc.merge(_med_pd, on="__fdc_idx", how="left")
        del _med_pd
    else:
        frag_errors = [np.array(x, dtype=float) if x is not None and len(x) > 0 else np.array([]) for x in fdc.frag_errors]
        non_empty = [i for i in frag_errors if len(i) > 0]
        median = np.median(np.concatenate(non_empty)) if non_empty else 0.0
        fdc["med_frag_error"] = [np.median(np.abs(median-i)) if len(i) > 0 else np.nan for i in frag_errors]

    # Signal-weighted mean fragment length. Each matched fragment contributes
    # its length in residues weighted by its share of the precursor's total
    # observed MS2 intensity, so a precursor whose signal sits in long fragments
    # scores high and one carrying it in short ions scores low. The _frac form
    # expresses that as a fraction of the precursor's own length, which makes it
    # comparable between a 9-mer and a 25-mer.
    if fdc_list_cols is not None:
        _wfl_pd = _compute_wtd_frag_len_from_polars(fdc_list_cols)
        fdc = fdc.merge(_wfl_pd, on="__fdc_idx", how="left")
        del _wfl_pd
    elif "frag_names" in fdc.columns and "obs_int" in fdc.columns:
        fdc["wtd_frag_len"] = _compute_wtd_frag_len(fdc["frag_names"], fdc["obs_int"])
    else:
        fdc["wtd_frag_len"] = np.nan
    fdc["wtd_frag_len_frac"] = fdc["wtd_frag_len"] / fdc["pep_len"]

    # Fragment-ion correlation features (pairwise Pearson across MS2 scans)
    from src.fragment_correlation import compute_fragment_correlations
    corr_features = compute_fragment_correlations(
        spectra=spectra,
        library=library,
        fdc=fdc,
        fwhm=elution_fwhm,
        mz_tol=(config.args.ppm * 1e-6),
        timeplex=timeplex,
    )
    for col in corr_features.columns:
        fdc[col] = corr_features[col].values

    # Cross-channel RT-residual consistency feature: SD of the per-PSM RT residual
    # (rt_error = observed - predicted) for each peptide, pooled across ALL timeplex
    # channels AND charge states (group = tag-stripped sequence). A peptide seen
    # consistently across channels/charges has a tight residual spread (small SD); a
    # spurious ID scatters. Grouped by is_decoy as well so targets and decoys never mix.
    # Guard: groups with < 2 finite residuals can't have a dispersion -> sentinel -1.
    if "untag_seq" in fdc.columns and "rt_error" in fdc.columns:
        _grp = fdc.groupby(["untag_seq", "is_decoy"])["rt_error"]
        _sd = _grp.transform("std")              # ddof=1; NaN for n<2, skips NaN residuals
        _n = _grp.transform("count")             # count of finite residuals in the group
        fdc["rt_resid_xchannel_sd"] = np.where(_n.to_numpy() >= 2,
                                               _sd.to_numpy(), -1.0)
    else:
        fdc["rt_resid_xchannel_sd"] = -1.0

    # Channel-unique MS2 refit + cross-channel empirical-library features.
    #
    # Deliberately BEFORE scoring and with no q-value gate.  These are scoring
    # features, and a column that exists only for rows already past 1% FDR
    # predicts the label outright.  It also produces the uncompressed
    # coeff_ChannelUnique that the RT normalizer prefers as its MS2 primary.
    #
    # The per-fragment list columns are still held out of fdc at this point, so
    # this joins them back one precursor-chunk at a time rather than merging the
    # whole frame -- the slim-frame memory guard is the reason they were held
    # back, and undoing it is what previously ran large searches out of RAM.
    if getattr(config.args, "ms2_prescoring", True):
        from src.ms2_prescoring import add_prescoring_ms2
        fdc = add_prescoring_ms2(
            fdc, fdc_list_cols=fdc_list_cols, spectra=spectra,
            plexDIA=bool(config.args.plexDIA or mass_tag is not None),
            ppm=float(config.args.ppm),
            n_chunks=int(getattr(config.args, "ms2_prescoring_chunks", 12)),
            want_areas=bool(getattr(config.args, "ms2_areas", True)),
            n_adjacent=int(getattr(config.args,
                                   "ms2_area_adjacent_scans", 1)),
            shared_frag_mode=str(getattr(config.args, "ms2_shared_frags",
                                         "proportional")),
            tag=mass_tag,
            # Always on: measured on JD0588 in a single-variable production
            # A/B (baseline_run vs area_run, one flag apart) it cuts
            # sample-level mean |error| 0.185 -> 0.134 and lifts the
            # compression slope 0.841 -> 0.902.  Not exposed, because there is
            # no configuration in which turning it off was better.
            isotope_correct=True,
            n_iso=int(getattr(config.args, "ms2_channel_unique_isotope_n", 5)),
            purity_diag=bool(getattr(config.args,
                                     "ms2_fragment_purity_diag", False)),
            min_siblings=int(getattr(config.args,
                                     "ms2_channel_unique_min_siblings", 0)),
            weighted_apex=bool(getattr(config.args,
                                       "ms2_weighted_common_apex", True)),
            )

    fdx = score_precursors(fdc.reset_index(drop=True), config.score_model, config.fdr_threshold, folder=results_folder)

    fdx['PredVal'] = fdx['PredVal'].fillna(0)
    fdx['Qvalue'] = fdx['Qvalue'].fillna(1)


    # if config.args.plexDIA:
    #     if config.args.timeplex:
    #         fdx["BestChannel_Qvalue"] = fdx.groupby(["time_channel", "untag_prec", "decoy"])["Qvalue"].transform("min") #within a plexDIA set for each timechannel
    #     else:
    #         fdx["BestChannel_Qvalue"] = fdx.groupby(["file_name", "untag_prec", "decoy"])["Qvalue"].transform("min") #within a plexDIA set
    
    if config.args.plexDIA or config.args.timeplex:
        fdx["BestChannel_Qvalue"] = fdx.groupby(["file_name", "untag_prec", "is_decoy"])["Qvalue"].transform("min") #within a run
    else:
        fdx["BestChannel_Qvalue"] = fdx["Qvalue"] #applies to no plex

    # Best q-value across a plexDIA set but WITHIN one time channel.
    # BestChannel_Qvalue above minimises over the whole run, so it collapses the
    # time dimension too: a precursor found confidently in T0 is reported at that
    # same q-value in every other time channel, whether or not any evidence for it
    # exists there. That makes it unusable for per-time-channel counting and for
    # carryover measurements. This column keeps the time channels independent.
    # Label-free: one row per (time_channel, untag_prec), so it equals Qvalue.
    # dropna=False so rows with a NaN time_channel keep a value instead of NaN.
    if "time_channel" in fdx.columns and {"untag_prec", "is_decoy"}.issubset(fdx.columns):
        _btp_keys = [c for c in ("file_name", "time_channel", "untag_prec", "is_decoy")
                     if c in fdx.columns]
        fdx["Best_tP_specific_channelQvalue"] = (
            fdx.groupby(_btp_keys, dropna=False)["Qvalue"].transform("min"))
    else:
        fdx["Best_tP_specific_channelQvalue"] = fdx["Qvalue"]

    
    fdx_quant = ms1_quant(fdx, lp, dc, mass_tag, SILAC, spectra, mz_ppm, rt_tol, timeplex, vote_sigma=vote_sigma)

    fdx_quant["last_aa"] = [i[-1] for i in fdx_quant["stripped_seq"]]
    fdx_quant["seq_len"] = [len(i) for i in fdx_quant["stripped_seq"]]
    
    # have possible reannotate woth fasta here
    # fdx["org"] = np.array([";".join(orgs[[i in all_fasta_seqs[j] for j in range(3)]]) for i in fdx["stripped_seq"]])
    fdx_quant = compute_protein_FDR(fdx_quant,results_folder=results_folder)

    # Re-attach the list columns we held back from fdc through the heavy
    # in-memory phase. Done here, just before the CSV write, so the merge
    # only happens once and the pipeline ran on a slim frame until now.
    if fdc_list_cols is not None and "__fdc_idx" in fdx_quant.columns:
        _list_cols_pd = fdc_list_cols.to_pandas()
        fdx_quant = fdx_quant.merge(_list_cols_pd, on="__fdc_idx", how="left")
        del _list_cols_pd
        import gc as _gc_attach
        _gc_attach.collect()
    if "__fdc_idx" in fdx_quant.columns:
        fdx_quant = fdx_quant.drop(columns=["__fdc_idx"])

    # ---- quantitation add-ons ------------------------------------------------
    # Placed here, after the list columns are back and before the write, so the
    # normalizer can reach the per-fragment intensities as well as the scalars,
    # and so nothing upstream (scoring, FDR, protein rollup) ever sees an
    # adjusted value.

    # MS2 quant from fragments that can be assigned to a single plexDIA channel.
    # y-ions of an R-terminating peptide carry no mTRAQ tag, so they land at the
    # same m/z in every channel and their observed intensity is a superposition.
    # Skipped when the pre-scoring pass already produced the column: it fits the
    # same quantity from the same spectra, ungated, and re-running it here would
    # both cost a second pass and overwrite the ungated values with gated ones.
    if (getattr(config.args, "ms2_channel_unique", True)
            and "coeff_ChannelUnique" not in fdx_quant.columns):
        from src.ms2_unique_quant import add_channel_unique_coeff
        fdx_quant = add_channel_unique_coeff(
            fdx_quant,
            spectra=spectra,
            plexDIA=bool(config.args.plexDIA or mass_tag is not None),
            ppm=float(config.args.ppm),
            q_gate=float(getattr(config.args, "ms2_channel_unique_qvalue",
                                 0.01)),
            tag=mass_tag,
            # Always on: measured on JD0588 in a single-variable production
            # A/B (baseline_run vs area_run, one flag apart) it cuts
            # sample-level mean |error| 0.185 -> 0.134 and lifts the
            # compression slope 0.841 -> 0.902.  Not exposed, because there is
            # no configuration in which turning it off was better.
            isotope_correct=True,
            n_iso=int(getattr(config.args, "ms2_channel_unique_isotope_n", 5)))

    # RT-dependent quantitative normalization: one factor curve per time channel
    # for MS1 and another for MS2, applied to every reported intensity.
    if getattr(config.args, "rt_norm", True):
        from src.quant_normalization import (apply_rt_channel_normalization,
                                             config_from_args)
        _plot = (None if getattr(config.args, "no_rt_norm_plot", False)
                 else os.path.join(results_folder, "outputs", "rtnorm_qc.png"))
        fdx_quant = apply_rt_channel_normalization(
            fdx_quant,
            timeplex=timeplex,
            plexDIA=bool(config.args.plexDIA or mass_tag is not None
                         or SILAC is not None),
            cfg=config_from_args(config.args),
            spectra=spectra,
            inplace_cols=bool(getattr(config.args, "rt_norm_inplace", False)),
            plot_path=_plot,
        )


    logger.info("")
    logger.info(f"Saving Results to Folder - {os.path.abspath(results_folder)}")
    ## save to results folder
    fdx_quant.to_csv(results_folder+"/outputs/all_IDs.csv",index=False)
    filtered = fdx_quant[np.logical_and(~fdx_quant["is_decoy"],fdx_quant["BestChannel_Qvalue"] < config.fdr_threshold)].copy()

    # Decode packed int32 frag codes into "y5_1" / "b3-H2O_2" strings, only for
    # the FDR-passing rows since decoding is expensive. In-place reassignment
    # keeps frag_names in its parquet position adjacent to the other six
    # fragment list columns (frag_errors, frag_mz, frag_int, obs_int,
    # unique_frag_mz, unique_obs_int).
    if "frag_names" in filtered.columns:
        from src.utils.frag_encoding import decode_frag_names
        filtered["frag_names"] = [
            list(decode_frag_names(np.asarray(codes, dtype=np.int32)))
            if codes is not None and len(codes) > 0 else []
            for codes in filtered["frag_names"]
        ]
    filtered.to_csv(results_folder+"/filtered_IDs.csv",index=False)
    # fdx_quant[np.logical_and(~fdx_quant["is_decoy"],fdx_quant["BestChannel_Qvalue"]<config.fdr_threshold)].to_csv(results_folder+"/filtered_IDs.csv",index=False)

    ### select minimum columns for parquet
    parquet_columns = ["stripped_seq","z","untag_prec","file_name","channel","is_decoy","Qvalue", "Protein_Qvalue","PredVal",
                       "protein",'BestChannel_Qvalue', 'Best_tP_specific_channelQvalue',
                       'plex_Area', 'seq', 'silac_channel', 'untag_seq',"rt","mz",
                       # Quantities and their RT-normalized counterparts. Absent
                       # under --rt_norm_inplace (the base columns carry the
                       # normalized values then) and filtered out below either way.
                       'coeff', 'MS1_Area',
                       'coeff_ChannelUnique',
                       'plex_Area_rtnorm', 'MS1_Area_rtnorm',
                       'coeff_rtnorm',
                       'coeff_ChannelUnique_rtnorm',
                       'rtnorm_factor_MS1', 'rtnorm_factor_MS2', 'time_channel']
    parquet_columns = [i for i in parquet_columns if i in fdx_quant.columns]
    fdx_quant[parquet_columns].to_parquet(results_folder+"/outputs/all_IDs_filtered.parquet")

    # filtered IDs with parquet columns 
    filtered[parquet_columns].to_parquet(results_folder + "/filtered_IDs_parquet_columns.parquet", index=False)