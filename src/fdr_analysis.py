"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""



from src.utils.io.read_output import get_large_prec

import mokapot
from scipy import stats
from sklearn.metrics import auc

import numpy as np
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

from src import config
from src.logger import logger
from src.utils.misc_functions import p_result
from src.mass_tags import massTag
from pyteomics import mass



def area(x):max_idx = np.argmax(x);top_3 = x[np.maximum(0,max_idx-1):max_idx+2];return np.sum(top_3)#auc(range(len(top_3)),top_3)


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


def ms1_quant(dat,lp,dc,mass_tag,DIAspectra,mz_ppm,rt_tol,timeplex=False):
    # X = fdc.iloc[:,6:-5]
    fit_whole_MS1 = False
   
    logger.info("")
    logger.info("Performing MS1 Quantitation") 
    
    fdc = dat[dat["decoy"] == False].copy().reset_index(drop=True)  #remove decoys
    
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
                                timeplex=timeplex,
                                num_iso = config.num_iso_ms1,
                                num_iso_r = config.num_iso_r,
                                additional_scans = config.additional_scans,
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
                            fit_whole_MS1=fit_whole_MS1
                            )

    return dat



def process_ms1_quant(dat, fdc, all_keys, group_p_corrs, group_ms1_traces, group_ms2_traces, group_iso_ratios, group_keys, group_fitted, new_output_dict, fake_fdc_dict, fit_whole_MS1=False):
    
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
    #fdc["plex_Area"]=[area(list(map(float,fdc.plexfittrace.iloc[idx].split(";")))) for idx in range(len(fdc))]
    fdc["plex_Area"]=[area(list(map(float,fdc.plexfittrace.iloc[idx].split(";")))) if fdc.plexfittrace.iloc[idx] != '' else np.nan for idx in range(len(fdc))]
       


    
    
    fdc["ms1_cor"] = [i[0] for i in p_corrs]
    
    for idx in range(config.num_iso_r):
        iso_num = idx+1
        fdc[f"iso{iso_num}_cor"] = [i[iso_num] for i in p_corrs]
    # fdc["iso1_cor"] = [i[1] for i in p_corrs]
    # fdc["iso2_cor"] = [i[2] for i in p_corrs]
    
    fdc["traceproduct"] = np.log10(fdc["ms1_cor"]*fdc["iso1_cor"]*fdc["iso2_cor"]+1e-6)
    
    # fdc["MS1_is1cor"] = [stats.pearsonr(list(i[0].values())[:10], list(i[1].values())[:10]).statistic for i in ms1_traces]
    
    
    fdc["iso_cor"] = [i[0].statistic for i in iso_ratios]
    
    fdc["MS1_Int"] = [i[2][0] for i in iso_ratios]
    fdc["MS1_Int"] = [np.linalg.lstsq(np.array(i[1])[:,np.newaxis], i[2], rcond=-1)[0][0] for i in iso_ratios]
    
    # X[np.isnan(X)]=0 ## set nans to zero (mostly for r2 values)
    fdc["all_ms1_specs"] = [";".join(map(str,trace[0].keys())) for trace in ms1_traces]
    for i in range(config.num_iso_ms1):
        fdc[f"all_ms1_iso{i}vals"] = [";".join(map(str,trace[i].values())) for trace in ms1_traces]
    # fdc["ms2_trace"] = [";".join(map(str,trace.values())) for trace in ms2_traces]
    
    fdc["MS1_Area"]=[auc(list(map(float,fdc.all_ms1_specs.iloc[idx].split(";"))),list(map(float,fdc.all_ms1_iso0vals.iloc[idx].split(";")))) for idx in range(len(fdc))]


        # Define selected columns that we want to merge
    selected_cols = [
        "plexfitMS1", "plexfitMS1_p", "plexfittrace", "plexfit_ps",
        "plexfittrace_spec_all", "plexfittrace_all", "plexfittrace_ps_all",
        "plex_Area", "ms1_cor", "traceproduct", "iso_cor", "MS1_Int",
        "all_ms1_specs", "MS1_Area"
    ] + [f"all_ms1_iso{i}vals" for i in range(config.num_iso_ms1)]
    
    # Ensure we only select columns that actually exist in fdc
    existing_cols = [col for col in selected_cols if col in fdc.columns]
    
    # Perform the merge safely
    dat = dat.merge(fdc[["untag_prec", "channel"] + existing_cols], how="left", on=["untag_prec", "channel"]).fillna(0)


    return dat



def score_precursors(fdc, model_type="rf", fdr_t=0.01, folder=None):
    """
    Score PSMs using Mokapot's semi-supervised learning (Percolator algorithm).

    Parameters
    ----------
    fdc : pandas.DataFrame
        All PSMs identified. Must include 'decoy', 'stripped_seq', 'protein',
        and feature columns for rescoring.
    model_type : string
        Ignored (kept for API compatibility). Mokapot always uses Percolator SVM.
    fdr_t : float
        False discovery rate threshold (default 0.01).
    folder : str, optional
        Folder path for saving plots.

    Returns
    -------
    fdc : pandas.DataFrame
        Updated dataframe with PredVal and Qvalue columns added.
    """
    logger.info("Scoring IDs with Mokapot")

    # Create working copy with required columns for Mokapot
    df = fdc.copy()
    df["target"] = ~df["decoy"]  # Mokapot expects target=True for targets
    df["psm_id"] = df.index.astype(str)  # Unique identifier for merging results

    # Columns to exclude from features (non-numeric or metadata columns)
    drop_columns = {
        'spec_id', 'Ms1_spec_id', 'seq', 'window_mz', 'frag_names', 'frag_errors',
        'frag_mz', 'frag_int', 'obs_int', 'stripped_seq', 'untag_seq', 'decoy',
        'all_ms1_specs', 'all_ms1_iso0vals', 'all_ms1_iso1vals', 'all_ms1_iso2vals',
        'all_ms1_iso3vals', 'all_ms1_iso4vals', 'all_ms1_iso5vals', 'all_ms1_iso6vals',
        'all_ms1_iso7vals', 'plexfittrace', 'plexfit_ps', 'untag_prec',
        'plexfittrace_spec_all', 'plexfittrace_all', 'plexfittrace_ps_all',
        'unique_frag_mz', 'channels_matched', 'unique_obs_int', 'MS1_Int',
        'MS1_Area', 'iso_cor', 'cosine', 'traceproduct', 'iso1_cor', 'iso2_cor',
        'ms1_cor', 'plexfitMS1', 'plexfitMS1_p', 'plex_Area', 'channel',
        'time_channel', 'file_name', 'protein', 'target', 'psm_id'
    }

    # Get initial feature columns (numeric only)
    feature_cols = [c for c in df.columns if c not in drop_columns]

    # Clean feature values and identify problematic columns
    zero_var_cols = []
    # Features with extreme ranges that benefit from log transformation
    log_transform_cols = ['coeff', 'frac_int_uniq_pred', 'smoothness', 'hyperscore', 'frac_int_uniq']
    # Debug/duplicate columns to exclude
    exclude_cols = {'debug_window_value'}  # This is a duplicate of manhattan_distances

    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            # Replace inf/nan with 0
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

            # Check for zero variance (useless for discrimination)
            col_std = df[col].std()
            if col_std == 0 or np.isnan(col_std):
                zero_var_cols.append(col)

            # Log-transform features with extreme ranges (add 1 to handle zeros)
            elif col in log_transform_cols and col in df.columns:
                # Use log1p for positive values, handle negative values separately
                if df[col].min() >= 0:
                    df[col] = np.log1p(df[col])
                else:
                    # For features that can be negative, use signed log transform
                    df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    # Remove zero-variance and excluded columns
    cols_to_remove = set(zero_var_cols) | exclude_cols
    if cols_to_remove:
        logger.info(f"Removing {len(cols_to_remove)} problematic features: {sorted(cols_to_remove)}")
        feature_cols = [c for c in feature_cols if c not in cols_to_remove]

    logger.info(f"Using {len(feature_cols)} features for Mokapot scoring")

    # Write out Mokapot input for debugging
    debug_path = folder if folder else "."
    df.to_csv(f"{debug_path}/mokapot_input.tsv", sep="\t", index=False)
    with open(f"{debug_path}/mokapot_feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))
    logger.info(f"Wrote Mokapot input to {debug_path}/mokapot_input.tsv")

    # Create Mokapot dataset
    psms = mokapot.LinearPsmDataset(
        psms=df,
        target_column="target",
        spectrum_columns=["psm_id"],
        peptide_column="stripped_seq",
        protein_column="protein",
        feature_columns=feature_cols,
        copy_data=True
    )

    # Let Mokapot auto-select the best initial direction
    # This allows it to find the feature that best separates targets from decoys
    logger.info(f"Running Mokapot with {len(df)} PSMs ({df['target'].sum()} targets, {(~df['target']).sum()} decoys)")
    logger.info(f"Features: {feature_cols}")

    # Determine initial direction - prefer scribe_scores if available
    # scribe_scores is the neural network prediction (lower is better = negative direction)
    if "scribe_scores" in feature_cols:
        initial_direction = "scribe_scores"
        logger.info(f"Using initial direction: scribe_scores (lower is better)")
    elif "scribe_score" in feature_cols:
        initial_direction = "scribe_score"
        logger.info(f"Using initial direction: scribe_score (lower is better)")
    else:
        initial_direction = None
        logger.info("No scribe_score found, letting Mokapot auto-select direction")

    # Run Mokapot with Percolator model
    # brew() returns (models, scores) where scores is list of numpy arrays
    # Use 5 folds to match original RF, max_iter=10 for convergence
    # Use train_fdr=0.05 to be more permissive (similar to coeff > 1 approach)
    model = mokapot.PercolatorModel(
        direction=initial_direction,
        train_fdr=0.05,  # More permissive than 0.01 to include more positive examples
        max_iter=15,
    )
    models, scores = mokapot.brew([psms], model=model, folds=5)

    # Calculate q-values using TDC (target-decoy competition)
    # This bypasses mokapot's assign_confidence which has issues with PEP calculation
    mokapot_scores = scores[0]
    targets = df["target"].values

    # Log score statistics
    logger.info(f"Score range: [{mokapot_scores.min():.4f}, {mokapot_scores.max():.4f}]")
    logger.info(f"Mean target score: {mokapot_scores[targets].mean():.4f}")
    logger.info(f"Mean decoy score: {mokapot_scores[~targets].mean():.4f}")

    # Sort by score (descending - higher is better)
    score_order = np.argsort(-mokapot_scores)
    sorted_targets = targets[score_order]

    # Calculate FDR using standard TDC: FDR = decoys / targets
    cumsum_decoys = np.cumsum(~sorted_targets)
    cumsum_targets = np.cumsum(sorted_targets)
    cumsum_targets_safe = np.maximum(cumsum_targets, 1)  # Avoid division by zero
    fdr = cumsum_decoys / cumsum_targets_safe

    # Convert FDR to q-values (minimum FDR at this score or better)
    qvalues_sorted = np.minimum.accumulate(fdr[::-1])[::-1]

    # Map back to original order
    orig_order = np.argsort(score_order)
    qvalues = qvalues_sorted[orig_order]

    # Assign scores and q-values back to original dataframe
    fdc["PredVal"] = mokapot_scores
    fdc["Qvalue"] = qvalues

    logger.info(f"Mokapot complete: {(fdc['Qvalue'] < 0.01).sum()} PSMs at 1% FDR")

    # Generate diagnostic plots
    if folder:
        _plot_score_distributions(fdc, folder)

    return fdc


def _plot_score_distributions(fdc, folder):
    """Generate diagnostic plots for score distributions."""
    output = fdc["PredVal"].values
    above_t = fdc["Qvalue"] <= 0.01

    # Score distribution plot
    plt.figure()
    vals, bins, _ = plt.hist(output, 50, label="All")
    plt.hist(output[~fdc["decoy"]], bins, alpha=0.5, label="Targets")
    plt.hist(output[fdc["decoy"]], bins, alpha=0.5, label="Decoys")
    plt.legend()
    plt.title(f"Mokapot Score Distribution")
    if vals.max() > 0:
        plt.vlines(output[above_t].min() if above_t.any() else 0, 0, vals.max(), colors='r', linestyles='dashed')
    plt.savefig(folder + "/ModelScore.png", dpi=600, bbox_inches="tight")
    plt.close()

    # RT and m/z error plots
    for feat in ['rt_error', 'mz_error']:
        if feat not in fdc.columns:
            continue
        plt.figure()
        vals, bins, _ = plt.hist(fdc[feat].values, 40, label="All")
        plt.hist(fdc.loc[above_t, feat].values, bins, alpha=0.5, label="1% FDR")
        plt.hist(fdc.loc[~above_t & ~fdc["decoy"], feat].values, bins, alpha=0.5, label="Low Scoring")
        plt.hist(fdc.loc[fdc["decoy"], feat].values, bins, alpha=0.5, label="Decoy")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.title(f"Mokapot - {feat}")
        plt.legend()
        plt.savefig(folder + f"/{feat}.png", dpi=600, bbox_inches="tight")
        plt.close()


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

  
    df["run_chan"] = df["file_name"].astype(str) + df["channel"].astype(str)
    df_seqchargeqvals = df[df["Qvalue"] < 0.01].copy().reset_index(drop=True) #filter
    df_seqchargeqvals["maxPredval"] = df_seqchargeqvals.groupby(["protein", "decoy"])["PredVal"].transform("max")
    df_seqchargeqvals = df_seqchargeqvals.drop_duplicates(subset=["protein", "decoy"]).reset_index(drop=True)
    
    # Rank by descending maxPredval and compute accum_decoys & Protein_Qvalue
    df_seqchargeqvals = df_seqchargeqvals.sort_values(by="maxPredval", ascending=False).reset_index(drop=True)
    df_seqchargeqvals["prot_rank"] = df_seqchargeqvals.index + 1  # Equivalent to row_number()
    df_seqchargeqvals["accum_decoys"] = df_seqchargeqvals["decoy"].cumsum()
    df_seqchargeqvals["Protein_Qvalue"] = df_seqchargeqvals["accum_decoys"] / (~df_seqchargeqvals["decoy"]).cumsum()
    
    # Filter for non-decoy proteins and select distinct protein values
    df_seqchargeqvals_distinct = (
        df_seqchargeqvals[df_seqchargeqvals["decoy"] == False]
        .drop_duplicates(subset=["protein"])
        [["protein", "Protein_Qvalue"]]
    )
    
    df = df.drop(columns=["Protein_Qvalue"], errors="ignore")
    df = df.merge(df_seqchargeqvals_distinct, on="protein", how="left")
        
    df_counts_prec = (
        df[(df["decoy"] == False) & (df["Qvalue"] < 0.01)]
        .drop_duplicates(subset=["run_chan", "untag_prec"])
        .groupby(["file_name", "channel"])
        .size()
        .reset_index(name="Precursor_IDs")
        .sort_values("channel")
    )    
    logger.info("")
    logger.info("Number of precursors at 1% FDR:")
    logger.info(f"All Channels:{np.sum(df_counts_prec.Precursor_IDs)}")
    log_df(df_counts_prec)
    

    df_counts_prots = (
        df[(df["Protein_Qvalue"] < 0.01) & (df["decoy"] == False) & (df["Qvalue"] < 0.01)]
        .drop_duplicates(subset=["run_chan", "protein"])
        .groupby(["run_chan","channel"])
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
            df[(df["decoy"] == False) & (df["BestChannel_Qvalue"] < 0.01)]
            .drop_duplicates(subset=["run_chan", "untag_prec"])
            .groupby(["file_name", "channel"])
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
            df[(df["Protein_Qvalue"] < 0.01) & (df["decoy"] == False) & (df["BestChannel_Qvalue"] < 0.01)]
            .drop_duplicates(subset=["run_chan", "protein"])
            .groupby(["run_chan", "channel"])
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

def process_data(file,spectra,library,mass_tag=None,timeplex=False):
    
    results_folder = os.path.dirname(file)
    mz_ppm = config.opt_ms1_tol
    rt_tol = config.opt_rt_tol
    
    # After loading data and adding basic features
    lp,fdc,dc = get_large_prec(file,condense_output=False,timeplex=timeplex)
    
    # Add standard features
    fdc["stripped_seq"] = np.array([re.sub("Decoy_","",re.sub("\(.*?\)","",i)) for i in fdc["seq"]])
    fdc["pep_len"] = [len(re.findall("([A-Z](?:\(.*?\))?)",re.sub("Decoy","",i))) for i in fdc["stripped_seq"]]
    fdc["sq_rt_error"] = np.power(fdc["rt_error"],2)
    fdc["sq_mz_error"] = np.power(fdc["mz_error"],2)

    # Handle untag_seq
    if mass_tag:
        fdc["untag_seq"] = [re.sub(f"(\({mass_tag.name}-\d+\))?","",peptide) for peptide in fdc["seq"]]
    else:
        fdc["untag_seq"] = fdc["seq"]
    #print(fdc.columns)  # Ensure 'seq' is in fdc

    # Add untag_prec and channels_matched
    fdc["untag_prec"] = ["_".join([i[0],str(int(i[1]))]) for i in zip(fdc["untag_seq"],fdc["z"])]
    
    
    
    
    
    
    
    channel_matches_counts = fdc["untag_prec"].value_counts()
    channel_matches_counts_dict = {i:j for i,j in zip(channel_matches_counts.index,channel_matches_counts)}
    fdc["channels_matched"] = [channel_matches_counts_dict[i] for i in fdc["untag_prec"]]

    # Use the helper function to add median-based features
    metrics_to_process = ["gof_stats", "scribe_scores", "max_matched_residuals", "manhattan_distances"]
    fdc = add_median_based_features(fdc, metrics_to_process)

    if timeplex:
        if mass_tag:
            tag_name = mass_tag.name
            fdc["channel"] = [str(int(t))+"_"+re.findall(f"{tag_name}-(\d+)",i)[0] for i,t in zip(fdc.seq,fdc.time_channel)]
        else:
            fdc["channel"] = fdc["time_channel"]
            
    elif mass_tag:
        tag_name = mass_tag.name
        ## mTRAQ label
        fdc["channel"] = [int(re.findall(f"{tag_name}-(\d+)",i)[0]) for i in fdc.seq]

    else: 
        fdc["channel"] = 0 #if LF

    #this was previously in ms1_quant function.. we need it for the target/decoy classification
    frag_errors = [unstring_floats(mz) for mz in fdc.frag_errors]
    median  = np.median(np.concatenate([i for i in frag_errors]))
    fdc["med_frag_error"] = [np.median(np.abs(median-i)) for i in frag_errors]

    ## What precursors are labeled as decoys
    fdc["decoy"] = np.array(["Decoy" in i for i in fdc["seq"]])

    
    minfraclib_toscore = getattr(config.args, "score_lib_frac", 0) 
    fdx_toscore = fdc[fdc['frac_lib_int'].fillna(0) >= minfraclib_toscore].reset_index(drop=True)
    
    fin = score_precursors(fdx_toscore,config.score_model,config.fdr_threshold,folder=results_folder)
    new_columns = [col for col in fin.columns if col not in fdc.columns and col not in ["untag_prec", "channel"]]
    fdx = fdc.merge(fin[["untag_prec", "channel"] + new_columns], how="left", on=["untag_prec", "channel"])

    ##fill NA's appropriately
    fdx['PredVal'] = fdx['PredVal'].fillna(0)  
    fdx['Qvalue'] = fdx['Qvalue'].fillna(1)     


    # if config.args.plexDIA:
    #     if config.args.timeplex:
    #         fdx["BestChannel_Qvalue"] = fdx.groupby(["time_channel", "untag_prec", "decoy"])["Qvalue"].transform("min") #within a plexDIA set for each timechannel
    #     else:
    #         fdx["BestChannel_Qvalue"] = fdx.groupby(["file_name", "untag_prec", "decoy"])["Qvalue"].transform("min") #within a plexDIA set
    
    if config.args.plexDIA or config.args.timeplex:
        fdx["BestChannel_Qvalue"] = fdx.groupby(["file_name", "untag_prec", "decoy"])["Qvalue"].transform("min") #within a run
    else:
        fdx["BestChannel_Qvalue"] = fdx["Qvalue"] #applies to no plex

    
    fdx_quant = ms1_quant(fdx, lp, dc, mass_tag, spectra, mz_ppm, rt_tol, timeplex)


    fdx_quant["last_aa"] = [i[-1] for i in fdx_quant["stripped_seq"]]
    fdx_quant["seq_len"] = [len(i) for i in fdx_quant["stripped_seq"]]
    
    # have possible reannotate woth fasta here
    # fdx["org"] = np.array([";".join(orgs[[i in all_fasta_seqs[j] for j in range(3)]]) for i in fdx["stripped_seq"]])
    fdx_quant = compute_protein_FDR(fdx_quant,results_folder=results_folder)

    logger.info("")
    logger.info(f"Saving Results to Folder - {os.path.abspath(results_folder)}")
    ## save to results folder
    fdx_quant.to_csv(results_folder+"/all_IDs.csv",index=False)
    fdx_quant[np.logical_and(~fdx_quant["decoy"],fdx_quant["BestChannel_Qvalue"]<config.fdr_threshold)].to_csv(results_folder+"/filtered_IDs.csv",index=False)
    fdx_quant[["stripped_seq","z","untag_prec","file_name","channel","decoy","Qvalue", "Protein_Qvalue","PredVal","protein"]].to_parquet(results_folder+"/all_IDs_filtered.parquet")